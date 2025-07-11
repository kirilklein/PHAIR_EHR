import logging
import os
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from corebehrt.azure.util.config import load_config
from corebehrt.constants.causal.data import EXPOSURE, OUTCOME
from corebehrt.constants.causal.paths import (
    BINARY_EXPOSURE_FILE,
    BINARY_OUTCOMES_FILE,
    EXPOSURES_FILE,
    INDEX_DATE_MATCHING_FILE,
)
from corebehrt.constants.data import ABSPOS_COL, DEATH_CODE, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import COHORT_CFG, FOLLOW_UPS_FILE, INDEX_DATES_FILE
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.causal.convert import (
    dataframe_to_causal_patient_list,
)
from corebehrt.functional.preparation.causal.extract import extract_death
from corebehrt.functional.preparation.causal.outcomes import (
    get_binary_exposure,
    get_binary_outcome,
)
from corebehrt.functional.preparation.filter import (
    censor_patient,
    censor_patient_with_delays,
    exclude_short_sequences,
)
from corebehrt.functional.preparation.truncate import truncate_patient
from corebehrt.functional.preparation.utils import (
    get_background_length,
    get_concept_id_to_delay,
    get_non_priority_tokens,
)
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module


class Artifacts:
    def __init__(
        self,
        data: CausalPatientDataset,
        exposures: pd.DataFrame,
        binary_outcomes: pd.DataFrame,
        binary_exposure: pd.Series,
        index_dates: pd.DataFrame,
        follow_ups: pd.DataFrame,
    ):
        self.data = data
        self.exposures = exposures
        self.binary_exposure = binary_exposure
        self.binary_outcomes = binary_outcomes
        self.index_dates = index_dates
        self.follow_ups = follow_ups


class CausalDatasetPreparer:
    """
    Prepares and processes patient data for causal inference.
    The major difference to the DatasetPreparer is that it also assigns exposures to the patients.
    """

    def __init__(self, cfg: Config):
        self.ds_preparer = DatasetPreparer(cfg)

    def prepare_finetune_data(self, mode="tuning") -> CausalPatientDataset:
        """
        Prepares and processes patient data for fine-tuning, including censoring, truncation, and outcome assignment.

        Loads patient data, exposures, outcomes, applies cohort and cutoff filtering, assigns binary exposures and outcomes,
            computes and validates censoring dates, applies censoring (optionally with concept-specific delays),
            excludes short sequences, truncates and normalizes patient records,
            and saves the processed dataset for downstream modeling.

        Args:
            mode: Specifies which data split to use for fine-tuning (default is "tuning").

        Returns:
            A PatientDataset object containing the processed and labeled patient data ready for fine-tuning.
        """
        exposure_cfg, outcome_cfg, paths_cfg, data_cfg = self._access_cfg_paths()

        pids = self.ds_preparer.load_cohort(paths_cfg)
        cohort_cfg = load_config(join(paths_cfg.cohort, COHORT_CFG))
        data = self.load_shards_into_patient_data(paths_cfg, data_cfg, pids, mode)

        # Load data and do some basic filtering
        exposures, index_date_matching, index_dates = self.load_cohort_data(
            paths_cfg.cohort
        )
        index_dates[PID_COL] = index_dates[PID_COL].astype(int)
        index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])
        # Load outcomes
        multitarget = paths_cfg.get("outcome_files", None) is not None
        if multitarget:
            outcomes = {}
            for outcome_name, outcome_file in paths_cfg.outcome_files.items():
                outcome_df = pd.read_csv(outcome_file)
                outcome_df[PID_COL] = outcome_df[PID_COL].astype(int)
                outcomes[outcome_name] = outcome_df
        else:
            outcome_df = pd.read_csv(paths_cfg.outcome)
            outcome_df[PID_COL] = outcome_df[PID_COL].astype(int)
            outcomes = {"outcome": outcome_df}

        # Include only relevant pids
        pids = data.get_pids()
        index_dates = filter_df_by_pids(index_dates, pids)
        exposures = filter_df_by_pids(exposures, pids)
        outcomes = {name: filter_df_by_pids(df, pids) for name, df in outcomes.items()}

        deaths = data.process_in_parallel(
            extract_death, death_token=self.ds_preparer.vocab[DEATH_CODE]
        )
        deaths = {patient.pid: death for patient, death in zip(data.patients, deaths)}

        logger.info("Handling exposures and outcomes")
        data_end = self.get_data_end(cohort_cfg)
        binary_exposure = get_binary_exposure(
            exposures,
            index_dates,
            exposure_cfg.get("n_hours_start_follow_up", -1),
            exposure_cfg.get("n_hours_end_follow_up"),
            data_end,
        )

        binary_outcomes = {}
        for outcome_name, outcome_df in outcomes.items():
            binary_outcomes[outcome_name], follow_ups = get_binary_outcome(
                index_dates,
                outcome_df,
                outcome_cfg.get("n_hours_start_follow_up", 0),
                outcome_cfg.get("n_hours_end_follow_up", np.inf),
                outcome_cfg.get("n_hours_compliance", np.inf),
                index_date_matching=index_date_matching,
                deaths=deaths,
                exposures=exposures,
                data_end=data_end,  # Pass the actual data end time
            )
        binary_outcomes = pd.DataFrame(binary_outcomes)
        logger.info("Assigning exposures and outcomes")
        data.assign_attributes(EXPOSURE, binary_exposure)
        if multitarget:
            data.assign_outcomes(binary_outcomes)
        else:
            data.assign_attributes(OUTCOME, binary_outcomes["outcome"])

        censor_dates = (
            index_dates.set_index(PID_COL)[ABSPOS_COL] + exposure_cfg.n_hours_censoring
        )
        self.ds_preparer._validate_censoring(data.patients, censor_dates, logger)
        if "concept_pattern_hours_delay" in self.ds_preparer.cfg:
            concept_id_to_delay = get_concept_id_to_delay(
                self.ds_preparer.cfg.concept_pattern_hours_delay, self.ds_preparer.vocab
            )
            data.patients = data.process_in_parallel(
                censor_patient_with_delays,
                censor_dates=censor_dates,
                predict_token_id=self.ds_preparer.predict_token,
                concept_id_to_delay=concept_id_to_delay,
            )
        else:
            data.patients = data.process_in_parallel(
                censor_patient,
                censor_dates=censor_dates,
                predict_token_id=self.ds_preparer.predict_token,
            )
        background_length = get_background_length(data, self.ds_preparer.vocab)
        # Exclude short sequences
        logger.info("Excluding short sequences")
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + background_length,
        )

        non_priority_tokens = (
            None
            if data_cfg.get("low_priority_prefixes", None) is None
            else get_non_priority_tokens(
                self.ds_preparer.vocab, data_cfg.low_priority_prefixes
            )
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=self.ds_preparer.vocab["[SEP]"],
            non_priority_tokens=non_priority_tokens,
        )

        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        # Check if max segment is larger than type_vocab b_size
        # TODO: pass pt_model_config and perform this check
        # max_segment(data, model_cfg.type_vocab_size)
        # Previously had issue with it
        logger.info(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )
        # save
        out_dir = join(self.ds_preparer.processed_dir, "causal")
        artifacts = Artifacts(
            data, exposures, binary_outcomes, binary_exposure, index_dates, follow_ups
        )
        self.save_artifacts(out_dir, artifacts)
        return data

    def _access_cfg_paths(self):
        exposure_cfg = self.ds_preparer.cfg.exposure
        outcome_cfg = self.ds_preparer.cfg.outcome
        paths_cfg = self.ds_preparer.cfg.paths
        data_cfg = self.ds_preparer.cfg.data
        return exposure_cfg, outcome_cfg, paths_cfg, data_cfg

    def load_shards_into_patient_data(
        self, paths_cfg, data_cfg, pids=None, mode="tuning"
    ) -> CausalPatientDataset:
        # Load tokenized data
        loader = ShardLoader(
            data_dir=paths_cfg.tokenized,
            splits=[f"features_{mode}"],
            patient_info_path=None,
        )
        patient_list = []
        for df, _ in tqdm(
            loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
        ):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            if data_cfg.get("cutoff_date"):
                df = self.ds_preparer._cutoff_data(df, data_cfg.cutoff_date)
            # !TODO: if index date is the same for all patients, then we can censor here.
            self.ds_preparer._check_sorted(df)
            batch_patient_list = dataframe_to_causal_patient_list(df)
            patient_list.extend(batch_patient_list)
        logger.info(f"Number of patients: {len(patient_list)}")
        data = CausalPatientDataset(patients=patient_list)
        return data

    @staticmethod
    def load_cohort_data(
        cohort_path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        exposures = pd.read_csv(join(cohort_path, EXPOSURES_FILE))
        index_date_matching = pd.read_csv(join(cohort_path, INDEX_DATE_MATCHING_FILE))
        index_dates = pd.read_csv(
            join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        return exposures, index_date_matching, index_dates

    @staticmethod
    def get_data_end(cohort_cfg: Config) -> pd.Timestamp:
        if hasattr(cohort_cfg, "time_windows") and hasattr(
            cohort_cfg.time_windows, "data_end"
        ):
            return pd.to_datetime(cohort_cfg.time_windows.data_end)
        else:
            logger.warning(
                "No data end time found in cohort configuration. Setting to today."
            )
            return pd.to_datetime("today")

    def save_artifacts(self, out_dir: str, artifacts: Artifacts):
        os.makedirs(out_dir, exist_ok=True)
        save_vocabulary(self.ds_preparer.vocab, out_dir)
        artifacts.data.save(out_dir)
        artifacts.exposures.to_csv(join(out_dir, EXPOSURES_FILE), index=False)
        artifacts.index_dates.to_csv(join(out_dir, INDEX_DATES_FILE), index=False)
        artifacts.follow_ups.to_csv(join(out_dir, FOLLOW_UPS_FILE), index=False)
        artifacts.binary_outcomes.to_csv(join(out_dir, BINARY_OUTCOMES_FILE))
        artifacts.binary_exposure.to_csv(join(out_dir, BINARY_EXPOSURE_FILE))
