import logging
import os
from os.path import join

import pandas as pd
import numpy as np
from tqdm import tqdm

from corebehrt.azure.util.config import load_config
from corebehrt.constants.causal.data import EXPOSURE
from corebehrt.constants.causal.paths import (
    EXPOSURES_FILE,
    INDEX_DATE_MATCHING_FILE,
    BINARY_OUTCOMES_FILE,
)
from corebehrt.constants.data import ABSPOS_COL, DEATH_CODE, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import (
    FOLLOW_UPS_FILE,
    INDEX_DATES_FILE,
    COHORT_CFG,
)
from corebehrt.functional.preparation.causal.extract import extract_death
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.causal.outcomes import (
    get_binary_exposure,
    get_binary_outcome,
)
from corebehrt.functional.preparation.causal.convert import (
    dataframe_to_causal_patient_list,
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
from corebehrt.functional.utils.time import (
    get_hours_since_epoch,
)
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.preparation.causal.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module


# TODO: Add option to load test set only!
class CausalDatasetPreparerMultitarget:
    """
    Prepares and processes patient data for causal inference with multiple outcomes.
    The major difference to the CausalDatasetPreparer is that it also assigns exposures to the patients.
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
        exposure_cfg = self.ds_preparer.cfg.exposure
        outcome_cfg = self.ds_preparer.cfg.outcome
        paths_cfg = self.ds_preparer.cfg.paths
        data_cfg = self.ds_preparer.cfg.data
        pids = self.ds_preparer.load_cohort(paths_cfg)
        cohort_cfg = load_config(join(paths_cfg.cohort, COHORT_CFG))
        # Load index dates and convert to abspos
        index_dates = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        index_dates[PID_COL] = index_dates[PID_COL].astype(int)
        index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])

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

        exposures = pd.read_csv(paths_cfg.exposure)
        index_date_matching = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATE_MATCHING_FILE)
        )

        outcome_dfs = {}
        for outcome_name, outcome_file in paths_cfg.outcome_files.items():
            outcome_dfs[outcome_name] = pd.read_csv(outcome_file)
        index_dates[PID_COL] = index_dates[PID_COL].astype(int)
        for outcome_name, outcome_df in outcome_dfs.items():
            outcome_df[PID_COL] = outcome_df[PID_COL].astype(int)

        index_dates = filter_df_by_pids(index_dates, data.get_pids())
        exposures = filter_df_by_pids(exposures, data.get_pids())
        for outcome_name, outcome_df in outcome_dfs.items():
            outcome_df = filter_df_by_pids(outcome_df, data.get_pids())

        deaths = data.process_in_parallel(
            extract_death, death_token=self.ds_preparer.vocab[DEATH_CODE]
        )
        deaths = {patient.pid: death for patient, death in zip(data.patients, deaths)}

        logger.info("Handling exposures and outcomes")
        # Get data end time from cohort configuration
        if hasattr(cohort_cfg, "time_windows") and hasattr(
            cohort_cfg.time_windows, "data_end"
        ):
            data_end = pd.to_datetime(cohort_cfg.time_windows.data_end)
        else:
            logger.warning(
                "No data end time found in cohort configuration. Setting to today."
            )
            data_end = pd.to_datetime("today")
        # Outcome Handler now only needs to do 1 thing: if outcome is in follow up window 1 else 0
        binary_exposure = get_binary_exposure(
            exposures,
            index_dates,
            exposure_cfg.get("n_hours_start_follow_up", -1),
            exposure_cfg.get("n_hours_end_follow_up"),
            data_end,
        )

        binary_outcomes = {}
        for outcome_name, outcome_df in outcome_dfs.items():
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
        binary_outcomes_df = pd.DataFrame(binary_outcomes)

        logger.info("Assigning exposures and outcomes")
        data = data.assign_attributes(EXPOSURE, binary_exposure)
        data.assign_outcomes(binary_outcomes)

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

        # Truncation
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
        out_dir = join(self.ds_preparer.processed_dir, "causal_multitarget")
        os.makedirs(out_dir, exist_ok=True)
        save_vocabulary(self.ds_preparer.vocab, out_dir)
        data.save(out_dir)
        exposures.to_csv(join(out_dir, EXPOSURES_FILE), index=False)

        index_dates.to_csv(join(out_dir, INDEX_DATES_FILE), index=False)
        binary_outcomes_df.to_csv(join(out_dir, BINARY_OUTCOMES_FILE))
        follow_ups.to_csv(join(out_dir, FOLLOW_UPS_FILE), index=False)
        return data
