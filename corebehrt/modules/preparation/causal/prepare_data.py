import logging
import os
from os.path import join

import pandas as pd
import numpy as np
from tqdm import tqdm

from corebehrt.azure.util.config import load_config
from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSURE, OUTCOME
from corebehrt.constants.causal.paths import EXPOSURES_FILE, INDEX_DATE_MATCHING_FILE
from corebehrt.constants.data import ABSPOS_COL, DEATH_CODE, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import (
    FOLLOW_UPS_FILE,
    INDEX_DATES_FILE,
    OUTCOMES_FILE,
    COHORT_CFG,
)
from corebehrt.functional.preparation.causal.utils import (
    get_group_dict,
    get_non_compliance_abspos,
)
from corebehrt.functional.preparation.causal.convert import abspos_to_binary_outcome
from corebehrt.functional.preparation.causal.extract import extract_death
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.causal.convert import (
    dataframe_to_causal_patient_list,
)
from corebehrt.functional.preparation.causal.follow_up import (
    prepare_follow_ups_simple,
    prepare_follow_ups_adjusted,
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
from corebehrt.modules.preparation.prepare_data import DatasetPreparer

logger = logging.getLogger(__name__)  # Get the logger for this module


# TODO: Add option to load test set only!
class CausalDatasetPreparer(DatasetPreparer):
    """
    Prepares and processes patient data for causal inference.
    The major difference to the DatasetPreparer is that it also assigns exposures to the patients.
    """

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
        exposure_cfg = self.cfg.exposure
        outcome_cfg = self.cfg.outcome
        paths_cfg = self.cfg.paths
        data_cfg = self.cfg.data

        pids = self.load_cohort(paths_cfg)
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
                df = self._cutoff_data(df, data_cfg.cutoff_date)
            # !TODO: if index date is the same for all patients, then we can censor here.
            self._check_sorted(df)
            batch_patient_list = dataframe_to_causal_patient_list(df)
            patient_list.extend(batch_patient_list)
        logger.info(f"Number of patients: {len(patient_list)}")
        data = CausalPatientDataset(patients=patient_list)

        # Loading and processing outcomes
        index_dates = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        exposures = pd.read_csv(join(paths_cfg.cohort, EXPOSURES_FILE))
        index_date_matching = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATE_MATCHING_FILE)
        )
        outcomes = pd.read_csv(paths_cfg.outcome)

        index_dates[PID_COL] = index_dates[PID_COL].astype(int)
        outcomes[PID_COL] = outcomes[PID_COL].astype(int)

        index_dates = filter_df_by_pids(index_dates, data.get_pids())
        exposures = filter_df_by_pids(exposures, data.get_pids())
        outcomes = filter_df_by_pids(outcomes, data.get_pids())

        deaths = data.process_in_parallel(
            extract_death, death_token=self.vocab[DEATH_CODE]
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
        binary_exposure = self._get_binary_exposure(
            exposures,
            index_dates,
            exposure_cfg.get("n_hours_start_follow_up", -1),
            exposure_cfg.get("n_hours_end_follow_up"),
            data_end,
        )

        binary_outcome, follow_ups = self._get_binary_outcome(
            index_dates,
            outcomes,
            outcome_cfg.get("n_hours_start_follow_up", 0),
            outcome_cfg.get("n_hours_end_follow_up", np.inf),
            outcome_cfg.get("n_hours_compliance", np.inf),
            index_date_matching=index_date_matching,
            deaths=deaths,
            exposures=exposures,
            data_end=data_end,  # Pass the actual data end time
        )

        logger.info("Assigning exposures and outcomes")
        data = data.assign_attributes(EXPOSURE, binary_exposure)
        data = data.assign_attributes(OUTCOME, binary_outcome)

        censor_dates = (
            index_dates.set_index(PID_COL)[ABSPOS_COL] + exposure_cfg.n_hours_censoring
        )
        self._validate_censoring(data.patients, censor_dates, logger)
        if "concept_pattern_hours_delay" in self.cfg:
            concept_id_to_delay = get_concept_id_to_delay(
                self.cfg.concept_pattern_hours_delay, self.vocab
            )
            data.patients = data.process_in_parallel(
                censor_patient_with_delays,
                censor_dates=censor_dates,
                predict_token_id=self.predict_token,
                concept_id_to_delay=concept_id_to_delay,
            )
        else:
            data.patients = data.process_in_parallel(
                censor_patient,
                censor_dates=censor_dates,
                predict_token_id=self.predict_token,
            )
        background_length = get_background_length(data, self.vocab)
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
            else get_non_priority_tokens(self.vocab, data_cfg.low_priority_prefixes)
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=self.vocab["[SEP]"],
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
        os.makedirs(self.processed_dir, exist_ok=True)
        save_vocabulary(self.vocab, self.processed_dir)
        data.save(self.processed_dir)
        exposures.to_csv(join(self.processed_dir, EXPOSURES_FILE), index=False)
        outcomes.to_csv(join(self.processed_dir, OUTCOMES_FILE), index=False)
        index_dates.to_csv(join(self.processed_dir, INDEX_DATES_FILE), index=False)

        follow_ups.to_csv(join(self.processed_dir, FOLLOW_UPS_FILE), index=False)
        return data

    @staticmethod
    def _get_binary_exposure(
        exposures: pd.DataFrame,
        index_dates: pd.DataFrame,
        n_hours_start_follow_up: int,
        n_hours_end_follow_up: int,
        data_end: pd.Timestamp,
    ) -> pd.Series:
        """
        Create binary exposure indicators for patients based on exposure events within follow-up periods.

        Since index dates were determined using exposure criteria, this method uses simple
        follow-up windows without additional adjustments for compliance or deaths.

        Args:
            exposures: DataFrame with columns 'subject_id', 'abspos' (exposure events)
            index_dates: DataFrame with columns 'subject_id', 'abspos' (index dates)
            n_hours_start_follow_up: Hours after index date to start follow-up
            n_hours_end_follow_up: Hours after index date to end follow-up

        Returns:
            pd.Series: Binary exposure indicator (1 if exposed during follow-up, 0 otherwise)
        """
        follow_ups = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )
        return abspos_to_binary_outcome(follow_ups, exposures)

    @staticmethod
    def _get_binary_outcome(
        index_dates: pd.DataFrame,
        outcomes: pd.DataFrame,
        n_hours_start_follow_up: int,
        n_hours_end_follow_up: int,
        n_hours_compliance: int,
        index_date_matching: pd.DataFrame,
        deaths: pd.Series,
        exposures: pd.DataFrame,
        data_end: pd.Timestamp,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Create binary outcome indicators for patients using adjusted follow-up periods.

        This method accounts for multiple censoring events:
        1. Death events (from deaths parameter)
        2. Non-compliance periods (last exposure + n_hours_compliance)
        3. End of follow-up periods

        The final follow-up period for each patient is the minimum of these three values.
        Within matched groups, all patients receive the minimum follow-up period of the group.

        Args:
            index_dates: DataFrame with columns 'subject_id', 'abspos' (index dates)
            outcomes: DataFrame with columns 'subject_id', 'abspos' (outcome events)
            n_hours_start_follow_up: Hours after index date to start follow-up
            n_hours_end_follow_up: Hours after index date to end follow-up
            n_hours_compliance: Hours to add to last exposure for non-compliance cutoff
            index_date_matching: DataFrame defining matched groups (control_subject_id, exposed_subject_id)
            deaths: Series mapping patient IDs to death times (NaN if no death)
            exposures: DataFrame with columns 'subject_id', 'abspos' (exposure events)

        Returns:
            tuple: (binary_outcomes, adjusted_follow_ups)
                - binary_outcomes: pd.Series with binary outcome indicators
                - adjusted_follow_ups: pd.DataFrame with final follow-up periods
        """

        index_date_matching = CausalDatasetPreparer._filter_index_date_matching(
            index_date_matching, index_dates
        )
        group_dict = get_group_dict(index_date_matching)
        non_compliance_abspos = get_non_compliance_abspos(exposures, n_hours_compliance)
        follow_ups = prepare_follow_ups_simple(
            index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
        )
        follow_ups = prepare_follow_ups_adjusted(
            follow_ups,
            non_compliance_abspos,
            deaths,
            group_dict,
        )
        return abspos_to_binary_outcome(follow_ups, outcomes), follow_ups

    @staticmethod
    def _filter_index_date_matching(
        index_date_matching: pd.DataFrame, index_dates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter index date matching to only include patients in index dates
        """
        return index_date_matching[
            index_date_matching[CONTROL_PID_COL].isin(index_dates[PID_COL].unique())
        ]
