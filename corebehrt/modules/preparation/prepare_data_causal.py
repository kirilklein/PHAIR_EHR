import logging
import os
from os.path import join

import pandas as pd
from tqdm import tqdm

from corebehrt.constants.causal.data import EXPOSURE, OUTCOME
from corebehrt.constants.causal.paths import EXPOSURES_FILE
from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, OUTCOMES_FILE
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.convert_causal import (
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
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.preparation.dataset_causal import CausalPatientDataset
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
        vocab = load_vocabulary(paths_cfg.tokenized)

        # Loading and processing outcomes
        exposures = pd.read_csv(paths_cfg.exposure)
        outcomes = pd.read_csv(paths_cfg.outcome)

        exposures[PID_COL] = exposures[PID_COL].astype(int)
        outcomes[PID_COL] = outcomes[PID_COL].astype(int)

        exposures = filter_df_by_pids(exposures, data.get_pids())
        outcomes = filter_df_by_pids(outcomes, data.get_pids())

        logger.info("Handling exposures and outcomes")
        # Outcome Handler now only needs to do 1 thing: if outcome is in follow up window 1 else 0
        binary_exposure = get_binary_outcomes(
            index_dates,
            exposures,
            exposure_cfg.get("n_hours_start_follow_up", 0),
            exposure_cfg.get("n_hours_end_follow_up", None),
        )

        binary_outcome = get_binary_outcomes(
            index_dates,
            outcomes,
            outcome_cfg.get("n_hours_start_follow_up", 0),
            outcome_cfg.get("n_hours_end_follow_up", None),
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
                self.cfg.concept_pattern_hours_delay, vocab
            )
            data.patients = data.process_in_parallel(
                censor_patient_with_delays,
                censor_dates=censor_dates,
                concept_id_to_delay=concept_id_to_delay,
            )
        else:
            data.patients = data.process_in_parallel(
                censor_patient, censor_dates=censor_dates
            )
        background_length = get_background_length(data, vocab)
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
            else get_non_priority_tokens(vocab, data_cfg.low_priority_prefixes)
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
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
        save_vocabulary(vocab, self.processed_dir)
        data.save(self.processed_dir)
        exposures.to_csv(join(self.processed_dir, EXPOSURES_FILE), index=False)
        outcomes.to_csv(join(self.processed_dir, OUTCOMES_FILE), index=False)
        index_dates.to_csv(join(self.processed_dir, INDEX_DATES_FILE), index=False)

        return data
