import logging
import os
from dataclasses import dataclass
from os.path import join
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from corebehrt.azure.util.config import load_config
from corebehrt.constants.causal.data import EXPOSURE
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


@dataclass
class Artifacts:
    """A container for the output artifacts of the data preparation process."""

    data: CausalPatientDataset
    exposures: pd.DataFrame
    binary_outcomes: pd.DataFrame
    binary_exposure: pd.Series
    index_dates: pd.DataFrame
    follow_ups: pd.DataFrame
    vocabulary: dict


class CausalDatasetPreparer:
    """
    Prepares and processes patient data for causal inference.
    Differs from DatasetPreparer by also assigning exposures to patients.
    """

    def __init__(self, cfg: Config):
        self.ds_preparer = DatasetPreparer(cfg)
        self.exposure_cfg = cfg.exposure
        self.outcome_cfg = cfg.outcome
        self.paths_cfg = cfg.paths
        self.data_cfg = cfg.data
        self.cohort_cfg = load_config(join(self.paths_cfg.cohort, COHORT_CFG))
        self.vocabulary = self.ds_preparer.vocab
        self.min_instances_per_class = self.data_cfg.get("min_instances_per_class", 10)

    def prepare_finetune_data(self, mode: str = "tuning") -> CausalPatientDataset:
        """
        Prepares and processes patient data for fine-tuning.

        This method orchestrates the loading, filtering, labeling, censoring,
        and truncation of patient data to create a dataset ready for causal modeling.

        Args:
            mode: The data split to use (e.g., "tuning", "validation").

        Returns:
            A CausalPatientDataset object ready for fine-tuning.
        """
        # 1. Load and filter initial data
        pids = self.ds_preparer.load_cohort(self.paths_cfg)
        data = self.load_shards_into_patient_data(pids, mode)
        pids = data.get_pids()  # Use PIDs actually present in the data

        exposures, index_date_matching, index_dates = self.load_cohort_data(
            self.paths_cfg.cohort
        )
        outcomes = self._load_outcomes()

        exposures, index_dates, outcomes = self._filter_dataframes_by_pids(
            pids, exposures, index_dates, outcomes
        )

        # 2. Compute labels and outcomes
        index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])
        deaths = self._extract_deaths(data)

        binary_exposure, binary_outcomes, follow_ups = self._compute_binary_labels(
            exposures, outcomes, index_dates, index_date_matching, deaths
        )

        # 3. Assign labels to patient data
        self._assign_labels(data, binary_exposure, binary_outcomes)

        # 4. Censor, truncate, and normalize sequences
        self._censor_and_truncate_sequences(data, index_dates)
        data.patients = data.process_in_parallel(normalize_segments_for_patient)

        logger.info(
            f"Max segment length: {max(max(p.segments, default=0) for p in data.patients)}"
        )
        class_counts = binary_outcomes.drop(columns=PID_COL, errors="ignore").apply(
            pd.Series.value_counts
        )
        logger.info(
            "\nClass counts:\n" + class_counts.fillna(0).astype(int).to_string()
        )
        # 5. Save all generated artifacts
        artifacts = Artifacts(
            data=data,
            exposures=exposures,
            binary_outcomes=binary_outcomes,
            binary_exposure=binary_exposure,
            index_dates=index_dates,
            follow_ups=follow_ups,
            vocabulary=self.vocabulary,
        )
        self._save_artifacts(artifacts, self.paths_cfg.prepared_data)

        return data

    def _load_outcomes(self) -> Dict[str, pd.DataFrame]:
        """Loads single or multiple outcome files into a dictionary."""
        outcomes = {}
        for name, outcome_file in self.paths_cfg.outcome_files.items():
            df = pd.read_csv(outcome_file)
            df[PID_COL] = df[PID_COL].astype(int)
            outcomes[name] = df
        return outcomes

    def _filter_dataframes_by_pids(
        self,
        pids: List[int],
        exposures: pd.DataFrame,
        index_dates: pd.DataFrame,
        outcomes: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Filters all relevant dataframes by the provided list of patient IDs."""
        logger.info(f"Filtering dataframes for {len(pids)} patients.")
        index_dates = filter_df_by_pids(index_dates, pids)
        exposures = filter_df_by_pids(exposures, pids)
        filtered_outcomes = {
            name: filter_df_by_pids(df, pids) for name, df in outcomes.items()
        }
        return exposures, index_dates, filtered_outcomes

    def _extract_deaths(self, data: CausalPatientDataset) -> Dict[int, Any]:
        """Extracts death information for each patient."""
        deaths_list = data.process_in_parallel(
            extract_death, death_token=self.vocabulary[DEATH_CODE]
        )
        return {
            patient.pid: death for patient, death in zip(data.patients, deaths_list)
        }

    def _compute_binary_labels(
        self,
        exposures: pd.DataFrame,
        outcomes: Dict[str, pd.DataFrame],
        index_dates: pd.DataFrame,
        index_date_matching: pd.DataFrame,
        deaths: Dict[int, Any],
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Computes binary exposure and outcome labels."""
        logger.info("Handling exposures and outcomes")
        data_end = self.get_data_end(self.cohort_cfg)

        binary_exposure = get_binary_exposure(
            exposures,
            index_dates,
            self.exposure_cfg.get("n_hours_start_follow_up", -1),
            self.exposure_cfg.get("n_hours_end_follow_up"),
            data_end,
        )

        binary_outcomes = {}
        follow_ups = None
        min_instances_per_class = self.outcome_cfg.get("min_instances_per_class", 10)
        for outcome_name, outcome_df in outcomes.items():
            if outcome_df.empty:
                logger.warning(f"Outcome {outcome_name} has no data. Skipping.")
                continue
            binary_outcome, follow_ups = get_binary_outcome(
                index_dates,
                outcome_df,
                self.outcome_cfg.get("n_hours_start_follow_up", 0),
                self.outcome_cfg.get("n_hours_end_follow_up", np.inf),
                self.outcome_cfg.get("n_hours_compliance", np.inf),
                index_date_matching=index_date_matching,
                deaths=deaths,
                exposures=exposures,
                data_end=data_end,
            )
            counts = binary_outcome.value_counts()
            if len(counts) < 2 or counts.min() < min_instances_per_class:
                logger.warning(
                    f"Outcome {outcome_name} has a class with fewer than "
                    f"{min_instances_per_class} instances. Value counts: {counts.to_dict()}. Skipping."
                )
                continue
            binary_outcomes[outcome_name] = binary_outcome
        return binary_exposure, pd.DataFrame(binary_outcomes), follow_ups

    def _assign_labels(
        self,
        data: CausalPatientDataset,
        binary_exposure: pd.Series,
        binary_outcomes: pd.DataFrame,
    ):
        """Assigns computed labels to the CausalPatientDataset."""
        logger.info("Assigning exposures and outcomes")
        data.assign_attributes(EXPOSURE, binary_exposure)
        data.assign_outcomes(binary_outcomes)

    def _censor_and_truncate_sequences(
        self, data: CausalPatientDataset, index_dates: pd.DataFrame
    ):
        """Applies censoring, filters short sequences, and truncates."""
        # Censor sequences based on index dates
        censor_dates = (
            index_dates.set_index(PID_COL)[ABSPOS_COL]
            + self.exposure_cfg.n_hours_censoring
        )
        self.ds_preparer._validate_censoring(data.patients, censor_dates, logger)

        if "concept_pattern_hours_delay" in self.ds_preparer.cfg:
            concept_id_to_delay = get_concept_id_to_delay(
                self.ds_preparer.cfg.concept_pattern_hours_delay,
                self.vocabulary,
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

        # Exclude short sequences
        logger.info("Excluding short sequences")
        background_length = get_background_length(data, self.vocabulary)
        min_len = self.data_cfg.get("min_len", 1) + background_length
        data.patients = exclude_short_sequences(data.patients, min_len)

        # Truncate sequences
        non_priority_tokens = (
            get_non_priority_tokens(
                self.vocabulary, self.data_cfg.low_priority_prefixes
            )
            if self.data_cfg.get("low_priority_prefixes")
            else None
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=self.data_cfg.truncation_len,
            background_length=background_length,
            sep_token=self.vocabulary["[SEP]"],
            non_priority_tokens=non_priority_tokens,
        )

    def load_shards_into_patient_data(
        self, pids: List[int] = None, mode: str = "tuning"
    ) -> CausalPatientDataset:
        """Loads and processes data shards into a CausalPatientDataset."""
        loader = ShardLoader(
            data_dir=self.paths_cfg.tokenized,
            splits=[f"features_{mode}"],
            patient_info_path=None,
        )
        patient_list = []
        desc = f"Loading and processing '{mode}' shards"
        for df, _ in tqdm(loader(), desc=desc, file=TqdmToLogger(logger)):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            if self.data_cfg.get("cutoff_date"):
                df = self.ds_preparer._cutoff_data(df, self.data_cfg.cutoff_date)

            self.ds_preparer._check_sorted(df)
            patient_list.extend(dataframe_to_causal_patient_list(df))

        logger.info(f"Loaded {len(patient_list)} patients.")
        return CausalPatientDataset(patients=patient_list)

    @staticmethod
    def load_cohort_data(
        cohort_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads exposure and index date files."""
        exposures = pd.read_csv(join(cohort_path, EXPOSURES_FILE))
        index_date_matching = pd.read_csv(join(cohort_path, INDEX_DATE_MATCHING_FILE))
        index_dates = pd.read_csv(
            join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        return exposures, index_date_matching, index_dates

    @staticmethod
    def get_data_end(cohort_cfg: Config) -> pd.Timestamp:
        """Retrieves the data end timestamp from the configuration."""
        try:
            return pd.to_datetime(cohort_cfg.time_windows.data_end)
        except AttributeError:
            logger.warning(
                "No data_end found in cohort_cfg.time_windows. Defaulting to today."
            )
            return pd.to_datetime("today")

    @staticmethod
    def _save_artifacts(artifacts: Artifacts, out_dir: str):
        """Saves all processed data and artifacts to disk."""
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Saving artifacts to {out_dir}")
        save_vocabulary(artifacts.vocabulary, out_dir)
        artifacts.data.save(out_dir)
        artifacts.exposures.to_csv(join(out_dir, EXPOSURES_FILE), index=False)
        artifacts.index_dates.to_csv(join(out_dir, INDEX_DATES_FILE), index=False)
        artifacts.follow_ups.to_csv(join(out_dir, FOLLOW_UPS_FILE), index=False)
        artifacts.binary_outcomes.to_csv(join(out_dir, BINARY_OUTCOMES_FILE))
        artifacts.binary_exposure.to_csv(join(out_dir, BINARY_EXPOSURE_FILE))
