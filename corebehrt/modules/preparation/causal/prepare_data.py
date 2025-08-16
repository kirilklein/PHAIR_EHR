import logging
import os
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


from corebehrt.constants.causal.data import EXPOSURE
from corebehrt.constants.causal.paths import (
    BINARY_EXPOSURE_FILE,
    BINARY_OUTCOMES_FILE,
    EXPOSURES_FILE,
    INDEX_DATE_MATCHING_FILE,
)
from corebehrt.constants.data import ABSPOS_COL, DEATH_CODE, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import FOLLOW_UPS_FILE, INDEX_DATES_FILE
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.causal.convert import (
    abspos_to_binary_outcome,
    dataframe_to_causal_patient_list,
)
from corebehrt.functional.preparation.causal.extract import extract_death
from corebehrt.functional.preparation.causal.follow_up import (
    get_combined_follow_ups,
    prepare_follow_ups_simple,
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
from corebehrt.functional.visualize.follow_ups import (
    plot_follow_up_distribution,
    plot_followups_timeline,
)
from corebehrt.functional.visualize.outcomes import (
    plot_outcome_distribution,
    plot_filtering_stats,
)
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.preparation.causal.config import ExposureConfig, OutcomeConfig
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import Config

logger = logging.getLogger(__name__)


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

    DEATH_OUTCOME_KEYWORDS = ["dod", "death", "all_cause_death"]

    def __init__(self, cfg: Config, cohort_cfg: Config, logger: logging.Logger):
        self.ds_preparer = DatasetPreparer(cfg)
        self.exposure_cfg = ExposureConfig(**cfg.exposure)
        self.outcome_cfg = OutcomeConfig(**cfg.outcome)
        self.paths_cfg = cfg.paths
        self.data_cfg = cfg.data
        self.end_date = self.get_end_date(cohort_cfg)
        self.vocabulary = self.ds_preparer.vocab
        self.min_instances_per_class = self.data_cfg.get("min_instances_per_class", 10)
        self.logger = logger

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
        self.logger.info("Loading and filtering initial data")
        pids = self.ds_preparer.load_cohort(self.paths_cfg)
        data = self.load_shards_into_patient_data(pids, mode)
        pids = data.get_pids()  # Use PIDs actually present in the data

        exposures, index_date_matching, index_dates = self.load_cohort_data(
            self.paths_cfg.cohort
        )
        self.logger.info("Loading outcomes")
        outcomes = self._load_outcomes()

        exposures, index_dates, outcomes = self._filter_dataframes_by_pids(
            pids, exposures, index_dates, outcomes
        )
        if ABSPOS_COL not in index_dates.columns:
            index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])
        # 2. Censor, truncate, and normalize sequences
        censor_dates = self._censor_and_truncate_sequences(data, index_dates)

        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        self.logger.info(
            f"Max segment length: {max(max(p.segments, default=0) for p in data.patients)}"
        )
        # 3. Compute labels and outcomes
        deaths = self._extract_deaths(data)
        self.logger.info("Computing binary labels")

        # Compute exposure targets from censored patient data instead of original exposure data
        # This ensures consistency between exposure targets and available concepts
        binary_exposure = self._compute_exposure_from_censored_data(data)

        # Compute outcome targets normally (outcomes are not typically present in concept data)
        binary_outcomes, follow_ups, filtering_stats = self._compute_outcome_labels(
            outcomes, index_dates, index_date_matching, deaths, exposures
        )

        # 4. Assign labels to patient data
        self._assign_labels(data, binary_exposure, binary_outcomes)

        # 5. Save all generated artifacts
        self.logger.info("Saving artifacts")
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
        if follow_ups is not None:
            fig_dir = join(self.paths_cfg.prepared_data, "figures")
            plot_follow_up_distribution(follow_ups, binary_exposure, fig_dir)
            plot_outcome_distribution(binary_outcomes, fig_dir)
            plot_filtering_stats(filtering_stats, fig_dir)
            plot_followups_timeline(
                exposures=exposures,
                outcomes=outcomes,
                follow_ups=follow_ups,
                index_date_matching=index_date_matching,
                censor_dates=censor_dates,
                save_dir=fig_dir,
                n_random_subjects=30,
            )
        return data

    def _load_outcomes(self) -> Dict[str, pd.DataFrame]:
        """Loads single or multiple outcome files into a dictionary."""
        outcomes = {}
        for name, outcome_file in self.paths_cfg.outcome_files.items():
            self.logger.info(f"Loading outcome file: {outcome_file}")
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
        self.logger.info(f"Filtering dataframes for {len(pids)} patients.")
        index_dates = filter_df_by_pids(index_dates, pids)
        exposures = filter_df_by_pids(exposures, pids)
        filtered_outcomes = {
            name: filter_df_by_pids(df, pids) for name, df in outcomes.items()
        }
        return exposures, index_dates, filtered_outcomes

    def _extract_deaths(self, data: CausalPatientDataset) -> pd.Series:
        """Extracts death information for each patient."""
        deaths_list = data.process_in_parallel(
            extract_death, death_token=self.vocabulary[DEATH_CODE]
        )
        return pd.Series(
            {patient.pid: death for patient, death in zip(data.patients, deaths_list)}
        )

    def _compute_binary_labels(
        self,
        exposures: pd.DataFrame,
        outcomes: Dict[str, pd.DataFrame],
        index_dates: pd.DataFrame,
        index_date_matching: pd.DataFrame,
        deaths: pd.Series,
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
        """
        Computes binary exposure and outcome labels.

        This refactored version pre-computes two sets of follow-up periods:
        1.  A standard follow-up, censored by death.
        2.  A special follow-up for death outcomes, which is not censored by death.
        Inside the loop, it selects the appropriate pre-computed follow-up,
        improving efficiency and clarity.
        """
        self.logger.info("Handling exposures and outcomes")
        filtering_stats = {}

        filtering_stats[EXPOSURE] = {"before": exposures[PID_COL].nunique()}
        exposure_follow_ups = prepare_follow_ups_simple(
            index_dates,
            self.exposure_cfg.n_hours_start_follow_up,
            self.exposure_cfg.n_hours_end_follow_up,
            self.end_date,
        )
        binary_exposure = abspos_to_binary_outcome(exposure_follow_ups, exposures)
        filtering_stats[EXPOSURE]["after"] = binary_exposure.value_counts().to_dict()
        # --- Refactored Outcome Handling ---
        self.logger.info("Pre-calculating follow-up periods for outcomes.")

        # 1. Calculate the standard follow-up period (censored by death)
        standard_follow_ups = get_combined_follow_ups(
            index_dates=index_dates,
            index_date_matching=index_date_matching,
            deaths=deaths,
            exposures=exposures,
            data_end=self.end_date,
            cfg=self.outcome_cfg,
            censor_by_death=True,
        )

        # 2. Calculate the special follow-up period for death outcomes (NOT censored by death)
        if any(self._is_death_outcome(name) for name in outcomes.keys()):
            death_follow_ups = get_combined_follow_ups(
                index_dates=index_dates,
                index_date_matching=index_date_matching,
                deaths=deaths,
                exposures=exposures,
                data_end=self.end_date,
                cfg=self.outcome_cfg,
                censor_by_death=False,
            )

        binary_outcomes = {}
        min_instances_per_class = self.outcome_cfg.min_instances_per_class

        for outcome_name, outcome_df in outcomes.items():
            filtering_stats[outcome_name] = {"before": outcome_df[PID_COL].nunique()}

            if outcome_df.empty:
                self.logger.warning(f"Outcome {outcome_name} has no data. Skipping.")
                filtering_stats[outcome_name]["after"] = {
                    "skipped": True,
                    "reason": "Empty dataframe",
                }
                continue

            if self._is_death_outcome(outcome_name):
                active_follow_ups = death_follow_ups
                self.logger.info(
                    f"Using non-censored follow-up for death outcome: {outcome_name}"
                )
            else:
                active_follow_ups = standard_follow_ups

            # Generate binary label using the selected follow-up
            binary_outcome = abspos_to_binary_outcome(active_follow_ups, outcome_df)

            # Validate class balance
            counts = binary_outcome.value_counts()
            if len(counts) < 2 or counts.min() < min_instances_per_class:
                self.logger.warning(
                    f"Outcome {outcome_name} has a class with fewer than "
                    f"{min_instances_per_class} instances. Value counts: {counts.to_dict()}. Skipping."
                )
                filtering_stats[outcome_name]["after"] = {
                    "skipped": True,
                    "reason": "Low class instances",
                }
                continue

            filtering_stats[outcome_name]["after"] = counts.to_dict()
            binary_outcomes[outcome_name] = binary_outcome
        binary_outcomes = pd.DataFrame(binary_outcomes)
        self.logger.info(
            f"Binary exposure distribution\n{binary_exposure.value_counts()}"
        )
        self.logger.info(
            f"Binary outcomes distribution\n{binary_outcomes.apply(pd.Series.value_counts)}"
        )
        # Return the standard follow-ups as the representative DataFrame
        return (
            binary_exposure,
            binary_outcomes,
            standard_follow_ups,
            filtering_stats,
        )

    def _compute_outcome_labels(
        self,
        outcomes: Dict[str, pd.DataFrame],
        index_dates: pd.DataFrame,
        index_date_matching: pd.DataFrame,
        deaths: pd.Series,
        exposures: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Computes binary outcome labels only (not exposure labels).

        This is a modified version of _compute_binary_labels that only handles outcomes.
        """
        self.logger.info("Handling outcomes")
        filtering_stats = {}

        # 1. Calculate the standard follow-up period (censored by death)
        standard_follow_ups = get_combined_follow_ups(
            index_dates=index_dates,
            index_date_matching=index_date_matching,
            deaths=deaths,
            exposures=exposures,  # Empty since we're not using it for outcomes
            data_end=self.end_date,
            cfg=self.outcome_cfg,
            censor_by_death=True,
        )

        # 2. Calculate the special follow-up period for death outcomes (NOT censored by death)
        death_follow_ups = None
        if any(self._is_death_outcome(name) for name in outcomes.keys()):
            death_follow_ups = get_combined_follow_ups(
                index_dates=index_dates,
                index_date_matching=index_date_matching,
                deaths=deaths,
                exposures=exposures,  # Empty since we're not using it for outcomes
                data_end=self.end_date,
                cfg=self.outcome_cfg,
                censor_by_death=False,
            )

        # 3. Process each outcome
        binary_outcomes = {}
        min_instances_per_class = self.min_instances_per_class

        for outcome_name, outcome_df in outcomes.items():
            self.logger.info(f"Processing outcome: {outcome_name}")
            filtering_stats[outcome_name] = {"before": outcome_df[PID_COL].nunique()}

            if outcome_df.empty:
                self.logger.warning(f"Outcome {outcome_name} has no data. Skipping.")
                filtering_stats[outcome_name]["after"] = {
                    "skipped": True,
                    "reason": "Empty dataframe",
                }
                continue

            if self._is_death_outcome(outcome_name):
                active_follow_ups = death_follow_ups
                self.logger.info(
                    f"Using non-censored follow-up for death outcome: {outcome_name}"
                )
            else:
                active_follow_ups = standard_follow_ups

            # Generate binary label using the selected follow-up
            binary_outcome = abspos_to_binary_outcome(active_follow_ups, outcome_df)

            # Validate class balance
            counts = binary_outcome.value_counts()
            if len(counts) < 2 or counts.min() < min_instances_per_class:
                self.logger.warning(
                    f"Outcome {outcome_name} has a class with fewer than "
                    f"{min_instances_per_class} instances. Value counts: {counts.to_dict()}. Skipping."
                )
                filtering_stats[outcome_name]["after"] = {
                    "skipped": True,
                    "reason": "Low class instances",
                }
                continue

            filtering_stats[outcome_name]["after"] = counts.to_dict()
            binary_outcomes[outcome_name] = binary_outcome

        binary_outcomes = pd.DataFrame(binary_outcomes)
        self.logger.info(
            f"Binary outcomes distribution\n{binary_outcomes.apply(pd.Series.value_counts)}"
        )
        return binary_outcomes, standard_follow_ups, filtering_stats

    def _is_death_outcome(self, outcome_name: str) -> bool:
        """Determines if the outcome is death-related."""
        return (
            outcome_name.lower() in self.DEATH_OUTCOME_KEYWORDS
            or "death" in outcome_name.lower()
        )

    def _compute_exposure_from_censored_data(
        self, data: CausalPatientDataset
    ) -> pd.Series:
        """
        Computes exposure targets based on the presence of exposure concepts
        in the censored patient concept data.

        This ensures consistency between the exposure targets and the actual
        concepts available in the censored sequences.
        """
        exposure_code = self.vocabulary.get("EXPOSURE")
        if exposure_code is None:
            raise ValueError("EXPOSURE concept not found in vocabulary")

        exposure_targets = {}
        for patient in data.patients:
            # Check if exposure concept is present in the censored concept data
            has_exposure = int(exposure_code in set(patient.concepts))
            exposure_targets[patient.pid] = has_exposure

        return pd.Series(exposure_targets, name="exposure")

    def _assign_labels(
        self,
        data: CausalPatientDataset,
        binary_exposure: pd.Series,
        binary_outcomes: pd.DataFrame,
    ):
        """Assigns computed labels to the CausalPatientDataset."""
        self.logger.info("Assigning exposures and outcomes")
        data.assign_attributes(EXPOSURE, binary_exposure)
        data.assign_outcomes(binary_outcomes)

    def _censor_and_truncate_sequences(
        self, data: CausalPatientDataset, index_dates: pd.DataFrame
    ) -> pd.Series:
        """Applies censoring, filters short sequences, and truncates."""
        # Censor sequences based on index dates
        index_dates = index_dates.drop_duplicates(subset=PID_COL, keep="last")
        censor_dates = (
            index_dates.set_index(PID_COL)[ABSPOS_COL]
            + self.exposure_cfg.n_hours_censoring
        )
        self.ds_preparer._validate_censoring(data.patients, censor_dates, self.logger)

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
        return censor_dates

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
        for df, _ in tqdm(loader(), desc=desc, file=TqdmToLogger(self.logger)):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            if self.data_cfg.get("cutoff_date"):
                df = self.ds_preparer._cutoff_data(df, self.data_cfg.cutoff_date)

            self.ds_preparer._check_sorted(df)
            patient_list.extend(dataframe_to_causal_patient_list(df))

        self.logger.info(f"Loaded {len(patient_list)} patients.")
        return CausalPatientDataset(patients=patient_list)

    @staticmethod
    def load_cohort_data(
        cohort_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
        """Loads exposure and index date files."""
        exposures = pd.read_csv(join(cohort_path, EXPOSURES_FILE))
        index_date_matching = None
        if os.path.exists(join(cohort_path, INDEX_DATE_MATCHING_FILE)):
            index_date_matching = pd.read_csv(
                join(cohort_path, INDEX_DATE_MATCHING_FILE)
            )
        index_dates = pd.read_csv(
            join(cohort_path, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        return exposures, index_date_matching, index_dates

    @staticmethod
    def get_end_date(cohort_cfg):
        time_windows = cohort_cfg.time_windows
        if data_end := time_windows.get("data_end"):
            return pd.Timestamp(**data_end)
        else:
            return pd.Timestamp.now()

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
