import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm import tqdm
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.constants.data import ABSPOS_COL, CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.constants.causal.data import (
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_OUTCOME_CONTROL,
    OUTCOME_COL,
)

logger = logging.getLogger("simulate")


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    data: str
    splits: List[str]
    outcomes: str


@dataclass
class ExposureConfig:
    """Configuration for exposure simulation."""

    code: str
    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]
    run_in_days: int = 365
    compliance_interval_days: int = 30
    min_compliance_days: int = 365


@dataclass
class OutcomeConfig:
    """Configuration for a single outcome simulation."""

    code: str
    run_in_days: int
    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]
    exposure_effect: float


@dataclass
class SimulationConfig:
    """Top-level configuration for the entire causal simulation."""

    paths: PathsConfig
    exposure: ExposureConfig
    outcomes: Dict[str, OutcomeConfig]

    def __init__(self, config: dict):
        # Parse paths
        self.paths = PathsConfig(**config["paths"])

        # Parse exposure
        self.exposure = ExposureConfig(**config["exposure"])

        # Parse outcomes
        self.outcomes = {}
        for outcome_key, outcome_data in config.get("outcomes", {}).items():
            self.outcomes[outcome_key] = OutcomeConfig(**outcome_data)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate that the configuration is consistent."""
        # Validate exposure section
        if len(self.exposure.trigger_codes) != len(self.exposure.trigger_weights):
            raise ValueError(
                f"Exposure section: trigger_codes length ({len(self.exposure.trigger_codes)}) "
                f"must match trigger_weights length ({len(self.exposure.trigger_weights)})"
            )

        # Validate each outcome section
        for outcome_key, outcome in self.outcomes.items():
            if len(outcome.trigger_codes) != len(outcome.trigger_weights):
                raise ValueError(
                    f"Outcome '{outcome_key}': trigger_codes length ({len(outcome.trigger_codes)}) "
                    f"must match trigger_weights length ({len(outcome.trigger_weights)})"
                )

        # Validate required fields are present and reasonable
        if self.exposure.p_base < 0 or self.exposure.p_base > 1:
            raise ValueError(
                f"Exposure p_base must be between 0 and 1, got {self.exposure.p_base}"
            )

        for outcome_key, outcome in self.outcomes.items():
            if outcome.p_base < 0 or outcome.p_base > 1:
                raise ValueError(
                    f"Outcome '{outcome_key}' p_base must be between 0 and 1, got {outcome.p_base}"
                )

    def get_all_trigger_codes(self) -> List[str]:
        """Get all trigger codes from exposure and outcomes."""
        all_codes = set(self.exposure.trigger_codes)
        for outcome in self.outcomes.values():
            all_codes.update(outcome.trigger_codes)
        return list(all_codes)

    def get_confounder_codes(self) -> Dict[str, List[str]]:
        """Get confounder codes for each outcome (codes that appear in both exposure and that outcome)."""
        confounders_by_outcome = {}
        exposure_codes = set(self.exposure.trigger_codes)

        for outcome_key, outcome_cfg in self.outcomes.items():
            outcome_codes = set(outcome_cfg.trigger_codes)
            confounders = exposure_codes.intersection(outcome_codes)
            confounders_by_outcome[outcome_key] = list(confounders)

        return confounders_by_outcome

    def get_all_confounder_codes(self) -> List[str]:
        """Get all unique confounder codes across all outcomes."""
        all_confounders = set()
        for confounders in self.get_confounder_codes().values():
            all_confounders.update(confounders)
        return list(all_confounders)


class CausalSimulator:
    """
    Simulates exposure and outcome events based on a patient's history.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def simulate_dataset(
        self, df: pd.DataFrame, seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Simulates exposures and outcomes for a cohort, including counterfactuals.

        Args:
            df: DataFrame with patient data (subject_id, time, code).
            seed: Random seed for reproducibility.

        Returns:
            A dictionary of DataFrames, including factual events, ITEs,
            and counterfactual outcomes.
        """
        np.random.seed(seed)

        factual_events, ite_records, cf_records = [], [], []
        subjects_as_dfs = [group for _, group in df.groupby(PID_COL)]

        for subj_df in tqdm(subjects_as_dfs, desc="Simulating exposures and outcomes"):
            subject_results = self._simulate_for_subject(subj_df)
            if not subject_results:
                continue

            factual_events.extend(subject_results["factual_events"])
            ite_records.append(subject_results["ite_record"])
            cf_records.append(subject_results["cf_record"])

        if not factual_events:
            return {}

        output_dfs = {}

        events_df = pd.DataFrame(factual_events)
        events_df[ABSPOS_COL] = get_hours_since_epoch(events_df[TIMESTAMP_COL])
        for code, group in events_df.groupby(CONCEPT_COL):
            output_dfs[str(code)] = group[[PID_COL, TIMESTAMP_COL, ABSPOS_COL]].copy()

        output_dfs["ite"] = pd.DataFrame(ite_records)
        output_dfs["counterfactuals"] = pd.DataFrame(cf_records)

        return output_dfs

    def _simulate_for_subject(self, subj_df: pd.DataFrame) -> dict:
        """Simulates all events and counterfactuals for a single subject."""
        if subj_df.empty:
            return {}

        subject_id = subj_df[PID_COL].iloc[0]
        start_date = subj_df[TIMESTAMP_COL].min()
        end_date = subj_df[TIMESTAMP_COL].max()

        index_date = start_date + pd.Timedelta(days=self.config.exposure.run_in_days)

        if index_date >= end_date:
            return {}

        history_at_index = self._get_history_codes(subj_df, index_date)
        exposure_cfg = self.config.exposure
        p_exposure = self._calculate_probability(exposure_cfg, history_at_index)
        is_exposed = np.random.binomial(1, p_exposure) == 1

        factual_events = []
        if is_exposed:
            factual_events.append(
                {
                    PID_COL: subject_id,
                    TIMESTAMP_COL: index_date,
                    CONCEPT_COL: exposure_cfg.code,
                }
            )

        history_for_outcomes = history_at_index.copy()
        if is_exposed:
            history_for_outcomes.add(exposure_cfg.code)

        ite_record = {PID_COL: subject_id}
        cf_record = {PID_COL: subject_id, "exposure": int(is_exposed)}

        for outcome_name, outcome_cfg in self.config.outcomes.items():
            assessment_time = index_date + pd.Timedelta(days=outcome_cfg.run_in_days)
            if assessment_time >= end_date:
                continue

            p_if_treated = self._calculate_probability(
                outcome_cfg, history_for_outcomes, is_exposed=True
            )
            p_if_control = self._calculate_probability(
                outcome_cfg, history_for_outcomes, is_exposed=False
            )

            ite_record[f"ite_{outcome_cfg.code}"] = p_if_treated - p_if_control

            outcome_exposed = np.random.binomial(1, p_if_treated)
            outcome_control = np.random.binomial(1, p_if_control)
            factual_outcome = outcome_exposed if is_exposed else outcome_control

            if factual_outcome:
                factual_events.append(
                    {
                        PID_COL: subject_id,
                        TIMESTAMP_COL: assessment_time,
                        CONCEPT_COL: outcome_cfg.code,
                    }
                )

            cf_record[f"{OUTCOME_COL}_{outcome_cfg.code}"] = factual_outcome
            cf_record[f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_cfg.code}"] = (
                outcome_exposed
            )
            cf_record[f"{SIMULATED_OUTCOME_CONTROL}_{outcome_cfg.code}"] = (
                outcome_control
            )

        return {
            "factual_events": factual_events,
            "ite_record": ite_record,
            "cf_record": cf_record,
        }

    def _get_history_codes(
        self, subj_df: pd.DataFrame, assessment_time: pd.Timestamp
    ) -> set:
        """Extracts codes from subject history up to assessment time."""
        history_mask = subj_df[TIMESTAMP_COL] <= assessment_time
        return set(subj_df.loc[history_mask, CONCEPT_COL])

    def _calculate_probability(
        self, event_cfg, history_codes: set, is_exposed: bool = False
    ) -> float:
        """
        Calculates event probability based on history and, for outcomes, exposure status.
        """
        trigger_codes_array = np.array(list(event_cfg.trigger_codes))
        trigger_weights_array = np.array(list(event_cfg.trigger_weights))

        codes_present_mask = np.isin(trigger_codes_array, list(history_codes))
        trigger_effect_sum = np.sum(trigger_weights_array[codes_present_mask])

        logit_p = logit(event_cfg.p_base) + trigger_effect_sum

        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p += event_cfg.exposure_effect

        return expit(logit_p)
