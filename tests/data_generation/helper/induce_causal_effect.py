"""
Causal Effect Simulation for Electronic Health Records (EHR) Data

This module provides a sophisticated framework for simulating causal effects in longitudinal
EHR data, enabling the generation of synthetic datasets with known causal relationships
between exposures and outcomes.

Key Features:
- Two-pass simulation approach ensuring proper temporal dependencies
- Time-to-event modeling for exposure initiation
- Individual Treatment Effect (ITE) calculation for each subject
- Compliance modeling with random treatment discontinuation
- Trigger-based probability modification using logistic regression
- Vectorized computations for computational efficiency

Simulation Process:
1. **Exposure Simulation (Pass 1)**: For each subject, simulates the first exposure event
   based on baseline probability and trigger codes, then generates subsequent exposures
   at regular intervals until a randomly determined compliance end date.

2. **Outcome Simulation (Pass 2)**: Uses the complete distribution of exposure dates
   to ensure fair comparison between exposed and unexposed subjects. Calculates both
   Individual Treatment Effects (ITEs) and simulates factual outcomes.

Mathematical Foundation:
- Probability calculations use logistic regression on the logit scale
- Daily probabilities derived from total probabilities over time periods
- Cumulative trigger effects maintain state across time
- Exposure effects modify outcome probabilities additively on logit scale

Usage:
    config = SimulationConfig(...)
    simulator = CausalSimulator(config)
    simulated_data, ite_data = simulator.simulate_dataset(original_data, seed=42)

The resulting datasets contain:
- Simulated EHR data with exposure and outcome events
- ITE data with counterfactual effect estimates for each subject

This framework is particularly useful for:
- Evaluating causal inference methods
- Generating training data for machine learning models
- Conducting simulation studies in pharmacoepidemiology
- Testing the performance of treatment effect estimation algorithms
"""

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm import tqdm
from tests.data_generation.helper.config import SimulationConfig


class CausalSimulator:
    """
    Handles a multi-phase, time-to-event causal simulation for EHR data,
    driven by a hierarchical configuration object.
    """

    # Constants
    MAX_RETRY_ATTEMPTS = 100

    def __init__(self, config: SimulationConfig):
        """
        Initializes the simulator with a single configuration object.

        Args:
            config: A SimulationConfig object containing all parameters.
        """
        self.config = config
        self.first_exposure_dates = []

    def simulate_dataset(
        self, df: pd.DataFrame, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the full causal simulation to the entire dataset using a two-pass approach."""
        np.random.seed(seed)

        # Pass 1: Simulate exposures and collect exposure dates
        subjects_with_exposure = self._simulate_all_exposures(df)

        # Pass 2: Simulate outcomes using complete exposure distribution
        return self._simulate_all_outcomes_dataset(subjects_with_exposure)

    def _simulate_all_exposures(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Simulates exposure processes for all subjects."""
        subjects_with_exposure = []
        self.first_exposure_dates = []
        for _, subj_df in tqdm(df.groupby("subject_id"), desc="Simulating exposures"):
            subj_df_sorted = subj_df.sort_values(
                "time", na_position="first"
            ).reset_index(drop=True)
            df_with_exposure = self._simulate_exposure_process(subj_df_sorted)
            subjects_with_exposure.append(df_with_exposure)

        return subjects_with_exposure

    def _simulate_all_outcomes_dataset(
        self, subjects_with_exposure: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulates outcomes for all subjects and combines results."""
        all_subject_dfs = []
        all_ite_records = []

        for df_with_exposure in tqdm(
            subjects_with_exposure, desc="Simulating outcomes"
        ):
            final_df, ite_record = self._simulate_subject_outcomes(df_with_exposure)
            all_subject_dfs.append(final_df)
            if ite_record:  # Only add if not empty
                all_ite_records.append(ite_record)

        # Combine all subject data
        simulated_df = pd.concat(all_subject_dfs, ignore_index=True)

        # Combine all ITE records
        ite_df = pd.DataFrame(all_ite_records) if all_ite_records else pd.DataFrame()

        return simulated_df, ite_df

    def _simulate_exposure_process(self, subj_df: pd.DataFrame) -> pd.DataFrame:
        """Simulates the complete exposure process for a single subject."""
        cfg = self.config.exposure

        # Find first exposure event
        df_with_temp_exposure = self._simulate_time_to_first_event(
            subj_df,
            cfg.p_base,
            cfg.trigger_codes,
            cfg.trigger_weights,
            f"TEMP_{cfg.code}",
            cfg.run_in_days,
        )

        # Check if exposure occurred
        temp_exposure_events = df_with_temp_exposure[
            df_with_temp_exposure["code"] == f"TEMP_{cfg.code}"
        ]
        if temp_exposure_events.empty:
            return subj_df

        # Record first exposure date and generate compliant exposures
        first_exposure_date = temp_exposure_events["time"].min()
        self.first_exposure_dates.append(first_exposure_date)

        compliance_end_date = self._get_random_compliance_end_date(
            first_exposure_date, subj_df["time"].max()
        )
        exposure_dates = self._generate_regular_exposures(
            first_exposure_date, compliance_end_date, cfg.compliance_interval_days
        )

        return self._replace_temp_with_final_exposures(
            df_with_temp_exposure, exposure_dates, cfg.code
        )

    def _get_random_compliance_end_date(
        self, first_exposure_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Selects a compliance end date with an evenly spaced probability
        """
        earliest_end = first_exposure_date + pd.Timedelta(
            days=self.config.exposure.min_compliance_days
        )

        if earliest_end >= end_date:
            return end_date

        total_days = (end_date - earliest_end).days

        # Handle edge case where there are very few days available
        if total_days < 1:
            return end_date

        # Work in days for better precision
        days = np.arange(total_days)

        # Calculate decay rate (lambda = 1 / (decay_years * 365.25))
        weights = np.arange(total_days, dtype="float64") ** 2
        weights /= weights.sum()

        # Choose a day offset according to exponential distribution
        chosen_offset = np.random.choice(days, p=weights)
        compliance_end = earliest_end + pd.Timedelta(days=int(chosen_offset))
        if compliance_end > end_date:
            return end_date

        return compliance_end

    def _generate_regular_exposures(
        self,
        first_exposure_date: pd.Timestamp,
        compliance_end_date: pd.Timestamp,
        interval_days: int,
    ) -> pd.DatetimeIndex:
        """
        Generates exposure dates at regular intervals between first exposure and compliance end.
        """
        return pd.date_range(
            start=first_exposure_date,
            end=compliance_end_date,
            freq=f"{interval_days}D",
        )

    def _replace_temp_with_final_exposures(
        self,
        df_with_temp: pd.DataFrame,
        exposure_dates: pd.DatetimeIndex,
        exposure_code: str,
    ) -> pd.DataFrame:
        """Replaces temporary exposure events with final ones."""
        # Remove temporary events
        df_clean = df_with_temp[df_with_temp["code"] != f"TEMP_{exposure_code}"]

        if exposure_dates.empty:
            return df_clean

        # Add final exposure events
        return self._add_events_to_dataframe(df_clean, exposure_dates, exposure_code)

    def _simulate_subject_outcomes(
        self, subj_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """Simulates all outcomes for a single subject."""
        subject_info = self._extract_subject_info(subj_df)
        if not subject_info:
            return subj_df, {}

        start_date, end_date, has_exposure, first_exposure_date, subject_id = (
            subject_info
        )

        df_with_outcomes = subj_df.copy()
        ite_record = {"subject_id": subject_id, "has_exposure": int(has_exposure)}

        # Simulate each outcome
        for outcome_cfg in self.config.outcomes.values():
            assessment_time = self._determine_assessment_time(
                has_exposure,
                first_exposure_date,
                start_date,
                end_date,
                outcome_cfg.run_in_days,
            )

            df_with_outcomes, ite_value = self._simulate_single_outcome_complete(
                df_with_outcomes, outcome_cfg, assessment_time
            )
            ite_record[f"ite_{outcome_cfg.code}"] = ite_value

        return df_with_outcomes, ite_record

    def _extract_subject_info(self, subj_df: pd.DataFrame) -> Optional[Tuple]:
        """Extracts key information about a subject."""
        if (
            subj_df.empty
            or "time" not in subj_df.columns
            or subj_df["time"].dropna().empty
        ):
            return None

        start_date = subj_df["time"].dropna().min().normalize()
        end_date = subj_df["time"].dropna().max().normalize()

        exposure_events = subj_df[subj_df["code"] == self.config.exposure.code]
        has_exposure = not exposure_events.empty
        first_exposure_date = exposure_events["time"].min() if has_exposure else None
        subject_id = subj_df.iloc[0]["subject_id"]

        return start_date, end_date, has_exposure, first_exposure_date, subject_id

    def _determine_assessment_time(
        self,
        has_exposure: bool,
        first_exposure_date: Optional[pd.Timestamp],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        run_in_days: int,
    ) -> pd.Timestamp:
        """Determines when to assess outcomes, ensuring valid timeline."""
        fallback_date = end_date - pd.Timedelta(days=1)

        if has_exposure:
            return self._get_exposed_assessment_time(
                first_exposure_date, end_date, run_in_days, fallback_date
            )
        else:
            return self._get_unexposed_assessment_time(
                start_date, end_date, run_in_days, fallback_date
            )

    def _get_exposed_assessment_time(
        self,
        first_exposure_date: pd.Timestamp,
        end_date: pd.Timestamp,
        run_in_days: int,
        fallback_date: pd.Timestamp,
    ) -> pd.Timestamp:
        """Gets assessment time for exposed subjects."""
        assessment_start = first_exposure_date.normalize()
        assessment_time = assessment_start + pd.Timedelta(days=run_in_days)

        return assessment_time if assessment_time <= end_date else fallback_date

    def _get_unexposed_assessment_time(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        run_in_days: int,
        fallback_date: pd.Timestamp,
    ) -> pd.Timestamp:
        """Gets assessment time for unexposed subjects using exposure distribution."""
        if not self.first_exposure_dates:
            candidate_time = start_date + pd.Timedelta(days=run_in_days)
            return candidate_time if candidate_time < end_date else fallback_date

        # Try to find suitable pseudo-exposure date
        for _ in range(self.MAX_RETRY_ATTEMPTS):
            pseudo_exposure_date = np.random.choice(
                self.first_exposure_dates
            ).normalize()
            assessment_time = pseudo_exposure_date + pd.Timedelta(days=run_in_days)

            if start_date <= assessment_time <= end_date:
                return assessment_time

        return fallback_date

    def _simulate_single_outcome_complete(
        self, subj_df: pd.DataFrame, outcome_cfg, assessment_time: pd.Timestamp
    ) -> Tuple[pd.DataFrame, float]:
        """
        Simulates a single outcome: calculates ITE and simulates factual outcome.
        """
        # Get historical information
        history_codes = self._get_history_codes(subj_df, assessment_time)
        is_exposed = self.config.exposure.code in history_codes

        # Calculate ITE and factual probability
        ite = self._calculate_ite(outcome_cfg, history_codes)
        p_factual = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed
        )

        # Simulate factual outcome
        if np.random.binomial(1, p_factual):
            subj_df = self._add_outcome_event(
                subj_df, outcome_cfg.code, assessment_time
            )

        return subj_df, ite

    def _get_history_codes(
        self, subj_df: pd.DataFrame, assessment_time: pd.Timestamp
    ) -> set:
        """Extracts codes from subject history up to assessment time."""
        history_mask = subj_df["time"] <= assessment_time
        return set(subj_df.loc[history_mask, "code"])

    def _calculate_ite(self, outcome_cfg, history_codes: set) -> float:
        """Calculates Individual Treatment Effect."""
        p_if_treated = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=True
        )
        p_if_control = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=False
        )
        return p_if_treated - p_if_control

    def _calculate_outcome_probability(
        self, outcome_cfg, history_codes: set, is_exposed: bool
    ) -> float:
        """Calculates outcome probability given history and exposure status."""
        # Vectorized computation of trigger effects
        trigger_codes_array = np.array(outcome_cfg.trigger_codes)
        trigger_weights_array = np.array(outcome_cfg.trigger_weights)
        codes_present_mask = np.array(
            [code in history_codes for code in trigger_codes_array]
        )

        # Calculate logit probability
        trigger_effect_sum = np.sum(trigger_weights_array[codes_present_mask])
        logit_p = logit(outcome_cfg.p_base) + trigger_effect_sum

        if is_exposed:
            logit_p += outcome_cfg.exposure_effect

        return expit(logit_p)

    def _add_outcome_event(
        self, subj_df: pd.DataFrame, outcome_code: str, assessment_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Adds an outcome event to the subject dataframe."""
        new_event = pd.DataFrame(
            {
                "subject_id": [subj_df.iloc[0]["subject_id"]],
                "time": [assessment_time],
                "code": [outcome_code],
            }
        )
        return (
            pd.concat([subj_df, new_event], ignore_index=True)
            .sort_values("time")
            .reset_index(drop=True)
        )

    def _add_events_to_dataframe(
        self, df: pd.DataFrame, event_times: pd.DatetimeIndex, event_code: str
    ) -> pd.DataFrame:
        """Generic method to add multiple events to a dataframe."""
        if event_times.empty:
            return df

        new_events = pd.DataFrame(
            {
                "subject_id": df.iloc[0]["subject_id"],
                "time": event_times,
                "code": event_code,
            }
        )

        return (
            pd.concat([df, new_events], ignore_index=True)
            .sort_values("time", na_position="first")
            .reset_index(drop=True)
        )

    # Time-to-event simulation methods (simplified)
    def _simulate_time_to_first_event(
        self,
        subj_df: pd.DataFrame,
        p_total_base: float,
        trigger_codes: List[str],
        trigger_weights: List[float],
        event_name: str,
        run_in_days: int,
    ) -> pd.DataFrame:
        """Simulates time to first event occurrence."""
        timeline_info = self._setup_simulation_timeline(subj_df, run_in_days)
        if timeline_info is None:
            return subj_df

        daily_timeline, total_days = timeline_info
        p_daily_base = self._compute_daily_prob(p_total_base, total_days - run_in_days)

        feature_matrix = self._build_feature_matrix(
            subj_df, daily_timeline, trigger_codes
        )
        event_probabilities = self._compute_event_probabilities(
            feature_matrix, trigger_codes, trigger_weights, p_daily_base
        )

        return self._simulate_and_add_event(
            subj_df, daily_timeline, event_probabilities, event_name
        )

    def _setup_simulation_timeline(
        self, subj_df: pd.DataFrame, run_in_days: int
    ) -> Optional[Tuple[pd.DatetimeIndex, int]]:
        """Sets up simulation timeline and validates window."""
        start_date = subj_df["time"].min().normalize()
        end_date = subj_df["time"].max().normalize()
        sim_window_start = start_date + pd.Timedelta(days=run_in_days)

        if sim_window_start >= end_date:
            return None

        total_days = (end_date - sim_window_start).days
        daily_timeline = pd.date_range(start=sim_window_start, end=end_date, freq="D")

        return daily_timeline, total_days

    def _build_feature_matrix(
        self,
        subj_df: pd.DataFrame,
        daily_timeline: pd.DatetimeIndex,
        trigger_codes: List[str],
    ) -> pd.DataFrame:
        """Creates feature matrix with cumulative triggers."""
        events_pivot = subj_df.pivot_table(
            index="time", columns="code", aggfunc="size", fill_value=0
        ).astype(bool)

        # Handle missing trigger codes
        all_codes = set(trigger_codes) | set(events_pivot.columns)
        feature_matrix = events_pivot.reindex(
            index=daily_timeline, columns=list(all_codes), fill_value=False
        )

        return feature_matrix.cummax(axis=0)

    def _compute_event_probabilities(
        self,
        feature_matrix: pd.DataFrame,
        trigger_codes: List[str],
        trigger_weights: List[float],
        p_daily_base: float,
    ) -> np.ndarray:
        """Computes daily event probabilities using vectorized operations."""
        weights_array = np.array(trigger_weights)
        trigger_matrix = feature_matrix[trigger_codes].values
        logit_p_days = logit(p_daily_base) + np.dot(trigger_matrix, weights_array)
        return expit(logit_p_days)

    def _simulate_and_add_event(
        self,
        subj_df: pd.DataFrame,
        daily_timeline: pd.DatetimeIndex,
        event_probabilities: np.ndarray,
        event_name: str,
    ) -> pd.DataFrame:
        """Simulates event occurrence and adds to dataframe if it occurs."""
        event_draws = np.random.binomial(1, event_probabilities)

        if event_draws.any():
            event_idx = np.argmax(event_draws)
            event_time = daily_timeline[event_idx]
            return self._add_events_to_dataframe(
                subj_df, pd.DatetimeIndex([event_time]), event_name
            )

        return subj_df

    def _compute_daily_prob(self, total_prob: float, num_days: int) -> float:
        """Converts total probability over a period into daily probability."""
        if num_days <= 0 or total_prob >= 1.0:
            return total_prob
        return 1 - (1 - total_prob) ** (1 / num_days)
