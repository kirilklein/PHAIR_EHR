from typing import List, Tuple

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

    def __init__(self, config: SimulationConfig):
        """
        Initializes the simulator with a single configuration object.

        Args:
            config: A SimulationConfig object containing all parameters.
        """
        self.config = config
        self.ite_records = []  # Individual Treatment Effect records
        self.first_exposure_dates = []

    def simulate_dataset(
        self, df: pd.DataFrame, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the full causal simulation to the entire dataset."""
        np.random.seed(seed)

        # Pass 1: Simulate exposures for all subjects and collect exposure dates
        all_subjects_with_exposure = []

        for _, subj_df in tqdm(df.groupby("subject_id"), desc="Simulating exposures"):
            subj_df_sorted = subj_df.sort_values(
                "time", na_position="first"
            ).reset_index(drop=True)
            df_with_exposure = self._simulate_exposure_process(subj_df_sorted)
            all_subjects_with_exposure.append(df_with_exposure)

        # Pass 2: Simulate outcomes for all subjects using complete exposure distribution
        all_subject_dfs = []
        all_ite_records = []

        for df_with_exposure in tqdm(
            all_subjects_with_exposure, desc="Simulating outcomes"
        ):
            final_df, ite_record = self._simulate_all_outcomes(df_with_exposure)
            all_subject_dfs.append(final_df)
            if ite_record:  # Only add if not empty
                all_ite_records.append(ite_record)

        # Combine all subject data
        simulated_df = pd.concat(all_subject_dfs, ignore_index=True)

        # Combine all ITE records
        ite_df = pd.DataFrame(all_ite_records) if all_ite_records else pd.DataFrame()

        return simulated_df, ite_df

    def _simulate_exposure_process(self, subj_df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrates the full, multi-phase exposure simulation for a subject."""
        cfg = self.config.exposure

        # 1. Find the first exposure event
        df_with_first_exposure = self._simulate_time_to_first_event(
            subj_df,
            cfg.p_base,
            cfg.trigger_codes,
            cfg.trigger_weights,
            f"TEMP_{cfg.code}",
            cfg.run_in_days,
        )

        first_exposure_events = df_with_first_exposure[
            df_with_first_exposure["code"] == f"TEMP_{cfg.code}"
        ]
        if first_exposure_events.empty:
            return subj_df

        first_exposure_date = first_exposure_events["time"].min()
        self.first_exposure_dates.append(first_exposure_date)

        # 2. Randomly determine when compliance ends
        end_date = subj_df["time"].max()
        compliance_end_date = self._get_random_compliance_end_date(
            first_exposure_date, end_date
        )

        # 3. Generate regular exposures between first exposure and compliance end
        exposure_dates = self._generate_regular_exposures(
            first_exposure_date, compliance_end_date, cfg.compliance_interval_days
        )

        # 4. Add all exposure events to the dataframe
        return self._add_exposure_events_to_dataframe(
            df_with_first_exposure, exposure_dates, cfg.code
        )

    def _get_random_compliance_end_date(
        self, first_exposure_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Randomly selects a compliance end date between first exposure and end of timeline.
        """
        # Ensure there's at least some minimum compliance period
        min_compliance_days = 1
        latest_possible_end = end_date
        earliest_possible_end = first_exposure_date + pd.Timedelta(
            days=min_compliance_days
        )

        if earliest_possible_end >= latest_possible_end:
            return latest_possible_end

        # Random uniform selection between earliest and latest possible end dates
        total_days = (latest_possible_end - earliest_possible_end).days
        random_days = np.random.randint(0, total_days + 1)

        return earliest_possible_end + pd.Timedelta(days=random_days)

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

    def _add_exposure_events_to_dataframe(
        self,
        df_with_first_exposure: pd.DataFrame,
        exposure_dates: pd.DatetimeIndex,
        exposure_code: str,
    ) -> pd.DataFrame:
        """
        Adds the final exposure events to the dataframe, removing temporary events.
        """
        # Remove the temporary first exposure event
        df_without_temp = df_with_first_exposure[
            df_with_first_exposure["code"] != f"TEMP_{exposure_code}"
        ]

        if exposure_dates.empty:
            return df_without_temp

        # Add all final exposure events
        new_events_df = pd.DataFrame(
            {
                "subject_id": df_with_first_exposure.iloc[0]["subject_id"],
                "time": exposure_dates,
                "code": exposure_code,
            }
        )

        return (
            pd.concat([df_without_temp, new_events_df], ignore_index=True)
            .sort_values("time", na_position="first")
            .reset_index(drop=True)
        )

    def _simulate_all_outcomes(
        self, subj_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """Orchestrates outcome simulation and ITE calculation."""
        start_date = subj_df["time"].dropna().min().normalize()
        end_date = subj_df["time"].dropna().max().normalize()

        exposure_events = subj_df[subj_df["code"] == self.config.exposure.code]
        has_exposure = not exposure_events.empty
        first_exposure_date = exposure_events["time"].min() if has_exposure else None

        df_with_outcomes = subj_df.copy()
        ite_record = {
            "subject_id": subj_df.iloc[0]["subject_id"],
            "has_exposure": int(has_exposure),
        }

        # Simulate each outcome
        for outcome_cfg in self.config.outcomes.values():
            assessment_time = self._get_assessment_time(
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

    def _get_assessment_time(
        self,
        has_exposure: bool,
        first_exposure_date: pd.Timestamp,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        run_in_days: int,
    ) -> pd.Timestamp:
        """
        Determines the assessment start date ensuring enough room for run-in period.

        Returns None if no valid assessment window can be found.
        """

        def _within_timeline(
            date: pd.Timestamp,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp,
            run_in_days: int,
        ) -> bool:
            return (date + pd.Timedelta(days=run_in_days) <= end_date) and (
                date + pd.Timedelta(days=run_in_days) >= start_date
            )

        fallback_date = end_date - pd.Timedelta(days=1)
        if has_exposure:
            assessment_start = first_exposure_date.normalize()
            if _within_timeline(assessment_start, start_date, end_date, run_in_days):
                return assessment_start + pd.Timedelta(days=run_in_days)
            else:
                return fallback_date
        else:
            # For unexposed subjects, use the complete exposure distribution
            if not self.first_exposure_dates:
                # Fallback if no subjects were exposed in the entire dataset
                return (
                    start_date + pd.Timedelta(days=run_in_days)
                    if start_date + pd.Timedelta(days=run_in_days) < end_date
                    else fallback_date
                )

            # Try to find a suitable date from exposure distribution
            for _ in range(100):  # Max attempts
                pseudo_exposure_date = np.random.choice(self.first_exposure_dates)
                if _within_timeline(
                    pseudo_exposure_date, start_date, end_date, run_in_days
                ):
                    return pseudo_exposure_date.normalize() + pd.Timedelta(
                        days=run_in_days
                    )
        return fallback_date

    def _simulate_single_outcome_complete(
        self, subj_df: pd.DataFrame, outcome_cfg, assessment_time: pd.Timestamp
    ) -> Tuple[pd.DataFrame, float]:
        """
        Simulates a single outcome: calculates ITE and simulates factual outcome.
        """
        # Get history up to assessment time
        history_mask = subj_df["time"] <= assessment_time
        history_codes = set(subj_df.loc[history_mask, "code"])
        is_exposed_at_assessment = self.config.exposure.code in history_codes

        # Calculate probabilities for ITE
        p_if_treated = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=True
        )
        p_if_control = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=False
        )
        ite = p_if_treated - p_if_control

        # Calculate factual probability
        p_factual = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed_at_assessment
        )

        # Simulate factual outcome
        if np.random.binomial(1, p_factual):
            # Add outcome event at assessment time
            new_event = pd.DataFrame(
                {
                    "subject_id": [subj_df.iloc[0]["subject_id"]],
                    "time": [assessment_time],
                    "code": [outcome_cfg.code],
                }
            )
            subj_df = (
                pd.concat([subj_df, new_event], ignore_index=True)
                .sort_values("time")
                .reset_index(drop=True)
            )

        return subj_df, ite

    def _compute_daily_prob(self, total_prob: float, num_days: int) -> float:
        """Converts a total probability over a period into a daily probability."""
        if num_days <= 0 or total_prob >= 1.0:
            return total_prob
        return 1 - (1 - total_prob) ** (1 / num_days)

    def _simulate_time_to_first_event(
        self,
        subj_df: pd.DataFrame,
        p_total_base: float,
        trigger_codes: List[str],
        trigger_weights: List[float],
        event_name: str,
        run_in_days: int,
    ) -> pd.DataFrame:
        """
        Generic, vectorized simulation to find the first occurrence of a single event.
        """
        timeline_info = self._setup_simulation_timeline(subj_df, run_in_days)
        if timeline_info is None:
            return subj_df

        daily_timeline, k_days = timeline_info
        n_possible_days = k_days - run_in_days
        p_daily_base = self._compute_daily_prob(p_total_base, n_possible_days)

        # Build feature matrix and compute probabilities
        feature_matrix = self._build_feature_matrix(
            subj_df, daily_timeline, trigger_codes
        )
        event_probabilities = self._compute_event_probabilities(
            feature_matrix, trigger_codes, trigger_weights, p_daily_base
        )
        # Simulate event occurrence
        return self._simulate_and_add_event(
            subj_df, daily_timeline, event_probabilities, event_name
        )

    def _setup_simulation_timeline(
        self, subj_df: pd.DataFrame, run_in_days: int
    ) -> Tuple[pd.DatetimeIndex, int]:
        """
        Sets up the simulation timeline and validates the window.

        Returns:
            Tuple of (daily_timeline, k_days) or None if no valid window
        """
        start_date = subj_df["time"].min().normalize()
        end_date = subj_df["time"].max().normalize()

        sim_window_start = start_date + pd.Timedelta(days=run_in_days)
        if sim_window_start >= end_date:
            return None

        k_days = (end_date - sim_window_start).days
        daily_timeline = pd.date_range(start=sim_window_start, end=end_date, freq="D")

        return daily_timeline, k_days

    def _build_feature_matrix(
        self,
        subj_df: pd.DataFrame,
        daily_timeline: pd.DatetimeIndex,
        trigger_codes: List[str],
    ) -> pd.DataFrame:
        """
        Creates a feature matrix where triggers, once True, stay True.
        """
        # Create pivot table of existing events
        events_pivot = subj_df.pivot_table(
            index="time", columns="code", aggfunc="size", fill_value=0
        ).astype(bool)

        # Identify missing trigger codes and add them
        all_trigger_codes = set(trigger_codes)
        existing_codes = set(events_pivot.columns)
        missing_codes = all_trigger_codes - existing_codes

        # Reindex to full timeline and include missing codes
        feature_matrix = events_pivot.reindex(
            index=daily_timeline,
            columns=list(existing_codes) + list(missing_codes),
            fill_value=False,
        )

        # Apply cumulative max to ensure codes stay True after appearing
        return feature_matrix.cummax(axis=0)

    def _compute_event_probabilities(
        self,
        feature_matrix: pd.DataFrame,
        trigger_codes: List[str],
        trigger_weights: List[float],
        p_daily_base: float,
    ) -> np.ndarray:
        """
        Computes daily event probabilities using vectorized operations.
        """
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
        """
        Simulates event occurrence and adds it to the dataframe if it occurs.
        """
        event_draws = np.random.binomial(1, event_probabilities)

        if event_draws.any():
            event_idx = np.argmax(event_draws)
            event_time = daily_timeline[event_idx]

            new_event_df = pd.DataFrame(
                {
                    "subject_id": [subj_df.iloc[0]["subject_id"]],
                    "time": [event_time],
                    "code": [event_name],
                }
            )

            return (
                pd.concat([subj_df, new_event_df], ignore_index=True)
                .sort_values("time", na_position="first")
                .reset_index(drop=True)
            )

        return subj_df

    def _calculate_outcome_probability(
        self, outcome_cfg, history_codes: set, is_exposed: bool
    ) -> float:
        """
        Calculates the absolute probability of a single outcome given a patient's history.
        """
        # Vectorized computation of trigger effects
        trigger_codes_array = np.array(outcome_cfg.trigger_codes)
        trigger_weights_array = np.array(outcome_cfg.trigger_weights)

        # Create boolean mask for codes present in history
        codes_present_mask = np.array(
            [code in history_codes for code in trigger_codes_array]
        )

        # Sum trigger effects using vectorized operations
        trigger_effect_sum = np.sum(trigger_weights_array[codes_present_mask])

        # Start with the base probability and add all effects on the logit scale
        logit_p = logit(outcome_cfg.p_base) + trigger_effect_sum
        if is_exposed:
            logit_p += outcome_cfg.exposure_effect

        return expit(logit_p)

    def _simulate_single_outcome(
        self,
        subj_df: pd.DataFrame,
        outcome_cfg,
        assessment_window_start: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Simulates one outcome for a subject and calculates its ITE.

        Returns:
            A tuple containing the updated dataframe and the ITE value.
        """
        # 1. Select a random assessment time within the valid window
        total_assessment_days = (end_date - assessment_window_start).days
        random_offset = np.random.randint(0, total_assessment_days + 1)
        assessment_time = assessment_window_start + pd.Timedelta(days=random_offset)

        # 2. Check history at the assessment time - optimized using vectorized comparison
        history_mask = subj_df["time"] <= assessment_time
        history_codes = set(subj_df.loc[history_mask, "code"])
        factual_exposure_present = self.config.exposure.code in history_codes

        # 3. Calculate factual and counterfactual probabilities
        p_factual = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=factual_exposure_present
        )
        p_counterfactual = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=False
        )

        # 4. Calculate ITE
        ite = p_factual - p_counterfactual

        # 5. Simulate the factual outcome
        if np.random.binomial(1, p_factual):
            new_event = pd.DataFrame(
                {
                    "subject_id": [subj_df.iloc[0]["subject_id"]],
                    "time": [assessment_time],
                    "code": [outcome_cfg.code],
                }
            )
            subj_df = (
                pd.concat([subj_df, new_event], ignore_index=True)
                .sort_values("time")
                .reset_index(drop=True)
            )

        return subj_df, ite
