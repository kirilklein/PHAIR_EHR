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

    def simulate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the full causal simulation to the entire dataset."""
        all_subject_dfs = []
        all_ite_records = []

        for _, subj_df in tqdm(df.groupby("subject_id"), desc="Simulating dataset"):
            # Sort with NaT values first (background variables)
            subj_df_sorted = subj_df.sort_values(
                "time", na_position="first"
            ).reset_index(drop=True)

            # Stage 1: Simulate the full exposure process
            df_with_exposure = self._simulate_exposure_process(subj_df_sorted)

            # Stage 2: Simulate outcomes and calculate ITE in one step
            final_df, ite_record = self._simulate_outcomes_and_get_ite(df_with_exposure)

            all_subject_dfs.append(final_df)
            if ite_record:  # Only add if not empty
                all_ite_records.append(ite_record)

        # Combine all subject data
        simulated_df = pd.concat(all_subject_dfs, ignore_index=True)

        # Combine all ITE records
        ite_df = pd.DataFrame(all_ite_records) if all_ite_records else pd.DataFrame()

        return simulated_df, ite_df

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
        start_date = subj_df["time"].min().normalize()
        end_date = subj_df["time"].max().normalize()

        sim_window_start = start_date + pd.Timedelta(days=run_in_days)
        if sim_window_start >= end_date:
            return subj_df

        k_days = (end_date - sim_window_start).days
        daily_timeline = pd.date_range(start=sim_window_start, end=end_date, freq="D")

        p_daily_base = self._compute_daily_prob(p_total_base, k_days)

        # Create feature matrix where a trigger, once True, stays True
        events_pivot = subj_df.pivot_table(
            index="time", columns="code", aggfunc="size", fill_value=0
        ).astype(bool)
        # Reindex to the full daily timeline
        feature_matrix = events_pivot.reindex(daily_timeline, fill_value=False)
        # Apply cumulative max to ensure codes stay True after appearing
        feature_matrix = feature_matrix.cummax(axis=0)

        for code in trigger_codes:
            if code not in feature_matrix.columns:
                feature_matrix[code] = False

        weights_s = pd.Series(trigger_weights, index=trigger_codes)
        logit_p_days = logit(p_daily_base) + feature_matrix[trigger_codes].dot(
            weights_s
        )
        # Convert to numpy array before applying expit
        p_days = expit(logit_p_days.astype(float))

        event_draws = np.random.binomial(1, p_days.values)

        if event_draws.any():
            event_idx = np.argmax(event_draws)
            event_time = p_days.index[event_idx]

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

    def _simulate_exposure_process(self, subj_df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrates the full, multi-phase exposure simulation for a subject."""
        cfg = self.config.exposure  # Use the exposure config

        # 1. Find the first exposure event using the simplified trigger structure
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

        # 2. Generate compliant exposures
        compliance_end_date = first_exposure_date + pd.Timedelta(
            days=cfg.run_in_days
        )  # Default compliance period
        exposure_dates = pd.date_range(
            start=first_exposure_date,
            end=compliance_end_date,
            freq=f"{cfg.compliance_interval_days}D",
        )

        # 3. Simulate discontinuation
        stop_window_start = compliance_end_date + pd.Timedelta(
            days=cfg.compliance_interval_days
        )
        if stop_window_start < subj_df["time"].max():
            stop_duration = (subj_df["time"].max() - stop_window_start).days
            stop_draws = np.random.binomial(1, cfg.daily_stop_prob, size=stop_duration)
            if stop_draws.any():
                stop_date = stop_window_start + pd.Timedelta(days=np.argmax(stop_draws))
                exposure_dates = exposure_dates[exposure_dates < stop_date]

        # 4. Add all final exposure events to the dataframe
        df_without_temp = df_with_first_exposure[
            df_with_first_exposure["code"] != f"TEMP_{cfg.code}"
        ]
        if not exposure_dates.empty:
            new_events_df = pd.DataFrame(
                {
                    "subject_id": subj_df.iloc[0]["subject_id"],
                    "time": exposure_dates,
                    "code": cfg.code,
                }
            )
            return (
                pd.concat([df_without_temp, new_events_df], ignore_index=True)
                .sort_values("time", na_position="first")
                .reset_index(drop=True)
            )

        return df_without_temp

    def _simulate_outcomes_and_get_ite(
        self, subj_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Simulates outcomes using a simplified single-point-in-time model and
        calculates the Individual Treatment Effect (ITE) for each outcome.
        """
        if subj_df.empty:
            return subj_df, {}

        # Handle NaT values for background variables - assign them to the very beginning
        subj_df_processed = subj_df.copy()
        valid_times = subj_df_processed["time"].dropna()

        if len(valid_times) == 0:
            return subj_df, {}

        earliest_time = valid_times.min()
        # Assign background variables (NaT) to a time before the earliest event
        background_time = earliest_time - pd.Timedelta(days=1)
        subj_df_processed.loc[subj_df_processed["time"].isna(), "time"] = (
            background_time
        )

        start_date = subj_df_processed["time"].min().normalize()
        end_date = subj_df_processed["time"].max().normalize()
        subject_id = subj_df_processed.iloc[0]["subject_id"]

        # If there's no time window, no simulation can occur
        if start_date >= end_date:
            return subj_df, {}

        df_with_outcomes = subj_df.copy()
        ite_record = {"subject_id": subject_id}

        # Select a single random day for assessment for this patient
        # Only consider the period after run_in_days for the first outcome (or use a default)
        first_outcome = list(self.config.outcomes.values())[0]
        assessment_start = start_date + pd.Timedelta(days=first_outcome.run_in_days)

        if assessment_start >= end_date:
            # If no valid assessment window, set ITE to 0 for all outcomes
            for outcome_cfg in self.config.outcomes.values():
                ite_record[f"ite_{outcome_cfg.code}"] = 0.0
            return subj_df, ite_record

        total_days = (end_date - assessment_start).days
        if total_days <= 0:
            assessment_time = assessment_start
        else:
            random_offset = np.random.randint(0, total_days + 1)
            assessment_time = assessment_start + pd.Timedelta(days=random_offset)

        # Check if assessment time is after DOD (end_date represents last recorded event/DOD)
        if assessment_time > end_date:
            # Assessment time is after DOD, skip this patient
            for outcome_cfg in self.config.outcomes.values():
                ite_record[f"ite_{outcome_cfg.code}"] = 0.0
            return subj_df, ite_record

        # Get all codes that appeared on or before the assessment time
        history_df = subj_df_processed[subj_df_processed["time"] <= assessment_time]
        history_codes = set(history_df["code"])

        # Loop through each outcome defined in the config
        for outcome_cfg in self.config.outcomes.values():
            # Check for presence of triggers at the assessment time
            trigger_effects = []
            for i, trigger_code in enumerate(outcome_cfg.trigger_codes):
                is_present = trigger_code in history_codes
                weight = outcome_cfg.trigger_weights[i]
                trigger_effects.append(weight if is_present else 0.0)

            # Factual scenario: was the patient actually exposed at assessment time?
            factual_exposure_present = self.config.exposure.code in history_codes

            # Helper to compute probability based on exposure status
            def _compute_prob(is_exposed: bool) -> float:
                logit_p = logit(outcome_cfg.p_base)

                # Add effects from trigger codes
                logit_p += sum(trigger_effects)

                # Add exposure effect if exposed
                if is_exposed:
                    logit_p += outcome_cfg.exposure_effect

                return expit(logit_p)

            # Calculate probabilities for both scenarios
            p_factual = _compute_prob(is_exposed=factual_exposure_present)
            p_counterfactual = _compute_prob(
                is_exposed=False
            )  # Always no exposure for counterfactual

            # Store the ITE (difference in probabilities)
            ite_record[f"ite_{outcome_cfg.code}"] = p_factual - p_counterfactual

            # --- Factual Simulation ---
            # Use the factual probability to simulate if the outcome occurs
            if np.random.binomial(1, p_factual):
                new_event = pd.DataFrame(
                    {
                        "subject_id": [subject_id],
                        "time": [assessment_time],
                        "code": [outcome_cfg.code],
                    }
                )
                df_with_outcomes = (
                    pd.concat([df_with_outcomes, new_event], ignore_index=True)
                    .sort_values("time", na_position="first")
                    .reset_index(drop=True)
                )

        return df_with_outcomes, ite_record
