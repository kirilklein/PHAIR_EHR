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

    def _simulate_all_outcomes(
        self, subj_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Orchestrates the simulation of all outcomes for a subject.
        """
        if (
            subj_df.empty
            or "time" not in subj_df.columns
            or subj_df["time"].dropna().empty
        ):
            return subj_df, {}

        # --- Initial Setup ---
        start_date = subj_df["time"].dropna().min().normalize()
        end_date = subj_df["time"].dropna().max().normalize()

        exposure_events = subj_df[subj_df["code"] == self.config.exposure.code]
        has_exposure = not exposure_events.empty
        first_exposure_date = exposure_events["time"].min() if has_exposure else None

        df_with_outcomes = subj_df.copy()
        ite_record = {"subject_id": subj_df.iloc[0]["subject_id"]}

        # --- Loop Through Each Outcome ---
        for outcome_cfg in self.config.outcomes.values():
            # Define the start of the assessment window based on exposure status
            if has_exposure:
                assessment_window_start = first_exposure_date.normalize()
            else:
                assessment_window_start = start_date + pd.Timedelta(
                    days=outcome_cfg.run_in_days
                )

            # Skip if there's no valid window for assessment
            if assessment_window_start >= end_date:
                ite_record[f"ite_{outcome_cfg.code}"] = 0.0
                continue

            # Simulate this single outcome and get the ITE
            df_with_outcomes, ite_value = self._simulate_single_outcome(
                df_with_outcomes, outcome_cfg, assessment_window_start, end_date
            )
            ite_record[f"ite_{outcome_cfg.code}"] = ite_value

        return df_with_outcomes, ite_record

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

        # Vectorized reindex operation to include all trigger codes at once
        all_trigger_codes = set(trigger_codes)
        existing_codes = set(events_pivot.columns)
        missing_codes = all_trigger_codes - existing_codes

        # Reindex to the full daily timeline and add missing codes in one operation
        feature_matrix = events_pivot.reindex(
            index=daily_timeline,
            columns=list(existing_codes) + list(missing_codes),
            fill_value=False,
        )

        # Apply cumulative max to ensure codes stay True after appearing
        feature_matrix = feature_matrix.cummax(axis=0)

        # Vectorized computation using numpy arrays
        weights_array = np.array(trigger_weights)
        trigger_matrix = feature_matrix[trigger_codes].values

        logit_p_days = logit(p_daily_base) + np.dot(trigger_matrix, weights_array)
        p_days = expit(logit_p_days)

        event_draws = np.random.binomial(1, p_days)

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
