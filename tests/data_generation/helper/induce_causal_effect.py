from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit

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

        # Create feature matrix, ensuring all trigger codes are columns
        events_pivot = subj_df.pivot_table(
            index="time", columns="code", aggfunc="size", fill_value=0
        ).astype(bool)
        feature_matrix = events_pivot.reindex(daily_timeline, method="ffill").fillna(
            False
        )
        for code in trigger_codes:
            if code not in feature_matrix.columns:
                feature_matrix[code] = False

        # Run vectorized simulation
        weights_s = pd.Series(trigger_weights, index=trigger_codes)
        logit_p_days = logit(p_daily_base) + feature_matrix[trigger_codes].dot(
            weights_s
        )
        p_days = expit(logit_p_days)
        event_draws = np.random.binomial(1, p_days)

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
                .sort_values("time")
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
            days=30
        )  # Default compliance period
        exposure_dates = pd.date_range(
            start=first_exposure_date,
            end=compliance_end_date,
            freq=f"{cfg.compliance_interval_days}D",
        )

        # 3. Simulate discontinuation
        stop_window_start = compliance_end_date + pd.Timedelta(days=1)
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
                .sort_values("time")
                .reset_index(drop=True)
            )

        return df_without_temp

    def _simulate_all_outcomes(self, subj_df: pd.DataFrame) -> pd.DataFrame:
        """Iterates through and simulates all configured outcomes."""
        df_with_outcomes = subj_df.copy()

        # Loop through each outcome's config (it's a dictionary)
        for outcome_key, outcome_cfg in self.config.outcomes.items():
            # For each outcome, use its trigger_codes and trigger_weights, plus add exposure effect
            outcome_triggers = outcome_cfg.trigger_codes + [self.config.exposure.code]
            outcome_weights = outcome_cfg.trigger_weights + [
                outcome_cfg.exposure_effect
            ]

            df_with_outcomes = self._simulate_time_to_first_event(
                df_with_outcomes,
                outcome_cfg.p_base,
                outcome_triggers,
                outcome_weights,
                outcome_cfg.code,
                outcome_cfg.run_in_days,
            )
        return df_with_outcomes

    def _calculate_ite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Individual Treatment Effects for each outcome."""
        ite_records = []

        for subject_id, subj_df in df.groupby("subject_id"):
            ite_record = {"subject_id": subject_id}

            # Check if subject received exposure
            exposure_events = subj_df[subj_df["code"] == self.config.exposure.code]
            has_exposure = len(exposure_events) > 0

            if has_exposure:
                exposure_date = exposure_events["time"].min()

                # Calculate ITE for each outcome
                for outcome_key, outcome_cfg in self.config.outcomes.items():
                    outcome_code = outcome_cfg.code

                    # Check if outcome occurred after exposure
                    outcome_events = subj_df[subj_df["code"] == outcome_code]
                    if len(outcome_events) > 0:
                        outcome_date = outcome_events["time"].min()
                        outcome_after_exposure = outcome_date > exposure_date
                    else:
                        outcome_after_exposure = False

                    # Simple ITE calculation - this would be more sophisticated in practice
                    # For now, using the exposure effect as a proxy
                    ite_value = (
                        outcome_cfg.exposure_effect if outcome_after_exposure else 0
                    )
                    ite_record[f"ite_{outcome_code}"] = ite_value
            else:
                # No exposure, so ITE is 0 for all outcomes
                for outcome_cfg in self.config.outcomes.values():
                    ite_record[f"ite_{outcome_cfg.code}"] = 0

            ite_records.append(ite_record)

        return pd.DataFrame(ite_records)

    def simulate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the full causal simulation to the entire dataset."""
        all_subject_dfs = []

        for _, subj_df in df.groupby("subject_id"):
            subj_df_sorted = subj_df.sort_values("time").reset_index(drop=True)

            # Stage 1: Simulate the full exposure process
            df_with_exposure = self._simulate_exposure_process(subj_df_sorted)

            # Stage 2: Simulate all configured outcomes
            final_df = self._simulate_all_outcomes(df_with_exposure)

            all_subject_dfs.append(final_df)

        # Combine all subject data
        simulated_df = pd.concat(all_subject_dfs, ignore_index=True)

        # Calculate ITE for each outcome
        ite_df = self._calculate_ite(simulated_df)

        return simulated_df, ite_df
