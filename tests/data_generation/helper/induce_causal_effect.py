import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit


class CausalSimulator:
    """
    Handles causal effect simulation for EHR data with multiple trigger codes.

    This class simulates binary exposure and outcome events based on causal relationships
    between multiple medical codes. All parameters are set during initialization.
    """

    def __init__(
        self,
        confounder_codes: List[str],
        confounder_exposure_weights: List[float],
        confounder_outcome_weights: List[float],
        instrument_codes: List[str],
        instrument_weights: List[float],
        prognostic_codes: List[str],
        prognostic_weights: List[float],
        p_base_exposure: float,
        p_base_outcome: float,
        p_daily_base_exposure: float,
        p_daily_base_outcome: float,
        exposure_outcome_effect: float,
        exposure_name: str = "EXPOSURE",
        outcome_name: str = "OUTCOME",
        simulate_outcome: bool = True,
    ):
        """
        Initialize the causal simulator with all parameters.

        Args:
            confounder_codes: List of codes that affect both exposure and outcome
            confounder_exposure_weights: Weights for confounder effects on exposure
            confounder_outcome_weights: Weights for confounder effects on outcome
            instrument_codes: List of codes that only affect exposure
            instrument_weights: Weights for instrument effects on exposure
            prognostic_codes: List of codes that only affect outcome
            prognostic_weights: Weights for prognostic effects on outcome
            p_base_exposure: Base probability for exposure events
            p_base_outcome: Base probability for outcome events
            p_daily_base_exposure: Base daily probability for exposure events
            p_daily_base_outcome: Base daily probability for outcome events
            exposure_outcome_effect: Causal effect of exposure on outcome
            exposure_name: Name for simulated exposure events
            outcome_name: Name for simulated outcome events
            simulate_outcome: Whether to simulate outcome events
        """
        # Store all parameters
        self.confounder_codes = confounder_codes
        self.confounder_exposure_weights = confounder_exposure_weights
        self.confounder_outcome_weights = confounder_outcome_weights
        self.instrument_codes = instrument_codes
        self.instrument_weights = instrument_weights
        self.prognostic_codes = prognostic_codes
        self.prognostic_weights = prognostic_weights

        self.p_base_exposure = p_base_exposure
        self.p_base_outcome = p_base_outcome
        self.p_daily_base_exposure = p_daily_base_exposure
        self.p_daily_base_outcome = p_daily_base_outcome

        self.exposure_outcome_effect = exposure_outcome_effect
        self.exposure_name = exposure_name
        self.outcome_name = outcome_name
        self.simulate_outcome = simulate_outcome

        # Validate parameter lengths
        if len(confounder_codes) != len(confounder_exposure_weights):
            raise ValueError(
                "Number of confounder codes must match number of exposure weights"
            )
        if len(confounder_codes) != len(confounder_outcome_weights):
            raise ValueError(
                "Number of confounder codes must match number of outcome weights"
            )
        if len(instrument_codes) != len(instrument_weights):
            raise ValueError("Number of instrument codes must match number of weights")
        if len(prognostic_codes) != len(prognostic_weights):
            raise ValueError("Number of prognostic codes must match number of weights")

        # Initialize ITE tracking
        self.ite_records = []

    def _calculate_and_store_ite(
        self,
        subj_df: pd.DataFrame,
        p_base_outcome: float,
        confounder_weights: list[float],
        prognostic_weights: list[float],
        exposure_outcome_effect: float,
    ):
        """
        Calculates the ground-truth Individual Treatment Effect (ITE) for a subject
        and stores it. This represents P(Outcome|Treated) - P(Outcome|Control).
        """
        subject_id = subj_df.iloc[0]["subject_id"]

        # Check for presence of baseline trigger codes
        confounder_present = [
            (subj_df["code"] == code).any() for code in self.confounder_codes
        ]
        prognostic_present = [
            (subj_df["code"] == code).any() for code in self.prognostic_codes
        ]

        # Effects without treatment
        base_effects = list(zip(confounder_weights, confounder_present)) + list(
            zip(prognostic_weights, prognostic_present)
        )

        # Logistic function to compute probabilities
        def compute_prob(base_prob, effects):
            logit_p = logit(base_prob) + sum(
                effect * int(present) for effect, present in effects
            )
            return expit(logit_p)

        # Calculate probability with and without treatment
        p_control = compute_prob(p_base_outcome, base_effects)
        p_treated = compute_prob(
            p_base_outcome, base_effects + [(exposure_outcome_effect, True)]
        )

        self.ite_records.append(
            {"subject_id": subject_id, "ite": p_treated - p_control}
        )

    def _run_daily_simulation(
        self,
        subj_df: pd.DataFrame,
        p_daily_base: float,
        trigger_codes: list[str],
        trigger_weights: list[float],
        event_name: str,
        start_after_event: str | None = None,
    ) -> pd.DataFrame:
        """
        Performs a vectorized, daily simulation and adds ALL successful event draws.
        """
        subj_df["time"] = pd.to_datetime(subj_df["time"])

        start_date = subj_df["time"].min().normalize()
        end_date = subj_df["time"].max().normalize()

        # If specified, start simulation the day after the FIRST occurrence of a prior event
        if start_after_event and (subj_df["code"] == start_after_event).any():
            first_occurrence = subj_df[subj_df["code"] == start_after_event][
                "time"
            ].min()
            start_date = first_occurrence.normalize() + pd.Timedelta(days=1)

        if start_date >= end_date:
            return subj_df

        # Create a daily timeline for the simulation period
        daily_timeline = pd.date_range(start=start_date, end=end_date, freq="D")
        if daily_timeline.empty:
            return subj_df

        # Create a feature matrix: rows are days, columns are codes.
        # A cell is True if the code has appeared on or before that day.
        events_pivot = subj_df.pivot_table(
            index="time", columns="code", aggfunc="size", fill_value=0
        ).astype(bool)
        feature_matrix = events_pivot.reindex(daily_timeline, method="ffill").fillna(
            False
        )

        # Ensure all trigger codes exist as columns in the matrix
        for code in trigger_codes:
            if code not in feature_matrix.columns:
                feature_matrix[code] = False
        feature_matrix = feature_matrix[trigger_codes]

        weights_s = pd.Series(trigger_weights, index=trigger_codes)

        # === Vectorized Simulation ===
        # 1. Calculate daily probabilities for all days at once
        logit_p_days = logit(p_daily_base) + feature_matrix.dot(weights_s)
        p_days = expit(logit_p_days)

        # 2. Perform all random draws simultaneously
        event_draws = np.random.binomial(1, p_days)

        # 3. Add all successful draws to the dataframe
        if event_draws.any():
            event_times = p_days.index[event_draws == 1]

            new_events_df = pd.DataFrame(
                {
                    "subject_id": subj_df.iloc[0]["subject_id"],
                    "time": event_times,
                    "code": event_name,
                }
            )

            # Combine original data with new events and sort chronologically
            return (
                pd.concat([subj_df, new_events_df], ignore_index=True)
                .sort_values("time")
                .reset_index(drop=True)
            )

        return subj_df

    def simulate_dataset(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the full daily causal simulation to the entire dataset.
        """
        self.ite_records = []

        # === Define Trigger and Weight Sets ===
        # For EXPOSURE: Confounders + Instruments
        exposure_triggers = self.confounder_codes + self.instrument_codes
        exposure_weights = self.confounder_exposure_weights + self.instrument_weights

        # For OUTCOME: Confounders + Prognostic Factors + The Exposure itself
        outcome_triggers = (
            self.confounder_codes + self.prognostic_codes + [self.exposure_name]
        )
        outcome_weights = (
            self.confounder_outcome_weights
            + self.prognostic_weights
            + [self.exposure_outcome_effect]
        )

        all_subject_dfs = []
        for subject_id, subj_df in df.groupby("subject_id"):
            subj_df = subj_df.sort_values("time").reset_index(drop=True)

            # 1. Calculate and store ground-truth ITE for this subject
            if self.simulate_outcome:
                self._calculate_and_store_ite(
                    subj_df,
                    self.p_base_outcome,
                    self.confounder_outcome_weights,
                    self.prognostic_weights,
                    self.exposure_outcome_effect,
                )

            # 2. Simulate EXPOSURE events
            df_with_exposure = self._run_daily_simulation(
                subj_df,
                self.p_daily_base_exposure,
                exposure_triggers,
                exposure_weights,
                self.exposure_name,
            )

            # 3. Simulate OUTCOME events (if enabled)
            final_df = df_with_exposure
            if self.simulate_outcome:
                final_df = self._run_daily_simulation(
                    df_with_exposure,
                    self.p_daily_base_outcome,
                    outcome_triggers,
                    outcome_weights,
                    self.outcome_name,
                    start_after_event=self.exposure_name,
                )

            all_subject_dfs.append(final_df)

        # === Finalize Results ===
        result_df = pd.concat(all_subject_dfs, ignore_index=True)
        ite_df = pd.DataFrame(self.ite_records)

        return result_df, ite_df


class DataManager:
    """Handles data loading and saving operations for simulation."""

    @staticmethod
    def load_shards(shard_dir: str) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """Load and concatenate parquet shards from directory."""
        if not os.path.exists(shard_dir):
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

        parquet_files = [f for f in os.listdir(shard_dir) if f.endswith(".parquet")]
        if not parquet_files:
            raise ValueError(f"No parquet files found in {shard_dir}")

        dfs, shards = [], {}
        for i, filename in enumerate(parquet_files):
            file_path = os.path.join(shard_dir, filename)
            try:
                shard = pd.read_parquet(file_path)
                shards[i] = shard.subject_id.unique().tolist()
                dfs.append(shard)
            except Exception as e:
                raise ValueError(f"Error reading {file_path}: {e}")

        return pd.concat(dfs), shards

    @staticmethod
    def write_shards(
        df: pd.DataFrame, write_dir: str, shards: Dict[int, List[str]]
    ) -> None:
        """Write DataFrame as sharded parquet files."""
        os.makedirs(write_dir, exist_ok=True)

        for shard_id, subject_ids in shards.items():
            shard_df = df[df.subject_id.isin(subject_ids)]
            shard_df.to_parquet(os.path.join(write_dir, f"{shard_id}.parquet"))


class SimulationReporter:
    """Generates simulation statistics and reports."""

    @staticmethod
    def print_trigger_stats(
        df: pd.DataFrame, simulator: CausalSimulator, simulate_outcome: bool = True
    ) -> None:
        """Print statistics about trigger code presence before simulation."""
        total_subjects = df["subject_id"].nunique()
        subject_codes = df.groupby("subject_id")["code"].apply(set)

        print(f"\nTotal subjects: {total_subjects}")
        print("\nTrigger code presence before simulation:")

        # Count subjects with confounder codes
        for i, code in enumerate(simulator.confounder_codes):
            count = subject_codes.apply(lambda codes: code in codes).sum()
            print(
                f"  Confounder {i + 1} ({code}): {count} subjects ({100 * count / total_subjects:.1f}%)"
            )

        # Count subjects with instrument codes
        for i, code in enumerate(simulator.instrument_codes):
            count = subject_codes.apply(lambda codes: code in codes).sum()
            print(
                f"  Instrument {i + 1} ({code}): {count} subjects ({100 * count / total_subjects:.1f}%)"
            )

        # Count subjects with prognostic codes
        if simulate_outcome:
            for i, code in enumerate(simulator.prognostic_codes):
                count = subject_codes.apply(lambda codes: code in codes).sum()
                print(
                    f"  Prognostic {i + 1} ({code}): {count} subjects ({100 * count / total_subjects:.1f}%)"
                )

    @staticmethod
    def print_simulation_results(
        df: pd.DataFrame, simulator: CausalSimulator, simulate_outcome: bool = True
    ) -> None:
        """Print simulation results statistics."""
        total_subjects = df["subject_id"].nunique()

        # Count simulated events
        exposure_subjects = df.groupby("subject_id")["code"].apply(
            lambda codes: (codes == simulator.exposure_name).any()
        )
        exposure_count = exposure_subjects.sum()

        print("\nSimulation results:")
        print(
            f"  EXPOSURE events: {exposure_count} subjects ({100 * exposure_count / total_subjects:.1f}%)"
        )

        if simulate_outcome:
            outcome_subjects = df.groupby("subject_id")["code"].apply(
                lambda codes: (codes == simulator.outcome_name).any()
            )
            outcome_count = outcome_subjects.sum()

            # Conditional probabilities
            outcomes_given_exposure = outcome_subjects[exposure_subjects].mean() * 100
            outcomes_given_no_exposure = (
                outcome_subjects[~exposure_subjects].mean() * 100
            )

            print(
                f"  OUTCOME events: {outcome_count} subjects ({100 * outcome_count / total_subjects:.1f}%)"
            )
            print(f"  P(Outcome | Exposure): {outcomes_given_exposure:.1f}%")
            print(f"  P(Outcome | No Exposure): {outcomes_given_no_exposure:.1f}%")
