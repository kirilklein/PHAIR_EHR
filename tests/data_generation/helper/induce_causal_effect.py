import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit


class CausalSimulator:
    """Handles causal effect simulation for EHR data."""

    def __init__(
        self,
        confounder_code: str,
        exposure_only_code: str,
        outcome_only_code: str,
        exposure_name: str = "EXPOSURE",
        outcome_name: str = "OUTCOME",
    ):
        self.confounder_code = confounder_code
        self.exposure_only_code = exposure_only_code
        self.outcome_only_code = outcome_only_code
        self.exposure_name = exposure_name
        self.outcome_name = outcome_name

    def _compute_probability(
        self, base_prob: float, effects: List[Tuple[float, bool]]
    ) -> float:
        """Compute event probability using logistic function."""
        logit_p = logit(base_prob) + sum(
            effect * int(present) for effect, present in effects
        )
        return expit(logit_p)

    def _get_event_time(
        self,
        subj_df: pd.DataFrame,
        trigger_codes: List[str],
        days_offset: int,
        fallback_strategy: str = "random",
    ) -> pd.Timestamp:
        """Determine timing for new event based on triggers."""
        # Find latest trigger time if any triggers present
        trigger_times = []
        for code in trigger_codes:
            if (subj_df["code"] == code).any():
                trigger_times.append(subj_df.loc[subj_df["code"] == code, "time"].max())

        if trigger_times:
            return max(trigger_times) + pd.Timedelta(days=days_offset)

        # Fallback strategies when no triggers present
        if len(subj_df) == 0:
            return pd.Timestamp.now()

        if fallback_strategy == "random":
            return subj_df.iloc[random.randint(0, len(subj_df) - 1)]["time"]
        else:  # median
            return subj_df.iloc[len(subj_df) // 2]["time"]

    def _add_event(
        self,
        subj_df: pd.DataFrame,
        event_code: str,
        event_time: pd.Timestamp,
        **extra_cols,
    ) -> pd.DataFrame:
        """Add new event to subject DataFrame."""
        new_event = pd.DataFrame(
            {
                "subject_id": [
                    subj_df.iloc[0]["subject_id"] if len(subj_df) > 0 else "unknown"
                ],
                "time": [event_time],
                "code": [event_code],
                "numeric_value": [np.nan],
                **extra_cols,
            }
        )

        combined_df = pd.concat([subj_df, new_event], ignore_index=True)
        return combined_df.sort_values("time").reset_index(drop=True)

    def simulate_exposure(
        self,
        subj_df: pd.DataFrame,
        p_base: float,
        confounder_effect: float,
        exposure_only_effect: float,
        days_offset: int,
    ) -> pd.DataFrame:
        """Simulate exposure event for a single subject."""
        subj_df = subj_df.sort_values("time").reset_index(drop=True)

        # Check trigger presence
        confounder_present = (subj_df["code"] == self.confounder_code).any()
        exposure_only_present = (subj_df["code"] == self.exposure_only_code).any()

        # Compute probability and simulate
        effects = [
            (confounder_effect, confounder_present),
            (exposure_only_effect, exposure_only_present),
        ]
        prob = self._compute_probability(p_base, effects)

        if not np.random.binomial(1, prob):
            return subj_df

        # Determine event timing
        trigger_codes = [
            code
            for code, present in [
                (self.confounder_code, confounder_present),
                (self.exposure_only_code, exposure_only_present),
            ]
            if present
        ]

        event_time = self._get_event_time(subj_df, trigger_codes, days_offset)
        return self._add_event(subj_df, self.exposure_name, event_time)

    def simulate_outcome(
        self,
        subj_df: pd.DataFrame,
        p_base: float,
        confounder_effect: float,
        outcome_only_effect: float,
        exposure_outcome_effect: float,
        days_offset: int,
    ) -> pd.DataFrame:
        """Simulate outcome event for a single subject."""
        subj_df = subj_df.sort_values("time").reset_index(drop=True)

        # Check trigger presence
        confounder_present = (subj_df["code"] == self.confounder_code).any()
        outcome_only_present = (subj_df["code"] == self.outcome_only_code).any()
        exposure_present = (subj_df["code"] == self.exposure_name).any()

        # Compute probabilities for treated/control
        base_effects = [
            (confounder_effect, confounder_present),
            (outcome_only_effect, outcome_only_present),
        ]

        p_control = self._compute_probability(p_base, base_effects)
        p_treated = self._compute_probability(
            p_base, base_effects + [(exposure_outcome_effect, True)]
        )

        ite = p_treated - p_control
        p_actual = p_treated if exposure_present else p_control

        if not np.random.binomial(1, p_actual):
            return subj_df

        # Determine event timing
        trigger_codes = []
        if exposure_present:
            trigger_codes = [self.exposure_name]
        else:
            trigger_codes = [
                code
                for code, present in [
                    (self.confounder_code, confounder_present),
                    (self.outcome_only_code, outcome_only_present),
                ]
                if present
            ]

        event_time = self._get_event_time(subj_df, trigger_codes, days_offset, "median")
        return self._add_event(subj_df, self.outcome_name, event_time, ite=ite)

    def simulate_dataset(
        self,
        df: pd.DataFrame,
        p_base_exposure: float,
        p_base_outcome: float,
        confounder_exposure_effect: float,
        confounder_outcome_effect: float,
        exposure_only_effect: float,
        outcome_only_effect: float,
        exposure_outcome_effect: float,
        days_offset: int,
        simulate_outcome: bool = True,
    ) -> pd.DataFrame:
        """Apply causal simulation to entire dataset."""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        # Simulate exposures
        result_df = df.groupby("subject_id", group_keys=False).apply(
            lambda x: self.simulate_exposure(
                x,
                p_base_exposure,
                confounder_exposure_effect,
                exposure_only_effect,
                days_offset,
            )
        )

        # Simulate outcomes if requested
        if simulate_outcome:
            result_df = result_df.groupby("subject_id", group_keys=False).apply(
                lambda x: self.simulate_outcome(
                    x,
                    p_base_outcome,
                    confounder_outcome_effect,
                    outcome_only_effect,
                    exposure_outcome_effect,
                    days_offset,
                )
            )

        return result_df


class DataManager:
    """Handles data loading and saving operations."""

    @staticmethod
    def load_shards(shard_dir: str) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """Load and concatenate parquet shards."""
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
        """Print statistics about trigger code presence."""
        total_subjects = df["subject_id"].nunique()
        subject_codes = df.groupby("subject_id")["code"].apply(set)

        # Count subjects with each trigger
        confounder_count = subject_codes.apply(
            lambda codes: simulator.confounder_code in codes
        ).sum()
        exposure_count = subject_codes.apply(
            lambda codes: simulator.exposure_only_code in codes
        ).sum()

        print(f"\nTotal subjects: {total_subjects}")
        print("\nTrigger code presence before simulation:")
        print(
            f"  Confounder ({simulator.confounder_code}): {confounder_count} subjects ({100 * confounder_count / total_subjects:.1f}%)"
        )
        print(
            f"  Exposure trigger ({simulator.exposure_only_code}): {exposure_count} subjects ({100 * exposure_count / total_subjects:.1f}%)"
        )

        if simulate_outcome:
            outcome_count = subject_codes.apply(
                lambda codes: simulator.outcome_only_code in codes
            ).sum()
            print(
                f"  Outcome trigger ({simulator.outcome_only_code}): {outcome_count} subjects ({100 * outcome_count / total_subjects:.1f}%)"
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

            # Calculate ATE
            if "ite" in df.columns:
                ate = df["ite"].mean()
                print(f"\nAverage Treatment Effect: {ate:.4f}")
