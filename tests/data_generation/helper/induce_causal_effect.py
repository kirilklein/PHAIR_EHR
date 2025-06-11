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
        outcome: bool = False,
    ) -> pd.Timestamp:
        """Determine timing for new event based on triggers, ensuring it's before DOD.

        Args:
            subj_df: Subject dataframe
            trigger_codes: List of codes that can trigger the event (e.g., confounder, exposure)
            days_offset: Days to add after the latest trigger
            fallback_strategy: Strategy when no triggers present ("random" or "second_half")
            outcome: Whether this is for outcome simulation (allows closer to DOD)

        Returns:
            Event timestamp that is guaranteed to be before DOD (if DOD exists), or None if impossible
        """
        # First, check for DOD and establish time boundary
        dod_time = None
        dod_events = subj_df[subj_df["code"] == "DOD"]
        if not dod_events.empty:
            dod_time = dod_events["time"].min()
            # Validate DOD time
            if pd.isna(dod_time):
                dod_time = None

        # Find trigger times (first occurrence of each trigger code)
        trigger_times = []
        for code in trigger_codes:
            trigger_events = subj_df[subj_df["code"] == code]
            if not trigger_events.empty:
                trigger_time = trigger_events["time"].min()
                # Only add valid (non-NaT) times that are before DOD
                if not pd.isna(trigger_time):
                    if dod_time is None or trigger_time < dod_time:
                        trigger_times.append(trigger_time)

        # Calculate proposed event time based on triggers
        if trigger_times:
            # Use latest trigger time + offset
            latest_trigger = max(trigger_times)
            proposed_time = latest_trigger + pd.Timedelta(days=days_offset)

            # Ensure proposed time is before DOD (with appropriate buffer)
            if dod_time is not None:
                min_buffer_hours = 1 if outcome else 24  # Outcomes can be closer to death
                buffer_time = dod_time - pd.Timedelta(hours=min_buffer_hours)
                
                # If proposed time would be after the buffer, check if it's feasible
                if proposed_time >= buffer_time:
                    # If there's insufficient time between trigger and DOD, reject
                    if buffer_time <= latest_trigger:
                        return None
                    # Otherwise, place just before the buffer
                    proposed_time = buffer_time

            return proposed_time

        # Fallback strategies when no triggers present
        if len(subj_df) == 0:
            return pd.Timestamp.now()

        # Establish time bounds for fallback
        min_time = subj_df["time"].min()
        max_time = subj_df["time"].max()

        # Validate min/max times
        if pd.isna(min_time) or pd.isna(max_time):
            return None

        # Adjust max_time if DOD is present - be more conservative
        if dod_time is not None:
            conservative_buffer = pd.Timedelta(days=7 if not outcome else 1)
            max_time = min(max_time, dod_time - conservative_buffer)

        # Ensure we have a valid time window
        if max_time <= min_time:
            return None

        if fallback_strategy == "second_half":
            median_idx = len(subj_df) // 2
            median_time = subj_df.iloc[median_idx]["time"]

            # Validate median time
            if pd.isna(median_time):
                median_time = min_time

            # Randomly place in second half, but before max_time
            start_time = max(median_time, min_time)
            if start_time >= max_time:
                return None

            # Generate random timestamp between start_time and max_time
            time_diff_seconds = (max_time - start_time).total_seconds()

            # Check for invalid time difference
            if pd.isna(time_diff_seconds) or time_diff_seconds <= 0:
                return None

            random_seconds = random.uniform(0, time_diff_seconds)
            return start_time + pd.Timedelta(seconds=random_seconds)
        else:
            # Place near end of timeline but before DOD with buffer
            return max_time - pd.Timedelta(hours=12)

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
    ) -> pd.DataFrame:
        """Simulate exposure event for a single subject."""
        subj_df = subj_df.sort_values("time").reset_index(drop=True)

        # CRITICAL: Check if patient has DOD and validate timeline
        dod_events = subj_df[subj_df["code"] == "DOD"]
        if not dod_events.empty:
            dod_time = dod_events["time"].min()
            
            # Check if triggers occur before DOD
            confounder_events = subj_df[subj_df["code"] == self.confounder_code]
            exposure_only_events = subj_df[subj_df["code"] == self.exposure_only_code]
            
            # If triggers exist, ensure they're before DOD
            valid_triggers = True
            if not confounder_events.empty:
                latest_confounder = confounder_events["time"].max()
                if latest_confounder >= dod_time:
                    valid_triggers = False
            
            if not exposure_only_events.empty:
                latest_exposure_trigger = exposure_only_events["time"].max()
                if latest_exposure_trigger >= dod_time:
                    valid_triggers = False
            
            # If triggers are invalid (after DOD), cannot simulate exposure
            if not valid_triggers:
                return subj_df
        
        # Check trigger presence (only valid triggers)
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

        # Determine trigger codes for timing
        trigger_codes = []
        if confounder_present:
            trigger_codes.append(self.confounder_code)
        if exposure_only_present:
            trigger_codes.append(self.exposure_only_code)

        event_time = self._get_event_time(
            subj_df,
            trigger_codes,
            days_offset=1,
            fallback_strategy="second_half",
            outcome=False,
        )

        # If event_time is None, we can't place the event safely
        if event_time is None:
            return subj_df
        
        # FINAL CHECK: Ensure proposed exposure time is before DOD
        if not dod_events.empty:
            dod_time = dod_events["time"].min()
            if event_time >= dod_time:
                return subj_df

        return self._add_event(subj_df, self.exposure_name, event_time)

    def simulate_outcome(
        self,
        subj_df: pd.DataFrame,
        p_base: float,
        confounder_effect: float,
        outcome_only_effect: float,
        exposure_outcome_effect: float,
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

        # Store ITE for this subject (regardless of outcome realization)
        subject_id = subj_df.iloc[0]["subject_id"]
        self.ite_records.append({"subject_id": subject_id, "ite": ite})

        p_actual = p_treated if exposure_present else p_control

        if not np.random.binomial(1, p_actual):
            return subj_df

        # Determine trigger codes for timing
        trigger_codes = []
        if confounder_present:
            trigger_codes.append(self.confounder_code)
        if outcome_only_present:
            trigger_codes.append(self.outcome_only_code)
        if exposure_present:
            trigger_codes.append(self.exposure_name)

        event_time = self._get_event_time(
            subj_df,
            trigger_codes,
            days_offset=1,
            fallback_strategy="latest",
            outcome=True,
        )

        # If event_time is None, we can't place the event (e.g., would be after DOD)
        if event_time is None:
            return subj_df

        return self._add_event(subj_df, self.outcome_name, event_time)

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
        simulate_outcome: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply causal simulation to entire dataset."""
        # Reset ITE records for new simulation
        self.ite_records = []

        # Track simulation statistics
        exposure_attempted = 0
        exposure_blocked_by_dod = 0
        exposure_invalid_triggers = 0
        outcome_attempted = 0
        outcome_blocked_by_dod = 0
        temporal_violations = 0

        # Ensure datetime format
        if not pd.api.types.is_datetime64_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        # Simulate exposures
        def simulate_exposure_with_tracking(x):
            nonlocal exposure_attempted, exposure_blocked_by_dod, exposure_invalid_triggers
            exposure_attempted += 1

            original_len = len(x)
            result = self.simulate_exposure(
                x,
                p_base_exposure,
                confounder_exposure_effect,
                exposure_only_effect,
            )

            # Check if exposure was added
            if len(result) == original_len:
                has_dod = (x["code"] == "DOD").any()
                if has_dod:
                    exposure_blocked_by_dod += 1
                else:
                    exposure_invalid_triggers += 1

            return result

        result_df = df.groupby("subject_id", group_keys=False).apply(
            simulate_exposure_with_tracking
        )

        # CRITICAL: Validate no exposures occur after DOD
        def validate_exposure_dod_order(subject_data):
            exposure_events = subject_data[subject_data["code"] == self.exposure_name]
            dod_events = subject_data[subject_data["code"] == "DOD"]
            
            if not exposure_events.empty and not dod_events.empty:
                exposure_time = exposure_events["time"].min()
                dod_time = dod_events["time"].min()
                if exposure_time >= dod_time:
                    return True
            return False

        violations = result_df.groupby("subject_id").apply(validate_exposure_dod_order)
        temporal_violations = violations.sum()

        if temporal_violations > 0:
            print(f"⚠️  WARNING: {temporal_violations} subjects have EXPOSURE after DOD!")
            
            # Remove invalid exposures
            def fix_temporal_violations(subject_data):
                if validate_exposure_dod_order(subject_data):
                    # Remove exposure events that occur after DOD
                    dod_time = subject_data[subject_data["code"] == "DOD"]["time"].min()
                    exposure_mask = (subject_data["code"] == self.exposure_name) & (subject_data["time"] >= dod_time)
                    return subject_data[~exposure_mask].reset_index(drop=True)
                return subject_data
            
            result_df = result_df.groupby("subject_id", group_keys=False).apply(fix_temporal_violations)

        # Simulate outcomes if requested
        if simulate_outcome:
            def simulate_outcome_with_tracking(x):
                nonlocal outcome_attempted, outcome_blocked_by_dod
                outcome_attempted += 1

                original_len = len(x)
                result = self.simulate_outcome(
                    x,
                    p_base_outcome,
                    confounder_outcome_effect,
                    outcome_only_effect,
                    exposure_outcome_effect,
                )

                # Check if outcome was added
                if len(result) == original_len:
                    has_dod = (x["code"] == "DOD").any()
                    if has_dod:
                        outcome_blocked_by_dod += 1

                return result

            result_df = result_df.groupby("subject_id", group_keys=False).apply(
                simulate_outcome_with_tracking
            )

        # Print simulation statistics
        print(f"\nSimulation Statistics:")
        print(f"Exposure simulations attempted: {exposure_attempted}")
        print(f"Exposures blocked by DOD: {exposure_blocked_by_dod}")
        print(f"Exposures blocked by invalid triggers: {exposure_invalid_triggers}")
        print(f"Temporal violations detected and fixed: {temporal_violations}")
        if simulate_outcome:
            print(f"Outcome simulations attempted: {outcome_attempted}")
            print(f"Outcomes blocked by DOD: {outcome_blocked_by_dod}")

        # Create ITE dataframe
        ite_df = pd.DataFrame(self.ite_records) if simulate_outcome else pd.DataFrame()

        return result_df, ite_df


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
