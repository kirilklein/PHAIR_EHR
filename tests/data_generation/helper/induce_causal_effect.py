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
    between multiple medical codes. It supports:
    - Confounder codes: Affect both exposure and outcome probabilities
    - Instrument codes: Only affect exposure probability (like instrumental variables)
    - Prognostic codes: Only affect outcome probability (prognostic factors)
    """

    def __init__(
        self,
        confounder_codes: List[str],
        instrument_codes: List[str],
        prognostic_codes: List[str],
        exposure_name: str = "EXPOSURE",
        outcome_name: str = "OUTCOME",
    ):
        """
        Initialize the causal simulator with multiple trigger codes.

        Args:
            confounder_codes: List of codes that affect both exposure and outcome
            instrument_codes: List of codes that only affect exposure
            prognostic_codes: List of codes that only affect outcome
            exposure_name: Name for simulated exposure events
            outcome_name: Name for simulated outcome events
        """
        self.confounder_codes = confounder_codes
        self.instrument_codes = instrument_codes
        self.prognostic_codes = prognostic_codes
        self.exposure_name = exposure_name
        self.outcome_name = outcome_name
        self.ite_records = []

    def _compute_probability(
        self, base_prob: float, effects: List[Tuple[float, bool]]
    ) -> float:
        """
        Compute event probability using logistic function.

        Args:
            base_prob: Base probability without any triggers
            effects: List of (effect_size, is_present) tuples

        Returns:
            Probability after applying all effects
        """
        logit_p = logit(base_prob) + sum(
            effect * int(present) for effect, present in effects
        )
        return expit(logit_p)

    def _check_codes_present(
        self, subj_df: pd.DataFrame, codes: List[str]
    ) -> List[bool]:
        """
        Check which codes from a list are present in subject data.

        Args:
            subj_df: Subject dataframe
            codes: List of codes to check

        Returns:
            List of booleans indicating presence of each code
        """
        return [(subj_df["code"] == code).any() for code in codes]

    def _get_trigger_codes_for_timing(
        self, subj_df: pd.DataFrame, code_lists: List[List[str]]
    ) -> List[str]:
        """
        Get all trigger codes that are present in the subject data.

        Args:
            subj_df: Subject dataframe
            code_lists: List of code lists to check

        Returns:
            Flattened list of all present trigger codes
        """
        present_codes = []
        for code_list in code_lists:
            for code in code_list:
                if (subj_df["code"] == code).any():
                    present_codes.append(code)
        return present_codes

    def _get_event_time(
        self,
        subj_df: pd.DataFrame,
        trigger_codes: List[str],
        days_offset: int,
        fallback_strategy: str = "random",
        outcome: bool = False,
    ) -> pd.Timestamp:
        """
        Determine timing for new event based on triggers, ensuring it's before DOD.

        Args:
            subj_df: Subject dataframe
            trigger_codes: List of codes that can trigger the event
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
                min_buffer_hours = (
                    1 if outcome else 24
                )  # Outcomes can be closer to death
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
        """
        Add new event to subject DataFrame.

        Args:
            subj_df: Subject dataframe
            event_code: Code for the new event
            event_time: Timestamp for the new event
            **extra_cols: Additional columns to add

        Returns:
            Updated dataframe with new event
        """
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
        confounder_weights: List[float],
        instrument_weights: List[float],
    ) -> pd.DataFrame:
        """
        Simulate exposure event for a single subject based on multiple trigger codes.

        Args:
            subj_df: Subject dataframe
            p_base: Base probability for exposure
            confounder_weights: List of weights for confounder codes (exposure effects)
            instrument_weights: List of weights for instrument codes

        Returns:
            Updated dataframe with potential exposure event
        """
        subj_df = subj_df.sort_values("time").reset_index(drop=True)

        # Check which codes are present
        confounder_present = self._check_codes_present(subj_df, self.confounder_codes)
        instrument_present = self._check_codes_present(subj_df, self.instrument_codes)

        # Compute effects from all present codes
        effects = []

        # Add confounder effects (exposure effects - first half of weights)
        for i, (present, weight) in enumerate(
            zip(confounder_present, confounder_weights[::2])
        ):
            if i < len(confounder_weights) // 2:  # Take every other weight for exposure
                effects.append((weight, present))

        # Add instrument effects
        for present, weight in zip(instrument_present, instrument_weights):
            effects.append((weight, present))

        # Compute probability and simulate
        prob = self._compute_probability(p_base, effects)

        if not np.random.binomial(1, prob):
            return subj_df

        # Determine trigger codes for timing
        trigger_codes = self._get_trigger_codes_for_timing(
            subj_df, [self.confounder_codes, self.instrument_codes]
        )

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

        # Final check: Ensure proposed exposure time is before DOD
        dod_events = subj_df[subj_df["code"] == "DOD"]
        if not dod_events.empty:
            dod_time = dod_events["time"].min()
            if event_time >= dod_time:
                return subj_df

        return self._add_event(subj_df, self.exposure_name, event_time)

    def simulate_outcome(
        self,
        subj_df: pd.DataFrame,
        p_base: float,
        confounder_weights: List[float],
        prognostic_weights: List[float],
        exposure_outcome_effect: float,
    ) -> pd.DataFrame:
        """
        Simulate outcome event for a single subject based on multiple trigger codes.

        Args:
            subj_df: Subject dataframe
            p_base: Base probability for outcome
            confounder_weights: List of weights for confounder codes (outcome effects)
            prognostic_weights: List of weights for prognostic codes
            exposure_outcome_effect: Effect of exposure on outcome

        Returns:
            Updated dataframe with potential outcome event
        """
        subj_df = subj_df.sort_values("time").reset_index(drop=True)

        # Check which codes are present
        confounder_present = self._check_codes_present(subj_df, self.confounder_codes)
        prognostic_present = self._check_codes_present(subj_df, self.prognostic_codes)
        exposure_present = (subj_df["code"] == self.exposure_name).any()

        # Compute base effects (without exposure)
        base_effects = []

        # Add confounder effects (outcome effects - second half of weights)
        for i, (present, weight) in enumerate(
            zip(confounder_present, confounder_weights[1::2])
        ):
            if i < len(confounder_weights) // 2:  # Take every other weight for outcome
                base_effects.append((weight, present))

        # Add prognostic effects
        for present, weight in zip(prognostic_present, prognostic_weights):
            base_effects.append((weight, present))

        # Compute probabilities for treated/control
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
        trigger_codes = self._get_trigger_codes_for_timing(
            subj_df, [self.confounder_codes, self.prognostic_codes]
        )
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
        confounder_weights: List[float],
        instrument_weights: List[float],
        prognostic_weights: List[float],
        exposure_outcome_effect: float,
        simulate_outcome: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply causal simulation to entire dataset with multiple trigger codes.

        Args:
            df: Input dataframe
            p_base_exposure: Base probability for exposure
            p_base_outcome: Base probability for outcome
            confounder_weights: Alternating exposure/outcome weights for confounder codes
            instrument_weights: Weights for instrument codes (exposure only)
            prognostic_weights: Weights for prognostic codes (outcome only)
            exposure_outcome_effect: Main causal effect of exposure on outcome
            simulate_outcome: Whether to simulate outcome events

        Returns:
            Tuple of (simulated_dataframe, ite_dataframe)
        """
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
            nonlocal \
                exposure_attempted, \
                exposure_blocked_by_dod, \
                exposure_invalid_triggers
            exposure_attempted += 1

            original_len = len(x)
            result = self.simulate_exposure(
                x,
                p_base_exposure,
                confounder_weights,
                instrument_weights,
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

        # Validate no exposures occur after DOD
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
            print(
                f"⚠️  WARNING: {temporal_violations} subjects have EXPOSURE after DOD!"
            )

            # Remove invalid exposures
            def fix_temporal_violations(subject_data):
                if validate_exposure_dod_order(subject_data):
                    # Remove exposure events that occur after DOD
                    dod_time = subject_data[subject_data["code"] == "DOD"]["time"].min()
                    exposure_mask = (subject_data["code"] == self.exposure_name) & (
                        subject_data["time"] >= dod_time
                    )
                    return subject_data[~exposure_mask].reset_index(drop=True)
                return subject_data

            result_df = result_df.groupby("subject_id", group_keys=False).apply(
                fix_temporal_violations
            )

        # Simulate outcomes if requested
        if simulate_outcome:

            def simulate_outcome_with_tracking(x):
                nonlocal outcome_attempted, outcome_blocked_by_dod
                outcome_attempted += 1

                original_len = len(x)
                result = self.simulate_outcome(
                    x,
                    p_base_outcome,
                    confounder_weights,
                    prognostic_weights,
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
    """Handles data loading and saving operations for simulation."""

    @staticmethod
    def load_shards(shard_dir: str) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """
        Load and concatenate parquet shards from directory.

        Args:
            shard_dir: Directory containing .parquet files

        Returns:
            Tuple of (concatenated_dataframe, shard_mapping)
        """
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
        """
        Write DataFrame as sharded parquet files.

        Args:
            df: DataFrame to write
            write_dir: Output directory
            shards: Dictionary mapping shard_id to list of subject_ids
        """
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
        """
        Print statistics about trigger code presence before simulation.

        Args:
            df: Input dataframe
            simulator: CausalSimulator instance
            simulate_outcome: Whether outcome simulation is enabled
        """
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
        """
        Print simulation results statistics.

        Args:
            df: Simulated dataframe
            simulator: CausalSimulator instance
            simulate_outcome: Whether outcome simulation was enabled
        """
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
