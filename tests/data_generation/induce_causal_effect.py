"""
Enhanced causal effect simulation module for EHR data

This module provides functionality to simulate more complex causal relationships
by inducing exposure and outcome events based on multiple trigger conditions:
1. One code that influences both EXPOSURE and OUTCOME (common cause)
2. One code that only influences EXPOSURE
3. One code that only influences OUTCOME

Key Features:
    - Simulates binary exposure and outcome events with configurable probabilities
    - Supports more complex causal structures with multiple triggers
    - Maintains temporal relationships between events using specified day offsets
    - Preserves original data structure while adding simulated events
    - Handles data sharding for efficient processing of large datasets
    - Supports configurable effect sizes and base probabilities

The simulation uses a logistic model where:
    P(event) = expit(logit(p_base) + a1*trigger1_present + a2*trigger2_present + ...)
    where:
    - p_base is the base probability of the event
    - a1, a2, etc. are the effect size coefficients
    - trigger_present values are binary indicators (0/1)

Example:
    >>> # Simulate with enhanced causal structure
    >>> simulated_df = simulate_causal_effects(
    ...     df,
    ...     common_cause_code="LAB0",  # Affects both exposure and outcome
    ...     exposure_only_code="LAB1",  # Affects only exposure
    ...     outcome_only_code="LAB2",   # Affects only outcome
    ...     p_base_exposure=0.2,
    ...     p_base_outcome=0.2,
    ...     common_cause_exposure_effect=1.5,
    ...     common_cause_outcome_effect=1.0,
    ...     exposure_only_effect=2.0,
    ...     outcome_only_effect=1.5,
    ...     exposure_outcome_effect=2.0,
    ...     days_offset=30
    ... )
"""

import os
import random

import numpy as np
import pandas as pd
from scipy.special import expit, logit

EXPOSURE = "EXPOSURE"
OUTCOME = "OUTCOME"


def simulate_exposure_for_subject(
    subj_df: pd.DataFrame,
    common_cause_code: str,
    exposure_only_code: str,
    p_base: float,
    common_cause_effect: float,
    exposure_only_effect: float,
    days_offset: int,
    exposure_name: str,
) -> pd.DataFrame:
    """
    Simulation of EXPOSURE as a function of common_cause_code and exposure_only_code.

    Args:
        subj_df: DataFrame for a single subject
        common_cause_code: Code that affects both exposure and outcome
        exposure_only_code: Code that only affects exposure
        p_base: Base probability of the exposure event
        common_cause_effect: Effect size coefficient for common cause
        exposure_only_effect: Effect size coefficient for exposure-only trigger
        days_offset: Number of days to offset the exposure event

    Returns:
        DataFrame with simulated exposure event added if applicable
    """
    # Ensure events are sorted by time
    subj_df = subj_df.sort_values("time").reset_index(drop=True)

    # Determine whether trigger codes are present
    common_cause_present = (subj_df["code"] == common_cause_code).any()
    exposure_only_present = (subj_df["code"] == exposure_only_code).any()

    # Compute the probability using the logistic function with both triggers
    # p = expit(logit(p_base) + common_cause_effect * common_cause_present + exposure_only_effect * exposure_only_present)
    logit_p = (
        logit(p_base)
        + common_cause_effect * int(common_cause_present)
        + exposure_only_effect * int(exposure_only_present)
    )
    p = expit(logit_p)

    # Simulate the binary outcome based on p
    is_exposed = np.random.binomial(1, p)

    # If the exposure is not simulated, simply return the current subject data
    if not is_exposed:
        return subj_df

    # Determine the time for the new EXPOSURE event
    if common_cause_present or exposure_only_present:
        # If any trigger is present, take the latest trigger event and add the offset
        trigger_times = []
        if common_cause_present:
            trigger_times.append(
                subj_df.loc[subj_df["code"] == common_cause_code, "time"].max()
            )
        if exposure_only_present:
            trigger_times.append(
                subj_df.loc[subj_df["code"] == exposure_only_code, "time"].max()
            )

        latest_trigger_time = max(trigger_times)
        new_time = latest_trigger_time + pd.Timedelta(days=days_offset)
    else:
        # If no triggers are present, insert the EXPOSURE event at a random position
        if len(subj_df) > 0:
            random_index = random.randint(0, len(subj_df) - 1)
            new_time = subj_df.iloc[random_index]["time"]
        else:
            # If the subject has no events yet, use a default timestamp
            new_time = pd.Timestamp.now()

    # Create a new row for the EXPOSURE event
    exposure_event = pd.DataFrame(
        {
            "subject_id": [
                subj_df.iloc[0]["subject_id"] if len(subj_df) > 0 else "unknown"
            ],
            "time": [new_time],
            "code": [exposure_name],
            "numeric_value": [np.nan],
        }
    )

    # Combine the subject's events with the new exposure event and re-sort by time
    combined_df = pd.concat([subj_df, exposure_event], ignore_index=True)
    combined_df = combined_df.sort_values("time").reset_index(drop=True)

    return combined_df


def simulate_outcome_for_subject(
    subj_df: pd.DataFrame,
    common_cause_code: str,
    outcome_only_code: str,
    p_base: float,
    common_cause_effect: float,
    outcome_only_effect: float,
    exposure_outcome_effect: float,
    days_offset: int,
    outcome_name: str,
) -> pd.DataFrame:
    """
    Simulation of OUTCOME as a function of common_cause_code, outcome_only_code, and EXPOSURE.

    Args:
        subj_df: DataFrame for a single subject
        common_cause_code: Code that affects both exposure and outcome
        outcome_only_code: Code that only affects outcome
        p_base: Base probability of the outcome event
        common_cause_effect: Effect size coefficient for common cause
        outcome_only_effect: Effect size coefficient for outcome-only trigger
        exposure_outcome_effect: Effect size coefficient for exposure's effect on outcome
        days_offset: Number of days to offset the outcome event

    Returns:
        DataFrame with simulated outcome event added if applicable
        Difference in probabilities between treated and untreated status (ATE)
    """
    # Ensure events are sorted by time
    subj_df = subj_df.sort_values("time").reset_index(drop=True)

    # Determine whether trigger codes are present
    common_cause_present = (subj_df["code"] == common_cause_code).any()
    outcome_only_present = (subj_df["code"] == outcome_only_code).any()
    exposure_present = (subj_df["code"] == EXPOSURE).any()

    # Compute the probability using the logistic function with all triggers
    base_logit = (
        logit(p_base)
        + common_cause_effect * int(common_cause_present)
        + outcome_only_effect * int(outcome_only_present)
    )

    p_treated = expit(base_logit + exposure_outcome_effect)
    p_control = expit(base_logit)
    ite = p_treated - p_control

    p_actual = p_treated if exposure_present else p_control
    has_outcome = np.random.binomial(1, p_actual)

    # If the outcome is not simulated, simply return the current subject data
    if not has_outcome:
        return subj_df

    # Determine the time for the new OUTCOME event
    if exposure_present:
        # If EXPOSURE is present, add the outcome after it
        exposure_time = subj_df.loc[subj_df["code"] == EXPOSURE, "time"].max()
        new_time = exposure_time + pd.Timedelta(days=days_offset)
    elif common_cause_present or outcome_only_present:
        # If no EXPOSURE but other triggers are present, take the latest trigger
        trigger_times = []
        if common_cause_present:
            trigger_times.append(
                subj_df.loc[subj_df["code"] == common_cause_code, "time"].max()
            )
        if outcome_only_present:
            trigger_times.append(
                subj_df.loc[subj_df["code"] == outcome_only_code, "time"].max()
            )

        latest_trigger_time = max(trigger_times)
        new_time = latest_trigger_time + pd.Timedelta(days=days_offset)
    else:
        # If no triggers are present, insert the OUTCOME event at the median position
        if len(subj_df) > 0:
            median_index = len(subj_df) // 2
            new_time = subj_df.iloc[median_index]["time"]
        else:
            # If the subject has no events yet, use a default timestamp
            new_time = pd.Timestamp.now()

    # Create a new row for the OUTCOME event
    outcome_event = pd.DataFrame(
        {
            "subject_id": [
                subj_df.iloc[0]["subject_id"] if len(subj_df) > 0 else "unknown"
            ],
            "time": [new_time],
            "code": [outcome_name],
            "numeric_value": [np.nan],
            "ite": [ite],
        }
    )

    # Combine the subject's events with the new outcome event and re-sort by time
    combined_df = pd.concat([subj_df, outcome_event], ignore_index=True)
    combined_df = combined_df.sort_values("time").reset_index(drop=True)

    return combined_df


def simulate_causal_effects(
    df: pd.DataFrame,
    common_cause_code: str,
    exposure_only_code: str,
    outcome_only_code: str,
    p_base_exposure: float,
    p_base_outcome: float,
    common_cause_exposure_effect: float,
    common_cause_outcome_effect: float,
    exposure_only_effect: float,
    outcome_only_effect: float,
    exposure_outcome_effect: float,
    days_offset: int,
    simulate_outcome: bool,
    exposure_name: str,
    outcome_name: str,
) -> pd.DataFrame:
    """
    Apply the enhanced causal simulation to the entire dataset.

    Args:
        df: Input DataFrame with EHR data
        common_cause_code: Code that affects both exposure and outcome
        exposure_only_code: Code that only affects exposure
        outcome_only_code: Code that only affects outcome
        p_base_exposure: Base probability for exposure
        p_base_outcome: Base probability for outcome
        common_cause_exposure_effect: Effect size of common cause on exposure
        common_cause_outcome_effect: Effect size of common cause on outcome
        exposure_only_effect: Effect size of exposure-only trigger
        outcome_only_effect: Effect size of outcome-only trigger
        exposure_outcome_effect: Effect size of exposure on outcome
        days_offset: Days offset for temporal relationships

    Returns:
        DataFrame with simulated exposure and outcome events
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # First, simulate exposure events
    simulated_df = df.groupby("subject_id", group_keys=False)[df.columns].apply(
        simulate_exposure_for_subject,
        common_cause_code,
        exposure_only_code,
        p_base_exposure,
        common_cause_exposure_effect,
        exposure_only_effect,
        days_offset,
        exposure_name,
    )
    if simulate_outcome:
        # Then, simulate outcome events based on exposures and other factors
        simulated_df = simulated_df.groupby("subject_id", group_keys=False)[
            simulated_df.columns
        ].apply(
            simulate_outcome_for_subject,
            common_cause_code,
            outcome_only_code,
            p_base_outcome,
            common_cause_outcome_effect,
            outcome_only_effect,
            exposure_outcome_effect,
            days_offset,
            outcome_name,
        )

    return simulated_df


def load_data_from_shards(shard_dir: str) -> tuple[pd.DataFrame, dict]:
    """Load and concatenate all parquet shards from a directory.

    Args:
        shard_dir: Directory containing parquet shards

    Returns:
        tuple[pd.DataFrame, dict]: Tuple containing concatenated dataframe and
            dictionary mapping shard index to list of subject IDs
    """
    dfs = []
    shards = {}
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    parquet_files = [f for f in os.listdir(shard_dir) if f.endswith(".parquet")]
    if not parquet_files:
        raise ValueError(f"No parquet files found in {shard_dir}")

    for i, path in enumerate(parquet_files):
        file_path = os.path.join(shard_dir, path)
        try:
            shard = pd.read_parquet(file_path)
        except Exception as e:
            raise ValueError(f"Error reading parquet file {file_path}: {e}")
        shards[i] = shard.subject_id.unique()
        dfs.append(shard)
    return pd.concat(dfs), shards


def write_shards(df: pd.DataFrame, write_dir: str, shards: dict) -> None:
    """
    Split dataframe into shards with specified number of patients per shard and write to parquet files.

    Args:
        df: DataFrame to split
        write_dir: Directory to write shards to
        shards: Dictionary of shards
    """
    # Create write directory if it doesn't exist
    os.makedirs(write_dir, exist_ok=True)

    # Create and write shards
    for i, shard_subjects in shards.items():
        # Create shard with selected subjects
        shard = df[df.subject_id.isin(shard_subjects)]
        # Write shard to parquet file
        shard.to_parquet(os.path.join(write_dir, f"{i}.parquet"))


def main() -> None:
    """Main function to run the enhanced causal simulation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate enhanced simulated causal data"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing source data shards",
    )
    parser.add_argument(
        "--write_dir", type=str, required=True, help="Directory to write output shards"
    )
    parser.add_argument(
        "--common_cause_code",
        type=str,
        default="LAB8",
        help="Code that affects both exposure and outcome",
    )
    parser.add_argument(
        "--exposure_only_code",
        type=str,
        default="MIN02",
        help="Code that only affects exposure",
    )
    parser.add_argument(
        "--outcome_only_code",
        type=str,
        default="DE11",
        help="Code that only affects outcome",
    )
    parser.add_argument(
        "--p_base_exposure",
        type=float,
        default=0.3,
        help="Base probability for exposure",
    )
    parser.add_argument(
        "--p_base_outcome", type=float, default=0.2, help="Base probability for outcome"
    )
    parser.add_argument(
        "--common_cause_exposure_effect",
        type=float,
        default=-0.5,
        help="Effect size of common cause on exposure",
    )
    parser.add_argument(
        "--common_cause_outcome_effect",
        type=float,
        default=-0.3,
        help="Effect size of common cause on outcome",
    )
    parser.add_argument(
        "--exposure_only_effect",
        type=float,
        default=1,
        help="Effect size of exposure-only trigger",
    )
    parser.add_argument(
        "--outcome_only_effect",
        type=float,
        default=0.5,
        help="Effect size of outcome-only trigger",
    )
    parser.add_argument(
        "--exposure_outcome_effect",
        type=float,
        default=2.0,
        help="Effect size of exposure on outcome",
    )
    parser.add_argument(
        "--days_offset",
        type=int,
        default=30,
        help="Days offset for temporal relationships",
    )
    parser.add_argument(
        "--simulate_outcome",
        type=bool,
        default=True,
        help="Simulate outcome",
    )
    parser.add_argument(
        "--exposure_name",
        type=str,
        default=EXPOSURE,
        help="Name of exposure",
    )
    parser.add_argument(
        "--outcome_name",
        type=str,
        default=OUTCOME,
        help="Name of outcome",
    )
    args = parser.parse_args()

    # Load data
    df, shards = load_data_from_shards(args.source_dir)

    # Count initial presence of trigger codes (correctly by subject)
    total_subjects = len(df["subject_id"].unique())
    print(f"\nTotal subjects: {total_subjects}")

    # Group by subject_id and check for code presence within each subject
    subject_codes = df.groupby("subject_id")["code"].apply(set).reset_index()

    common_cause_count = (
        subject_codes["code"].apply(lambda codes: args.common_cause_code in codes).sum()
    )
    exposure_trigger_count = (
        subject_codes["code"]
        .apply(lambda codes: args.exposure_only_code in codes)
        .sum()
    )
    if args.simulate_outcome:
        outcome_trigger_count = (
            subject_codes["code"]
            .apply(lambda codes: args.outcome_only_code in codes)
            .sum()
        )

    print("\nTrigger code presence before simulation:")
    print(
        f"  Common cause ({args.common_cause_code}): {common_cause_count} subjects ({100 * common_cause_count / total_subjects:.1f}%)"
    )
    print(
        f"  Exposure trigger ({args.exposure_only_code}): {exposure_trigger_count} subjects ({100 * exposure_trigger_count / total_subjects:.1f}%)"
    )
    if args.simulate_outcome:
        print(
            f"  Outcome trigger ({args.outcome_only_code}): {outcome_trigger_count} subjects ({100 * outcome_trigger_count / total_subjects:.1f}%)"
        )

    # Apply simulation
    simulated_df = simulate_causal_effects(
        df,
        args.common_cause_code,
        args.exposure_only_code,
        args.outcome_only_code,
        args.p_base_exposure,
        args.p_base_outcome,
        args.common_cause_exposure_effect,
        args.common_cause_outcome_effect,
        args.exposure_only_effect,
        args.outcome_only_effect,
        args.exposure_outcome_effect,
        args.days_offset,
        args.simulate_outcome,
        args.exposure_name,
        args.outcome_name,
    )

    # Calculate stats after simulation
    exposure_counts = simulated_df.groupby("subject_id")["code"].apply(
        lambda codes: (codes == args.exposure_name).any()
    )
    if args.simulate_outcome:
        outcome_counts = simulated_df.groupby("subject_id")["code"].apply(
            lambda codes: (codes == args.outcome_name).any()
        )

    exposure_count = exposure_counts.sum()
    if args.simulate_outcome:
        outcome_count = outcome_counts.sum()
        exposure_pct = 100 * exposure_count / total_subjects
        outcome_pct = 100 * outcome_count / total_subjects

    # Calculate conditional probabilities
    if args.simulate_outcome:
        outcomes_given_exposure = outcome_counts[exposure_counts].mean() * 100
        outcomes_given_no_exposure = outcome_counts[~exposure_counts].mean() * 100

    print("\nSimulation results:")
    print(f"  EXPOSURE events: {exposure_count} subjects ({exposure_pct:.1f}%)")

    if args.simulate_outcome:
        print(f"  OUTCOME events: {outcome_count} subjects ({outcome_pct:.1f}%)")
        print(f"  P(Outcome | Exposure): {outcomes_given_exposure:.1f}%")
        print(f"  P(Outcome | No Exposure): {outcomes_given_no_exposure:.1f}%")

    os.makedirs(args.write_dir, exist_ok=True)

    if args.simulate_outcome:
        # Calculate and display ATE
        ate = simulated_df["ite"].mean()
        print(f"\nAverage Treatment Effect: {ate:.4f}")
        print(f"Number of records: {len(simulated_df)}")

        # Remove ite column before writing shards
        simulated_df.drop(columns=["ite"], inplace=True)

        # Write ATE to file
        with open(os.path.join(args.write_dir, ".ate.txt"), "w") as f:
            f.write(f"ATE: {ate}")
    # Write results
    write_shards(simulated_df, args.write_dir, shards)


if __name__ == "__main__":
    main()
