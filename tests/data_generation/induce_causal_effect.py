import os

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def simulate_exposure_for_subject(
    subj_df: pd.DataFrame,
    event_code: str,
    trigger_code: str,
    p_base: float,
    a: float,
    days_offset: int,
) -> pd.DataFrame:
    """
    Simulation of event_code as a function of trigger_code.
        event_code is the code of the event to be simulated.
        trigger_code is the code of the event that triggers the simulation.
        p_base is the base probability of the event.
        a is the coefficient for the trigger_code.
        days_offset is the number of days to offset the event.
        random_seed is the random seed for the simulation.
    Inserts the event after the trigger_code event with probability p. If no trigger_code is present, the event is inserted at the median time.
    """
    # Ensure events are sorted by time
    subj_df = subj_df.sort_values("time").reset_index(drop=True)

    # Determine whether LAB0 is present (you can extend this for LAB1, LAB2, etc.)
    trigger_present = (subj_df["code"] == trigger_code).any()

    # Compute the probability using the logistic function.
    # Here, we use:
    #   p = expit(logit(p_base) + a * LAB0_present)
    # where LAB0_present is 1 if present, 0 otherwise.
    p = expit(logit(p_base) + a * int(trigger_present))

    # Simulate the binary outcome based on p
    is_exposed = np.random.rand() < p

    # If the exposure is not simulated, simply return the current subject data.
    if not is_exposed:
        return subj_df

    # Determine the time for the new EXPOSURE event
    if trigger_present:
        # If LAB0 is present, take the latest LAB0 event and add a small offset
        lab0_time = subj_df.loc[subj_df["code"] == trigger_code, "time"].max()
        new_time = lab0_time + pd.Timedelta(days=days_offset)
    else:
        # If no LAB0 is present, insert the EXPOSURE event at the median position.
        # For simplicity, we take the time of the median event.
        median_index = len(subj_df) // 2
        new_time = subj_df.iloc[median_index]["time"]

    # Create a new row for the EXPOSURE event. Note numeric_value is set to NaN.
    exposure_event = pd.DataFrame(
        {
            "subject_id": [subj_df.iloc[0]["subject_id"]],
            "time": [new_time],
            "code": [event_code],
            "numeric_value": [np.nan],
        }
    )

    # Combine the subject's events with the new exposure event and re-sort by time
    combined_df = pd.concat([subj_df, exposure_event], ignore_index=True)
    combined_df = combined_df.sort_values("time").reset_index(drop=True)

    return combined_df


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
    
    parquet_files = [f for f in os.listdir(shard_dir) if f.endswith('.parquet')]
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


def main(
    source_dir: str,
    write_dir: str,
    p_base_exposure: float,
    a_exposure: float,
    p_base_outcome: float,
    a_outcome: float,
    days_offset: int,
    trigger_code: str,
) -> None:
    df, shards = load_data_from_shards(source_dir)
    df["time"] = pd.to_datetime(df["time"])

    # Apply the simulation function per patient
    simulated_df = df.groupby("subject_id", group_keys=False)[df.columns].apply(
        simulate_exposure_for_subject,
        "EXPOSURE",
        trigger_code,
        p_base_exposure,
        a_exposure,
        days_offset,
    )
    simulated_df = simulated_df.groupby("subject_id", group_keys=False)[
        simulated_df.columns
    ].apply(
        simulate_exposure_for_subject,
        "OUTCOME",
        "EXPOSURE",
        p_base_outcome,
        a_outcome,
        days_offset,
    )
    write_shards(simulated_df, write_dir, shards)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulated causal data")
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
        "--p_base_exposure",
        type=float,
        default=0.2,
        help="Base probability for exposure",
    )
    parser.add_argument(
        "--a_exposure", type=float, default=2.0, help="Effect size for exposure"
    )
    parser.add_argument(
        "--p_base_outcome", type=float, default=0.2, help="Base probability for outcome"
    )
    parser.add_argument(
        "--a_outcome", type=float, default=1.0, help="Effect size for outcome"
    )
    parser.add_argument(
        "--days_offset",
        type=int,
        default=100,
        help="Days offset for temporal relationships",
    )
    parser.add_argument("--trigger_code", type=str, default="LAB0", help="Trigger code")

    args = parser.parse_args()

    main(
        source_dir=args.source_dir,
        write_dir=args.write_dir,
        p_base_exposure=args.p_base_exposure,
        a_exposure=args.a_exposure,
        p_base_outcome=args.p_base_outcome,
        a_outcome=args.a_outcome,
        days_offset=args.days_offset,
        trigger_code=args.trigger_code,
    )
