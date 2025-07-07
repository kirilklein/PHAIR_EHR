import json
import os
from typing import Dict, List, Tuple

import pandas as pd

from tests.data_generation.helper.config import SimulationConfig


def save_ite_data(
    ite_df: pd.DataFrame,
    config: SimulationConfig,
    split_write_dir: str,
) -> None:
    """Save ITE data for each outcome."""
    # Save ITE data separately if outcomes were simulated
    if config.outcomes and not ite_df.empty:
        ite_df.to_csv(os.path.join(split_write_dir, ".ite.csv"), index=False)

        # Calculate and save ATE for each outcome
        ate_results = {}
        ate_file_content = []

        for outcome_config in config.outcomes.values():
            outcome_code = outcome_config.code
            ite_column = f"ite_{outcome_code}"

            if ite_column in ite_df.columns:
                ate = ite_df[ite_column].mean()
                ate_results[outcome_code] = ate
                ate_file_content.append(f"ATE_{outcome_code}: {ate}")

        # Save ATE results
        with open(os.path.join(split_write_dir, ".ate.txt"), "w") as f:
            f.write("\n".join(ate_file_content))

        # Also save as JSON for programmatic access
        with open(os.path.join(split_write_dir, ".ate.json"), "w") as f:
            json.dump(ate_results, f, indent=2)

        print(f"Saved ITE data for {len(ate_results)} outcomes")
        for outcome_code, ate in ate_results.items():
            print(f"  ATE for {outcome_code}: {ate:.4f}")


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
