"""
Input: Formatted Data
- Load concepts
- Handle wrong data
- Exclude patients with <k concepts
- Split data
- Tokenize
- truncate train and val
"""

import logging
from os.path import join
import pandas as pd


from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.features.loader import ShardLoader
from tqdm import tqdm
from corebehrt.constants.data import PID_COL, CONCEPT_COL

logger = logging.getLogger("get_pat_counts_by_code")
CONFIG_PATH = "./corebehrt/configs/causal/get_pat_counts_by_code.yaml"


def main(config_path):
    """
    Loads configuration, prepares directories, and calculates patient counts by code.
    """
    cfg = load_config(config_path)

    dir_preparer = CausalDirectoryPreparer(cfg)
    dir_preparer.setup_get_pat_counts_by_code()

    shard_loader = ShardLoader(
        data_dir=cfg.paths.data,
        splits=cfg.paths.splits,
    )

    get_pat_counts_by_code(
        shard_loader=shard_loader,
        counts_dir=cfg.paths.counts,
    )


def get_pat_counts_by_code(shard_loader: ShardLoader, counts_dir: str):
    """
    Calculates patient counts per code across all data shards.

    Iterates through each data shard, calculates patient counts for each code,
    aggregates these counts, and saves the final counts to a CSV file.
    """
    all_counts = pd.Series(dtype="int64")

    for shard, _ in tqdm(shard_loader(), desc="loop shards"):
        code_counts = shard.groupby(CONCEPT_COL)[PID_COL].nunique()
        all_counts = all_counts.add(code_counts, fill_value=0)

    all_counts = all_counts.sort_values(ascending=False).astype(int)

    output_path = join(counts_dir, "pat_counts_by_code.csv")
    all_counts.to_csv(output_path, header=["patient_count"])
    logger.info(f"Saved patient counts to {output_path}")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
