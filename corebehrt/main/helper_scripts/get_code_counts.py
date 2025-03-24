"""
This script is used to get the counts of codes in the meds data.
As input we use the meds directory, using train/tuning sets.
"""

import json
import logging
import os
from collections import Counter
from os.path import join
from typing import Iterator, List

import pandas as pd

from corebehrt.constants.data import CONCEPT_COL
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/helper/get_counts.yaml"
OUTPUT_FILE_NAME = "code_counts.json"

logger = logging.getLogger("get_code_counts")


def main(config_path):
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_logging("get_code_counts")
    counts_dict = get_and_save_code_counts(cfg.paths.data, cfg.splits, cfg.paths.counts)
    # Now you can use code_counts for additional processing if needed
    logger.info(f"Total unique codes found: {len(counts_dict)}")
    return counts_dict


def get_and_save_code_counts(data_dir: str, splits: List[str], write_dir: str) -> dict:
    """
    Get the counts of codes in the dataset.
    """
    all_code_counts = Counter()

    os.makedirs(write_dir, exist_ok=True)

    for split_name in splits:
        logger.info(f"Getting code counts for {split_name} split")
        path_name = f"{data_dir}/{split_name}"

        for shard_path in yield_shard_paths(path_name):
            concepts = pd.read_parquet(shard_path, columns=[CONCEPT_COL])
            shard_counts = concepts[CONCEPT_COL].value_counts().to_dict()
            all_code_counts.update(shard_counts)

    counts_dict = dict(all_code_counts)

    with open(join(write_dir, OUTPUT_FILE_NAME), "w") as f:
        json.dump(counts_dict, f)

    logger.info(f"Saved counts for {len(counts_dict)} unique codes to {write_dir}")

    return counts_dict


def yield_shard_paths(path_name: str) -> Iterator[str]:
    """Find all shard paths in the given path and yield them"""
    shards = [
        shard for shard in os.listdir(path_name) if not shard.startswith(".")
    ]  # MEDS on azure makes hidden files
    for shard in shards:
        shard_path = f"{path_name}/{shard}"
        yield shard_path


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(CONFIG_PATH)
