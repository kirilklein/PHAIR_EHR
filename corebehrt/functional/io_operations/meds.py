import os
from typing import Iterator, List


def iterate_splits_and_shards(data_dir: str, splits: List[str]) -> Iterator[str]:
    """
    Iterate through given splits and their respective shards, yielding each shard path.

    Parameters:
        data_dir (str): The directory containing the split subdirectories.
        splits (List[str]): A list of split names (subdirectory names).

    Yields:
        Iterator[str]: Full paths to each shard found in the splits.
    """
    for split in splits:
        split_path = os.path.join(data_dir, split)
        for shard_path in yield_shard_paths(split_path):
            yield shard_path


def yield_shard_paths(split_path: str) -> Iterator[str]:
    """Find all shard paths in the given path and yield them"""
    shards = [
        shard for shard in os.listdir(split_path) if not shard.startswith(".")
    ]  # MEDS on azure makes hidden files
    for shard in shards:
        shard_path = f"{split_path}/{shard}"
        yield shard_path
