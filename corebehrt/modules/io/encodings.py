import os

import pandas as pd


class ShardLoader:
    def __init__(self, shard_dir: str, num_shards: int | None = None):
        self.shard_dir = shard_dir
        self.num_shards = num_shards

    def load(self) -> pd.DataFrame:
        shard_files = [f for f in os.listdir(self.shard_dir) if f.endswith(".parquet")]
        if not shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")
        max_shard = len(shard_files) if self.num_shards is None else self.num_shards
        return pd.concat(
            [pd.read_parquet(os.path.join(self.shard_dir, f)) for i, f in enumerate(shard_files) if i < max_shard]
        )
