import os
from datetime import datetime
from os.path import join
from typing import Literal

import pandas as pd

from corebehrt.azure import log_metric, setup_metrics_dir


def compute_and_save_scores_mean_std(
    n_splits: int,
    finetune_folder: str,
    mode="val",
    target_type: Literal["exposure", "outcome"] = "exposure",
) -> None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    scores = []
    for fold in range(1, n_splits + 1):
        fold_checkpoints_folder = join(finetune_folder, f"fold_{fold}", "checkpoints")
        last_epoch = max(
            [
                int(f.split("_")[-2].split("epoch")[-1])
                for f in os.listdir(fold_checkpoints_folder)
                if f.startswith("checkpoint_epoch")
            ]
        )
        table_path = join(
            fold_checkpoints_folder, f"{mode}_{target_type}_scores_{last_epoch}.csv"
        )
        if not os.path.exists(table_path):
            continue
        fold_scores = pd.read_csv(table_path)
        scores.append(fold_scores)
    scores = pd.concat(scores)
    scores_mean_std = scores.groupby("metric")["value"].agg(["mean", "std"])
    date = datetime.now().strftime("%Y%m%d-%H%M")
    scores_mean_std.to_csv(
        join(finetune_folder, f"{mode}_{target_type}_scores_mean_std_{date}")
    )

    # Log to Azure
    with setup_metrics_dir(f"{mode} {target_type} scores"):
        for idx, row in scores_mean_std.iterrows():
            for col in scores_mean_std.columns:
                log_metric(f"{idx} {col} {target_type}", row[col])
