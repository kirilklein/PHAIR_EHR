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
    outcome_names: list = None,
) -> None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    
    if target_type == "outcome" and outcome_names:
        # Handle multiple outcomes separately
        for outcome_name in outcome_names:
            _compute_and_save_single_target_scores(
                n_splits, finetune_folder, mode, outcome_name
            )
    else:
        # Handle single target (exposure or single outcome)
        _compute_and_save_single_target_scores(
            n_splits, finetune_folder, mode, target_type
        )


def _compute_and_save_single_target_scores(
    n_splits: int,
    finetune_folder: str,
    mode: str,
    target_type: str,
) -> None:
    """Compute mean and std of test/val scores for a single target type."""
    scores = []
    
    for fold in range(1, n_splits + 1):
        fold_checkpoints_folder = join(finetune_folder, f"fold_{fold}", "checkpoints")
        
        if not os.path.exists(fold_checkpoints_folder):
            continue
            
        # Look for files with BEST_MODEL_ID (999) first, then try epoch numbers
        possible_files = [
            f"{mode}_{target_type}_scores_999.csv",  # BEST_MODEL_ID format
        ]
        
        # Also try to find files with actual epoch numbers
        try:
            checkpoint_files = [
                f for f in os.listdir(fold_checkpoints_folder)
                if f.startswith("checkpoint_epoch")
            ]
            if checkpoint_files:
                last_epoch = max(
                    [
                        int(f.split("_")[-2].split("epoch")[-1])
                        for f in checkpoint_files
                    ]
                )
                possible_files.append(f"{mode}_{target_type}_scores_{last_epoch}.csv")
        except (ValueError, IndexError):
            pass
        
        # Try to find any of the possible files
        fold_scores = None
        for filename in possible_files:
            table_path = join(fold_checkpoints_folder, filename)
            if os.path.exists(table_path):
                try:
                    fold_scores = pd.read_csv(table_path)
                    break
                except Exception as e:
                    print(f"Error reading {table_path}: {e}")
                    continue
        
        if fold_scores is not None:
            scores.append(fold_scores)
    
    # Check if we have any scores to process
    if not scores:
        print(f"Warning: No score files found for {mode}_{target_type}")
        return
    
    try:
        scores = pd.concat(scores)
        scores_mean_std = scores.groupby("metric")["value"].agg(["mean", "std"])
        
        date = datetime.now().strftime("%Y%m%d-%H%M")
        output_path = join(finetune_folder, f"{mode}_{target_type}_scores_mean_std_{date}.csv")
        scores_mean_std.to_csv(output_path)
        print(f"Saved aggregated scores to: {output_path}")

        # Log to Azure
        with setup_metrics_dir(f"{mode} {target_type} scores"):
            for idx, row in scores_mean_std.iterrows():
                for col in scores_mean_std.columns:
                    log_metric(f"{idx} {col} {target_type}", row[col])
                    
    except Exception as e:
        print(f"Error processing scores for {mode}_{target_type}: {e}")
