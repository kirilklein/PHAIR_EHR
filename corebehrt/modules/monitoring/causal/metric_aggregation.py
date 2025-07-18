import os
from datetime import datetime
from os.path import join

import pandas as pd
from corebehrt.constants.causal.data import EXPOSURE, OUTCOME
from corebehrt.azure import log_metric, setup_metrics_dir


def compute_and_save_combined_scores_mean_std(
    n_splits: int,
    finetune_folder: str,
    mode="val",
    outcome_names: list = None,
) -> None:
    """Compute mean and std of test/val scores for all targets and save to single file."""
    print("Save combined aggregated scores")

    all_scores = []

    # Collect exposure scores
    exposure_scores = _collect_single_target_scores(
        n_splits, finetune_folder, mode, EXPOSURE
    )
    if exposure_scores is not None:
        exposure_scores[OUTCOME] = EXPOSURE
        all_scores.append(exposure_scores)

    # Collect outcome scores
    if outcome_names:
        for outcome_name in outcome_names:
            outcome_scores = _collect_single_target_scores(
                n_splits, finetune_folder, mode, outcome_name
            )
            if outcome_scores is not None:
                outcome_scores[OUTCOME] = outcome_name
                all_scores.append(outcome_scores)

    # Combine all scores
    if not all_scores:
        print(f"Warning: No score files found for {mode}")
        return

    try:
        combined_scores = pd.concat(all_scores, ignore_index=True)
        scores_mean_std = (
            combined_scores.groupby(["metric", "outcome"])["value"]
            .agg(["mean", "std"])
            .reset_index()
        )

        date = datetime.now().strftime("%Y%m%d-%H%M")
        scores_dir = join(finetune_folder, "scores")
        os.makedirs(scores_dir, exist_ok=True)
        output_path = join(scores_dir, f"scores_{date}.csv")
        scores_mean_std.to_csv(output_path, index=False)

        # Log to Azure
        with setup_metrics_dir(f"{mode} combined scores"):
            for _, row in scores_mean_std.iterrows():
                metric_name = row["metric"]
                outcome_name = row["outcome"]
                log_metric(f"{metric_name} mean {outcome_name}", row["mean"])
                log_metric(f"{metric_name} std {outcome_name}", row["std"])

    except Exception as e:
        print(f"Error processing combined scores for {mode}: {e}")


def _collect_single_target_scores(
    n_splits: int,
    finetune_folder: str,
    mode: str,
    target_type: str,
) -> pd.DataFrame:
    """Collect scores for a single target type and return as DataFrame."""
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
                f
                for f in os.listdir(fold_checkpoints_folder)
                if f.startswith("checkpoint_epoch")
            ]
            if checkpoint_files:
                last_epoch = max(
                    [int(f.split("_")[-2].split("epoch")[-1]) for f in checkpoint_files]
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

    # Return concatenated scores or None if no scores found
    if not scores:
        print(f"Warning: No score files found for {mode}_{target_type}")
        return None

    combined_scores = pd.concat(scores, ignore_index=True)

    # Clean metric names by removing target_type prefix
    combined_scores["metric"] = combined_scores["metric"].str.replace(
        f"{target_type}_", "", regex=False
    )

    return combined_scores
