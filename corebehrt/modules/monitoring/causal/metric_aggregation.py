import logging
import os
from datetime import datetime
from os.path import join

import pandas as pd
from corebehrt.constants.causal.data import EXPOSURE, OUTCOME
from corebehrt.azure import is_mlflow_available, log_metric, setup_metrics_dir

logger = logging.getLogger(__name__)

_PROGRESS_EVERY_N_OUTCOMES = 50


def compute_and_save_combined_scores_mean_std(
    n_splits: int,
    finetune_folder: str,
    mode="val",
    outcome_names: list = None,
) -> None:
    """Compute mean and std of test/val scores for all targets and save to single file."""
    n_out = len(outcome_names) if outcome_names else 0
    msg = (
        f"Save combined aggregated scores ({n_splits} folds, "
        f"{n_out} outcomes + exposure; this can take several minutes on remote storage)"
    )
    print(msg, flush=True)
    logger.info(msg)

    all_scores = []

    logger.info("Collecting score CSVs for exposure...")
    print("  [combined scores] exposure...", flush=True)
    exposure_scores = _collect_single_target_scores(
        n_splits, finetune_folder, mode, EXPOSURE
    )
    if exposure_scores is not None:
        exposure_scores[OUTCOME] = EXPOSURE
        all_scores.append(exposure_scores)
        logger.info("Exposure: collected %d score rows", len(exposure_scores))

    if outcome_names:
        for i, outcome_name in enumerate(outcome_names, start=1):
            if i == 1 or i % _PROGRESS_EVERY_N_OUTCOMES == 0 or i == n_out:
                logger.info(
                    "Collecting score CSVs: outcome %d / %d (current=%r)",
                    i,
                    n_out,
                    outcome_name,
                )
                print(f"  [combined scores] outcomes {i} / {n_out}", flush=True)
            outcome_scores = _collect_single_target_scores(
                n_splits, finetune_folder, mode, outcome_name
            )
            if outcome_scores is not None:
                outcome_scores[OUTCOME] = outcome_name
                all_scores.append(outcome_scores)

    if not all_scores:
        w = f"Warning: No score files found for {mode}"
        print(w, flush=True)
        logger.warning(w)
        return

    try:
        logger.info(
            "Concatenating %d score tables (~%d rows total)...",
            len(all_scores),
            sum(len(x) for x in all_scores),
        )
        print("  [combined scores] concatenating and aggregating...", flush=True)
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
        logger.info("Wrote %s (%d rows)", output_path, len(scores_mean_std))
        print(f"  [combined scores] wrote {output_path}", flush=True)

        with setup_metrics_dir(f"{mode} combined scores"):
            if is_mlflow_available():
                from corebehrt.azure.util.log import get_run_and_prefix

                run, prefix = get_run_and_prefix()
                if run is not None:
                    import mlflow

                    batch = {}
                    for _, row in scores_mean_std.iterrows():
                        m, o = row["metric"], row["outcome"]
                        batch[f"{prefix}{m} mean {o}"] = float(row["mean"])
                        batch[f"{prefix}{m} std {o}"] = float(row["std"])
                    logger.info(
                        "Logging %d metrics to MLflow (single batch)...", len(batch)
                    )
                    mlflow.log_metrics(batch, run_id=run.info.run_id)
                else:
                    for _, row in scores_mean_std.iterrows():
                        log_metric(
                            f"{row['metric']} mean {row['outcome']}", row["mean"]
                        )
                        log_metric(f"{row['metric']} std {row['outcome']}", row["std"])
            else:
                for _, row in scores_mean_std.iterrows():
                    log_metric(
                        f"{row['metric']} mean {row['outcome']}", row["mean"]
                    )
                    log_metric(f"{row['metric']} std {row['outcome']}", row["std"])

        logger.info("Finished combined scores for mode=%s", mode)
        print("  [combined scores] done", flush=True)

    except Exception:
        logger.exception("Error processing combined scores for %s", mode)
        print(f"Error processing combined scores for {mode}: (see logs)", flush=True)
        raise


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

        possible_files = [
            f"{mode}_{target_type}_scores_999.csv",
        ]

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

        fold_scores = None
        for filename in possible_files:
            table_path = join(fold_checkpoints_folder, filename)
            if os.path.exists(table_path):
                try:
                    fold_scores = pd.read_csv(table_path)
                    break
                except Exception as e:
                    logger.warning("Error reading %s: %s", table_path, e)
                    continue

        if fold_scores is not None:
            scores.append(fold_scores)

    if not scores:
        logger.debug("No score files for %s_%s", mode, target_type)
        print(
            f"Warning: No score files found for {mode}_{target_type}",
            flush=True,
        )
        return None

    combined_scores = pd.concat(scores, ignore_index=True)

    combined_scores["metric"] = combined_scores["metric"].str.replace(
        f"{target_type}_", "", regex=False
    )

    return combined_scores
