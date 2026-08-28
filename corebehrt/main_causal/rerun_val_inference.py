import argparse
import logging
import os
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd
import torch

from corebehrt.constants.causal.data import EXPOSURE_COL
from corebehrt.constants.causal.paths import (
    BINARY_EXPOSURE_FILE,
    COMBINED_PREDICTIONS_FILE,
    RERUN_VAL_INFERENCE_CFG,
)
from corebehrt.constants.data import PID_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import (
    FINETUNE_CFG,
    FOLDS_FILE,
    OUTCOME_NAMES_FILE,
    PREPARED_ALL_PATIENTS,
    TEST_PIDS_FILE,
)
from corebehrt.functional.features.split import create_folds
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.main_causal.finetune_exp_y import validate_folds
from corebehrt.modules.monitoring.causal.metric_aggregation import (
    compute_and_save_combined_scores_mean_std,
)
from corebehrt.modules.preparation.causal.dataset import (
    CausalPatientDataset,
    ExposureOutcomesDataset,
)
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.causal.manager import CausalModelManager
from corebehrt.modules.setup.causal.prediction_accumulator import PredictionAccumulator
from corebehrt.modules.setup.config import Config, load_config
from corebehrt.modules.trainer.causal.trainer import CausalEHRTrainer

COHORT_AUDIT_FILE = "cohort_audit.txt"
LOG_NAME = "rerun_val_inference"


def _save_cohort_audit(output_dir: Path, audit: Dict[str, object]) -> str:
    """Persist cohort audit summary alongside model outputs."""
    audit_path = output_dir / COHORT_AUDIT_FILE
    lines = [f"{key}: {value}" for key, value in audit.items()]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(audit_path)


def _prepare_job_cfg(config_path: str) -> Config:
    """Load the Azure/job config and fill in default paths when needed."""
    job_cfg = load_config(config_path)
    finetune_model = Path(job_cfg.paths.finetune_model).resolve()

    if not job_cfg.paths.get("model"):
        job_cfg.paths.model = str(_build_output_dir(finetune_model, None))

    if not job_cfg.paths.get("prepared_data"):
        finetune_cfg = _load_finetune_cfg(finetune_model)
        job_cfg.paths.prepared_data = finetune_cfg.paths.prepared_data

    return job_cfg


def _build_cli_job_cfg(args: argparse.Namespace) -> Config:
    """Build a job config from CLI args for local runs."""
    finetune_model = Path(args.finetune_model).resolve()
    finetune_cfg = _load_finetune_cfg(finetune_model)
    prepared_data = args.prepared_data or finetune_cfg.paths.prepared_data
    paths = {
        "finetune_model": str(finetune_model),
        "prepared_data": prepared_data,
        "model": str(_build_output_dir(finetune_model, args.output_model)),
    }
    if args.subpopulation_pids:
        paths["subpopulation_pids"] = args.subpopulation_pids

    return Config(
        {
            "paths": paths,
            "logging": {"level": logging.INFO},
            "include_test_in_val": args.include_test_in_val,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run clean validation inference for a causal finetune model output, "
            "writing fold predictions that can be consumed by the normal "
            "calibrate and estimate scripts."
        )
    )
    parser.add_argument(
        "--finetune-model",
        required=True,
        help="Model directory produced by finetune_exp_y.py or finetune_subpop.py.",
    )
    parser.add_argument(
        "--prepared-data",
        default=None,
        help="Optional override for prepared_data. Defaults to the path in finetune_config.yaml.",
    )
    parser.add_argument(
        "--subpopulation-pids",
        default=None,
        help="Optional override for subpopulation_pids. If set, clean folds are regenerated on this filtered cohort.",
    )
    parser.add_argument(
        "--output-model",
        default=None,
        help=(
            "Optional output directory for the rerun model-like artifacts. "
            "Defaults to <finetune_model>/rerun_val_pipeline."
        ),
    )
    parser.add_argument(
        "--include-test-in-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If prepared_data contains test_pids.pt, include those patients in "
            "fold creation so each appears in exactly one validation fold. "
            "Use --no-include-test-in-val to exclude them."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing output directory.",
    )
    return parser.parse_args()


def _build_output_dir(finetune_model: Path, output_model: str | None) -> Path:
    if output_model is None:
        return finetune_model / "rerun_val_pipeline"
    return Path(output_model)


def _load_finetune_cfg(finetune_model: Path) -> Config:
    cfg_path = finetune_model / FINETUNE_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing finetune config in model directory: {cfg_path}")
    return load_config(str(cfg_path))


def _log_exposure_counts(logger: logging.Logger, prefix: str, exposures: List[int]) -> None:
    n_exposed = sum(1 for e in exposures if e == 1)
    n_unexposed = sum(1 for e in exposures if e == 0)
    n_other = len(exposures) - n_exposed - n_unexposed
    suffix = f", other/missing={n_other}" if n_other else ""
    logger.info(
        "%s exposure: exposed=%d, unexposed=%d, total=%d%s",
        prefix,
        n_exposed,
        n_unexposed,
        len(exposures),
        suffix,
    )


def _load_test_pids(prepared_data: str, all_pids: Sequence, logger: logging.Logger) -> Set:
    test_path = join(prepared_data, TEST_PIDS_FILE)
    if not os.path.exists(test_path):
        logger.info("No %s found in prepared_data — no held-out test set", TEST_PIDS_FILE)
        return set()

    raw_test_pids = torch.load(test_path)
    test_pids = set(raw_test_pids)
    all_pid_set = set(all_pids)
    overlap = test_pids & all_pid_set
    missing = test_pids - all_pid_set

    logger.info(
        "Found %s: file_count=%d, overlap_with_prepared=%d, not_in_prepared=%d",
        TEST_PIDS_FILE,
        len(test_pids),
        len(overlap),
        len(missing),
    )
    if missing:
        logger.warning(
            "%d test PIDs are not present in prepared patients and will be ignored",
            len(missing),
        )
    return overlap


def _log_prepared_label_audit(
    prepared_data: str, all_pids: Sequence, logger: logging.Logger
) -> Dict[str, int]:
    """Compare prepared label files to the loaded patient cohort."""
    audit: Dict[str, int] = {"prepared_patients": len(all_pids)}
    exposure_path = join(prepared_data, BINARY_EXPOSURE_FILE)
    if not os.path.exists(exposure_path):
        logger.warning("No %s in prepared_data — skipping label audit", BINARY_EXPOSURE_FILE)
        return audit

    binary_exposure = pd.read_csv(exposure_path, index_col=0).squeeze("columns")
    label_pids = set(binary_exposure.index.astype(int))
    prepared_pid_set = set(all_pids)
    overlap = label_pids & prepared_pid_set
    only_in_labels = label_pids - prepared_pid_set
    only_in_prepared = prepared_pid_set - label_pids

    exposed_in_labels = int((binary_exposure == 1).sum())
    audit.update(
        {
            "binary_exposure_rows": len(binary_exposure),
            "binary_exposure_exposed": exposed_in_labels,
            "binary_exposure_unexposed": int((binary_exposure == 0).sum()),
            "label_prepared_overlap": len(overlap),
            "only_in_binary_exposure": len(only_in_labels),
            "only_in_prepared_patients": len(only_in_prepared),
        }
    )
    logger.info(
        "%s audit: rows=%d, exposed=%d, unexposed=%d, "
        "prepared_patients=%d, overlap=%d, only_in_labels=%d, only_in_prepared=%d",
        BINARY_EXPOSURE_FILE,
        len(binary_exposure),
        exposed_in_labels,
        int((binary_exposure == 0).sum()),
        len(prepared_pid_set),
        len(overlap),
        len(only_in_labels),
        len(only_in_prepared),
    )
    return audit


def _log_combined_predictions_audit(
    combined_path: str,
    expected_pids: Set,
    test_pids: Set,
    prepared_data: str,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Reconcile combined_predictions.csv against the expected inference cohort."""
    audit: Dict[str, int] = {
        "expected_inference_patients": len(expected_pids),
        "test_pids_in_prepared": len(test_pids),
    }
    if not os.path.exists(combined_path):
        logger.warning("Combined predictions file not found: %s", combined_path)
        audit["combined_predictions_rows"] = 0
        return audit

    combined = pd.read_csv(combined_path)
    combined_pids = set(combined[PID_COL].astype(int))
    missing = expected_pids - combined_pids
    extra = combined_pids - expected_pids
    exposed_in_combined = (
        int((combined[EXPOSURE_COL] == 1).sum())
        if EXPOSURE_COL in combined.columns
        else -1
    )

    audit.update(
        {
            "combined_predictions_rows": len(combined),
            "combined_predictions_exposed": exposed_in_combined,
            "combined_missing_from_expected": len(missing),
            "combined_extra_vs_expected": len(extra),
        }
    )

    logger.info(
        "Combined predictions audit (%s): rows=%d, exposed=%d, expected=%d, "
        "missing=%d, extra=%d",
        combined_path,
        len(combined),
        exposed_in_combined,
        len(expected_pids),
        len(missing),
        len(extra),
    )

    if missing:
        missing_test = missing & test_pids
        audit["combined_missing_test_pids"] = len(missing_test)
        logger.warning(
            "%d patients missing from combined_predictions; %d overlap with test_pids.pt",
            len(missing),
            len(missing_test),
        )
        if len(missing_test) == len(missing) and test_pids:
            logger.warning(
                "All missing patients are test PIDs — set include_test_in_val=true "
                "(default) or remove prepared_data/test_pids.pt"
            )

    exposure_path = join(prepared_data, BINARY_EXPOSURE_FILE)
    if os.path.exists(exposure_path):
        binary_exposure = pd.read_csv(exposure_path, index_col=0).squeeze("columns")
        label_pids = set(binary_exposure.index.astype(int))
        missing_from_labels = label_pids - combined_pids
        if missing_from_labels:
            missing_exposed = sum(
                1 for pid in missing_from_labels if binary_exposure.loc[pid] == 1
            )
            audit["missing_from_binary_exposure"] = len(missing_from_labels)
            audit["missing_exposed_from_binary_exposure"] = missing_exposed
            logger.warning(
                "Compared to %s: %d labeled patients missing from combined_predictions "
                "(%d exposed)",
                BINARY_EXPOSURE_FILE,
                len(missing_from_labels),
                missing_exposed,
            )

    return audit


def _log_fold_summary(
    folds: List[Dict[str, list]],
    logger: logging.Logger,
    test_pids: Optional[Set] = None,
) -> None:
    test_pids = test_pids or set()
    all_val = []
    for i, fold in enumerate(folds, start=1):
        train_n = len(fold[TRAIN_KEY])
        val_n = len(fold[VAL_KEY])
        n_test_in_val = sum(1 for pid in fold[VAL_KEY] if pid in test_pids)
        n_test_in_train = sum(1 for pid in fold[TRAIN_KEY] if pid in test_pids)
        all_val.extend(fold[VAL_KEY])
        logger.info(
            "Fold %d: train=%d (test=%d), val=%d (test=%d)",
            i,
            train_n,
            n_test_in_train,
            val_n,
            n_test_in_val,
        )
    logger.info(
        "Validation coverage across folds: unique_val_pids=%d, total_val_slots=%d",
        len(set(all_val)),
        len(all_val),
    )


def _run_rerun_val_inference(
    job_cfg: Config,
    overwrite: bool,
    logger: logging.Logger,
) -> Dict[str, object]:
    finetune_model = Path(job_cfg.paths.finetune_model).resolve()
    output_dir = Path(job_cfg.paths.model).resolve()
    include_test_in_val = bool(job_cfg.get("include_test_in_val", True))
    prepared_data = job_cfg.paths.prepared_data
    subpopulation_pids = job_cfg.paths.get("subpopulation_pids", None)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite to reuse it."
        )

    audit: Dict[str, object] = {
        "finetune_model": str(finetune_model),
        "output_dir": str(output_dir),
        "include_test_in_val": include_test_in_val,
        "prepared_data": prepared_data,
    }

    logger.info("=" * 80)
    logger.info("Processing finetune model: %s", finetune_model)
    logger.info("Rerun output directory: %s", output_dir)
    logger.info("Output directory exists: %s", output_dir.exists())
    if output_dir.exists():
        logger.info("Output directory contents: %s", os.listdir(output_dir))
    logger.info("include_test_in_val=%s", include_test_in_val)

    cfg = _load_finetune_cfg(finetune_model)
    cfg.paths.model = str(output_dir)
    cfg.paths.restart_model = str(finetune_model)
    cfg.paths.prepared_data = prepared_data
    if subpopulation_pids:
        cfg.paths.subpopulation_pids = subpopulation_pids

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(loaded_data, vocab)
    all_pids = data.get_pids()
    logger.info("Prepared patients loaded: %d", len(all_pids))
    audit.update(_log_prepared_label_audit(cfg.paths.prepared_data, all_pids, logger))

    test_pids = _load_test_pids(cfg.paths.prepared_data, all_pids, logger)
    non_test_pids = [pid for pid in all_pids if pid not in test_pids]
    audit["test_pids_count"] = len(test_pids)
    audit["non_test_pids_count"] = len(non_test_pids)
    logger.info(
        "Cohort split: prepared=%d, test_in_prepared=%d, non_test=%d",
        len(all_pids),
        len(test_pids),
        len(non_test_pids),
    )

    if test_pids and include_test_in_val:
        logger.info(
            "include_test_in_val=true: including %d test PIDs in fold creation "
            "(each will appear in exactly one validation fold)",
            len(test_pids),
        )
        train_val_data = data
    elif test_pids:
        logger.info(
            "Excluding %d held-out test PIDs from fold validation "
            "(set include_test_in_val=true / --include-test-in-val to keep them)",
            len(test_pids),
        )
        train_val_data = data.filter_by_pids(non_test_pids)
    else:
        train_val_data = data

    logger.info("Patients after test handling: %d", len(train_val_data.get_pids()))
    audit["patients_after_test_handling"] = len(train_val_data.get_pids())

    data_cfg = cfg.get("data", {})
    n_folds = data_cfg.get("n_folds", data_cfg.get("cv_folds", 5))
    seed = data_cfg.get("seed", 42)
    val_ratio = data_cfg.get("val_ratio", 0.2)

    if subpopulation_pids:
        subpop_pids = torch.load(subpopulation_pids)
        before = len(train_val_data.get_pids())
        logger.info("Filtering to subpopulation: file_count=%d", len(subpop_pids))
        train_val_data = train_val_data.filter_by_pids(subpop_pids)
        after = len(train_val_data.get_pids())
        logger.info("Patients after subpopulation filter: %d → %d", before, after)
        if test_pids:
            test_pids = set(train_val_data.get_pids()) & test_pids
            logger.info(
                "Test PIDs remaining after subpopulation filter: %d", len(test_pids)
            )

    fold_pids = train_val_data.get_pids()
    folds = create_folds(
        fold_pids,
        n_folds,
        seed,
        val_ratio=val_ratio,
        bootstrap=False,
    )
    logger.info(
        "Generated %d clean folds from %d PIDs "
        "(bootstrap=False, seed=%s, val_ratio=%s, include_test_in_val=%s)",
        len(folds),
        len(fold_pids),
        seed,
        val_ratio,
        include_test_in_val,
    )
    _log_fold_summary(folds, logger, test_pids=test_pids)

    expected_pids = set(fold_pids)
    audit["patients_for_inference"] = len(expected_pids)
    validate_folds(folds, expected_pids, logger, bootstrap=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    outcome_names = train_val_data.get_outcome_names()
    torch.save(outcome_names, join(output_dir, OUTCOME_NAMES_FILE))
    torch.save(folds, join(output_dir, FOLDS_FILE))
    cfg.save_to_yaml(join(output_dir, FINETUNE_CFG))
    logger.info(
        "Saved folds.pt / outcome_names / finetune_config; "
        "n_outcomes=%d, n_patients_for_inference=%d",
        len(outcome_names),
        len(expected_pids),
    )

    total_val_evaluated = 0
    for fold_idx, fold_dict in enumerate(folds, start=1):
        train_data = train_val_data.filter_by_pids(fold_dict[TRAIN_KEY])
        val_data = train_val_data.filter_by_pids(fold_dict[VAL_KEY])
        fold_output_dir = output_dir / f"fold_{fold_idx}"
        os.makedirs(fold_output_dir, exist_ok=True)

        logger.info(
            "Re-running validation inference for fold %d/%d "
            "(train_pids=%d, val_pids=%d)",
            fold_idx,
            len(folds),
            len(train_data),
            len(val_data),
        )
        torch.save(train_data.get_pids(), fold_output_dir / "train_pids.pt")
        torch.save(val_data.get_pids(), fold_output_dir / "val_pids.pt")
        _log_exposure_counts(logger, "Train", train_data.get_exposures())
        _log_exposure_counts(logger, "Val", val_data.get_exposures())
        total_val_evaluated += len(val_data)

        train_dataset = ExposureOutcomesDataset(train_data.patients)
        val_dataset = ExposureOutcomesDataset(val_data.patients)

        modelmanager = CausalModelManager(cfg, fold_idx)
        # Always load fold weights from the source finetune model (restart_model),
        # not from the rerun output dir which may contain placeholder checkpoints.
        checkpoint = modelmanager.load_checkpoint(checkpoints=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint loaded from "
                f"{modelmanager.checkpoint_model_path} is missing 'model_state_dict'. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        outcomes: Dict[str, List[int]] = train_data.get_outcomes()
        exposures = train_data.get_exposures()
        model = modelmanager.initialize_finetune_model(checkpoint, outcomes, exposures)

        os.makedirs(fold_output_dir / "checkpoints", exist_ok=True)
        # Inference only: skip optimizer/scheduler setup (would require
        # replace_steps_with_epochs for epoch-based scheduler configs).
        trainer_args = dict(cfg.get("trainer_args", {}))
        trainer_args["use_pcgrad"] = False
        trainer_args["freeze_encoder_at_init"] = False

        trainer = CausalEHRTrainer(
            model=model,
            optimizer=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=None,
            args=trainer_args,
            metrics=getattr(cfg, "metrics", {}),
            sampler=None,
            scheduler=None,
            cfg=cfg,
            logger=logger,
            accumulate_logits=True,
            run_folder=str(fold_output_dir),
            last_epoch=None,
        )

        logger.info("Evaluating on validation set")
        *_, val_prediction_data = trainer.evaluate(mode="val")
        if val_prediction_data is None:
            raise ValueError(f"No validation predictions produced for fold {fold_idx}")
        trainer.process_causal_classification_results(
            val_prediction_data, mode="val", save_results=True
        )
        # PredictionAccumulator finds the epoch via checkpoint_epoch*_end.pt.
        # Eval saves predictions as *_999.npz (BEST_MODEL_ID); write a matching
        # placeholder so accumulate_and_save_predictions can locate them.
        torch.save(
            {"epoch": 999},
            fold_output_dir / "checkpoints" / "checkpoint_epoch999_end.pt",
        )

    logger.info(
        "Finished fold inference: sum_of_val_sizes=%d, unique_patients=%d",
        total_val_evaluated,
        len(expected_pids),
    )
    audit["sum_of_val_sizes"] = total_val_evaluated
    PredictionAccumulator(str(output_dir), outcome_names).accumulate_and_save_predictions()
    combined_path = join(output_dir, COMBINED_PREDICTIONS_FILE)
    audit.update(
        _log_combined_predictions_audit(
            combined_path,
            expected_pids,
            test_pids,
            cfg.paths.prepared_data,
            logger,
        )
    )
    audit_path = _save_cohort_audit(output_dir, audit)
    logger.info("Cohort audit saved to: %s", audit_path)
    job_cfg.save_to_yaml(join(output_dir, RERUN_VAL_INFERENCE_CFG))
    compute_and_save_combined_scores_mean_std(
        len(folds), str(output_dir), mode="val", outcome_names=outcome_names
    )
    logger.info("Clean validation inference complete. Output model dir: %s", output_dir)
    return audit


def main_rerun_val_inference(config_path: str) -> None:
    job_cfg = _prepare_job_cfg(config_path)
    job_cfg.logging.path = join(job_cfg.paths.model, "logs")

    preparer = CausalDirectoryPreparer(job_cfg)
    preparer.setup_rerun_val_inference()

    logger = logging.getLogger(LOG_NAME)
    logger.info("Output directory: %s", job_cfg.paths.model)
    logger.info("Log file: %s/rerun_val_inference.log", job_cfg.logging.path)

    _run_rerun_val_inference(
        job_cfg=job_cfg,
        overwrite=True,
        logger=logger,
    )


def main() -> None:
    args = _parse_args()
    job_cfg = _build_cli_job_cfg(args)
    job_cfg.logging.path = join(job_cfg.paths.model, "logs")

    CausalDirectoryPreparer(job_cfg).setup_rerun_val_inference()

    logger = logging.getLogger(LOG_NAME)
    logger.info("Output directory: %s", job_cfg.paths.model)
    logger.info("Log file: %s/rerun_val_inference.log", job_cfg.logging.path)
    _run_rerun_val_inference(
        job_cfg=job_cfg,
        overwrite=args.overwrite,
        logger=logger,
    )


if __name__ == "__main__":
    main()
