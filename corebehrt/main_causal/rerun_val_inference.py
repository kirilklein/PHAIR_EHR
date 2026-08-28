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
from corebehrt.modules.setup.causal.manager import CausalModelManager
from corebehrt.modules.setup.causal.prediction_accumulator import PredictionAccumulator
from corebehrt.modules.setup.config import Config, load_config
from corebehrt.modules.trainer.causal.trainer import CausalEHRTrainer


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


def _resolve_subpopulation_pids(cfg: Config, override: str | None) -> str | None:
    if override is not None:
        return override
    return cfg.paths.get("subpopulation_pids", None)


def _resolve_prepared_data(cfg: Config, override: str | None) -> str:
    return override if override is not None else cfg.paths.prepared_data


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
) -> None:
    """Compare prepared label files to the loaded patient cohort."""
    exposure_path = join(prepared_data, BINARY_EXPOSURE_FILE)
    if not os.path.exists(exposure_path):
        logger.warning("No %s in prepared_data — skipping label audit", BINARY_EXPOSURE_FILE)
        return

    binary_exposure = pd.read_csv(exposure_path, index_col=0).squeeze("columns")
    label_pids = set(binary_exposure.index.astype(int))
    prepared_pid_set = set(all_pids)
    overlap = label_pids & prepared_pid_set
    only_in_labels = label_pids - prepared_pid_set
    only_in_prepared = prepared_pid_set - label_pids

    exposed_in_labels = int((binary_exposure == 1).sum())
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


def _log_combined_predictions_audit(
    combined_path: str,
    expected_pids: Set,
    test_pids: Set,
    prepared_data: str,
    logger: logging.Logger,
) -> None:
    """Reconcile combined_predictions.csv against the expected inference cohort."""
    if not os.path.exists(combined_path):
        logger.warning("Combined predictions file not found: %s", combined_path)
        return

    combined = pd.read_csv(combined_path)
    combined_pids = set(combined[PID_COL].astype(int))
    missing = expected_pids - combined_pids
    extra = combined_pids - expected_pids

    logger.info(
        "Combined predictions audit (%s): rows=%d, exposed=%d, expected=%d, "
        "missing=%d, extra=%d",
        combined_path,
        len(combined),
        int((combined[EXPOSURE_COL] == 1).sum()) if EXPOSURE_COL in combined.columns else -1,
        len(expected_pids),
        len(missing),
        len(extra),
    )

    if missing:
        missing_test = missing & test_pids
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
            logger.warning(
                "Compared to %s: %d labeled patients missing from combined_predictions "
                "(%d exposed)",
                BINARY_EXPOSURE_FILE,
                len(missing_from_labels),
                missing_exposed,
            )


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
    finetune_model: str,
    prepared_data: str | None,
    subpopulation_pids: str | None,
    output_model: str | None,
    overwrite: bool,
    logger: logging.Logger,
    include_test_in_val: bool = True,
) -> None:
    finetune_model = Path(finetune_model).resolve()
    output_dir = _build_output_dir(finetune_model, output_model).resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite to reuse it."
        )

    logger.info("=" * 80)
    logger.info("Processing finetune model: %s", finetune_model)
    logger.info("Rerun output directory: %s", output_dir)
    logger.info("include_test_in_val=%s", include_test_in_val)

    cfg = _load_finetune_cfg(finetune_model)
    cfg.paths.model = str(output_dir)
    cfg.paths.restart_model = str(finetune_model)
    cfg.paths.prepared_data = _resolve_prepared_data(cfg, prepared_data)
    subpopulation_pids = _resolve_subpopulation_pids(cfg, subpopulation_pids)
    cfg.logging.path = str(output_dir / "logs")

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(loaded_data, vocab)
    all_pids = data.get_pids()
    logger.info("Prepared patients loaded: %d", len(all_pids))
    _log_prepared_label_audit(cfg.paths.prepared_data, all_pids, logger)

    test_pids = _load_test_pids(cfg.paths.prepared_data, all_pids, logger)
    non_test_pids = [pid for pid in all_pids if pid not in test_pids]
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
        os.makedirs(fold_output_dir / "checkpoints", exist_ok=True)

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
        checkpoint = modelmanager.load_checkpoint(checkpoints=True)
        outcomes: Dict[str, List[int]] = train_data.get_outcomes()
        exposures = train_data.get_exposures()
        model = modelmanager.initialize_finetune_model(checkpoint, outcomes, exposures)

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
    PredictionAccumulator(str(output_dir), outcome_names).accumulate_and_save_predictions()
    combined_path = join(output_dir, COMBINED_PREDICTIONS_FILE)
    _log_combined_predictions_audit(
        combined_path,
        expected_pids,
        test_pids,
        cfg.paths.prepared_data,
        logger,
    )
    compute_and_save_combined_scores_mean_std(
        len(folds), str(output_dir), mode="val", outcome_names=outcome_names
    )
    logger.info("Clean validation inference complete. Output model dir: %s", output_dir)


def main_rerun_val_inference(config_path: str) -> None:
    cfg = load_config(config_path)
    logger = logging.getLogger("rerun_val_inference")
    _run_rerun_val_inference(
        finetune_model=cfg.paths.finetune_model,
        prepared_data=cfg.paths.prepared_data,
        subpopulation_pids=cfg.paths.get("subpopulation_pids", None),
        output_model=cfg.paths.model,
        overwrite=True,
        logger=logger,
        include_test_in_val=bool(cfg.get("include_test_in_val", True)),
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("rerun_val_inference")
    _run_rerun_val_inference(
        finetune_model=args.finetune_model,
        prepared_data=args.prepared_data,
        subpopulation_pids=args.subpopulation_pids,
        output_model=args.output_model,
        overwrite=args.overwrite,
        logger=logger,
        include_test_in_val=args.include_test_in_val,
    )


if __name__ == "__main__":
    main()
