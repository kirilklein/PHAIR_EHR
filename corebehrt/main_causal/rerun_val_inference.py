import argparse
import logging
import os
from os.path import join
from pathlib import Path
from typing import Dict, List

import torch

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
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


def _run_rerun_val_inference(
    finetune_model: str,
    prepared_data: str | None,
    subpopulation_pids: str | None,
    output_model: str | None,
    overwrite: bool,
    logger: logging.Logger,
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

    cfg = _load_finetune_cfg(finetune_model)
    cfg.paths.model = str(output_dir)
    cfg.paths.restart_model = str(finetune_model)
    cfg.paths.prepared_data = _resolve_prepared_data(cfg, prepared_data)
    subpopulation_pids = _resolve_subpopulation_pids(cfg, subpopulation_pids)
    cfg.logging.path = str(output_dir / "logs")

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    vocab = load_vocabulary(cfg.paths.prepared_data)
    data = CausalPatientDataset(loaded_data, vocab)

    if os.path.exists(join(cfg.paths.prepared_data, TEST_PIDS_FILE)):
        test_pids = set(torch.load(join(cfg.paths.prepared_data, TEST_PIDS_FILE)))
        logger.info("Excluding %d held-out test PIDs from fold validation", len(test_pids))
    else:
        test_pids = set()

    train_val_data = data.filter_by_pids(
        [pid for pid in data.get_pids() if pid not in test_pids]
    )

    data_cfg = cfg.get("data", {})
    n_folds = data_cfg.get("n_folds", data_cfg.get("cv_folds", 5))
    seed = data_cfg.get("seed", 42)
    val_ratio = data_cfg.get("val_ratio", 0.2)

    if subpopulation_pids:
        subpop_pids = torch.load(subpopulation_pids)
        logger.info(
            "Filtering to %d subpopulation PIDs and regenerating clean folds",
            len(subpop_pids),
        )
        train_val_data = train_val_data.filter_by_pids(subpop_pids)

    folds = create_folds(
        train_val_data.get_pids(),
        n_folds,
        seed,
        val_ratio=val_ratio,
        bootstrap=False,
    )
    logger.info(
        "Generated %d clean folds (bootstrap=False, seed=%s, val_ratio=%s)",
        len(folds),
        seed,
        val_ratio,
    )

    expected_pids = set(train_val_data.get_pids())
    validate_folds(folds, expected_pids, logger, bootstrap=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    outcome_names = train_val_data.get_outcome_names()
    torch.save(outcome_names, join(output_dir, OUTCOME_NAMES_FILE))
    cfg.save_to_yaml(join(output_dir, FINETUNE_CFG))

    for fold_idx, fold_dict in enumerate(folds, start=1):
        train_data = train_val_data.filter_by_pids(fold_dict[TRAIN_KEY])
        val_data = train_val_data.filter_by_pids(fold_dict[VAL_KEY])
        fold_output_dir = output_dir / f"fold_{fold_idx}"
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_output_dir / "checkpoints", exist_ok=True)

        logger.info("Re-running validation inference for fold %d/%d", fold_idx, len(folds))
        torch.save(train_data.get_pids(), fold_output_dir / "train_pids.pt")
        torch.save(val_data.get_pids(), fold_output_dir / "val_pids.pt")
        _log_exposure_counts(logger, "Train", train_data.get_exposures())
        _log_exposure_counts(logger, "Val", val_data.get_exposures())

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

    PredictionAccumulator(str(output_dir), outcome_names).accumulate_and_save_predictions()
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
    )


if __name__ == "__main__":
    main()
