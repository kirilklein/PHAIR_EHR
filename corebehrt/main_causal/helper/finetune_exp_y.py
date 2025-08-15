"""
Fine-tuning module for causal inference models with exposure-outcome prediction.
This module provides functionality for training causal models across multiple
cross-validation folds, saving model checkpoints, and aggregating predictions.
It handles dataset preparation, model initialization, and training loop execution.
"""

import os
from os.path import join
from typing import Dict, List

import torch

from corebehrt.azure import setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.functional.visualize.model import visualize_weight_distributions
from corebehrt.modules.preparation.causal.dataset import (
    CausalPatientDataset,
    ExposureOutcomesDataset,
)
from corebehrt.modules.setup.causal.manager import CausalModelManager
from corebehrt.modules.trainer.causal.trainer import CausalEHRTrainer


def cv_loop(
    cfg,
    logger,
    finetune_folder: str,
    data: CausalPatientDataset,
    folds: list,
    test_data: CausalPatientDataset,
) -> None:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    for fold, fold_dict in enumerate(folds):
        fold += 1  # 1-indexed
        train_pids = fold_dict[TRAIN_KEY]
        val_pids = fold_dict[VAL_KEY]
        logger.info(f"Training fold {fold}/{len(folds)}")

        train_data = data.filter_by_pids(train_pids)
        val_data = data.filter_by_pids(val_pids)

        with setup_metrics_dir(f"Fold {fold}"):
            finetune_fold(
                cfg, logger, finetune_folder, train_data, val_data, fold, test_data
            )


def finetune_fold(
    cfg,
    logger,
    finetune_folder: str,
    train_data: CausalPatientDataset,
    val_data: CausalPatientDataset,
    fold: int,
    test_data: CausalPatientDataset = None,
) -> None:
    """Finetune model on one fold"""
    if "scheduler" in cfg:
        logger.info("Replacing steps with epochs in scheduler config")
        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.batch_size, len(train_data)
        )

    fold_folder = join(finetune_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Saving pids")
    torch.save(train_data.get_pids(), join(fold_folder, "train_pids.pt"))
    torch.save(val_data.get_pids(), join(fold_folder, "val_pids.pt"))
    if len(test_data) > 0:
        torch.save(test_data.get_pids(), join(fold_folder, "test_pids.pt"))

    logger.info("Initializing datasets")

    train_dataset = ExposureOutcomesDataset(train_data.patients)
    val_dataset = ExposureOutcomesDataset(val_data.patients)

    modelmanager = CausalModelManager(cfg, fold)
    checkpoint = modelmanager.load_checkpoint()
    outcomes: Dict[str, List[int]] = (
        train_data.get_outcomes()
    )  # needed for sampler/ can be made optional
    exposures = train_data.get_exposures()
    model = modelmanager.initialize_finetune_model(checkpoint, outcomes, exposures)

    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(
        model, outcomes
    )
    epoch = modelmanager.get_epoch()
    trainer = CausalEHRTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,  # test only after training
        args=cfg.trainer_args,
        metrics=getattr(cfg, "metrics", {}),
        sampler=sampler,
        scheduler=scheduler,
        cfg=cfg,
        logger=logger,
        accumulate_logits=True,
        run_folder=fold_folder,
        last_epoch=epoch,
    )
    trainer.train()

    logger.info("Load best finetuned model to compute test scores")
    modelmanager_trained = CausalModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_finetune_model(
        checkpoint, outcomes, exposures
    )
    visualize_weight_distributions(model, save_dir=join(fold_folder, "figs"))

    trainer.model = model
    trainer.val_dataset = val_dataset

    logger.info("Evaluating on validation set")
    *_, val_prediction_data = trainer._evaluate(
        mode="val", save_encodings=cfg.get("save_encodings", False)
    )
    if val_prediction_data is not None:
        trainer.process_causal_classification_results(
            val_prediction_data, mode="val", save_results=True
        )

    if test_data and len(test_data) > 0:
        logger.info("Evaluating on test set")
        test_dataset = ExposureOutcomesDataset(test_data.patients)
        trainer.test_dataset = test_dataset
        *_, test_prediction_data = trainer._evaluate(
            mode="test", save_encodings=cfg.get("save_encodings", False)
        )
        if test_prediction_data is not None:
            trainer.process_causal_classification_results(
                test_prediction_data, mode="test", save_results=True
            )
