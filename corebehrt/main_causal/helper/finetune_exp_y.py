import os
from os.path import join

import torch

from corebehrt.azure import setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.main.helper.finetune_cv import log_best_metrics
from corebehrt.modules.preparation.causal.dataset import (
    CausalPatientDataset,
    ExposureOutcomeDataset,
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

    train_dataset = ExposureOutcomeDataset(train_data.patients)
    val_dataset = ExposureOutcomeDataset(val_data.patients)

    modelmanager = CausalModelManager(cfg, fold)
    checkpoint = modelmanager.load_checkpoint()
    outcomes = train_data.get_outcomes()  # needed for sampler/ can be made optional
    model = modelmanager.initialize_finetune_model(checkpoint, outcomes)

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
    model = modelmanager_trained.initialize_finetune_model(checkpoint, outcomes)

    trainer.model = model
    trainer.val_dataset = val_dataset

    val_loss, val_metrics = trainer._evaluate(epoch, mode="val")
    log_best_metrics(val_loss, val_metrics, "val")
