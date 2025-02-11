from os.path import join
from typing import List

import torch
from torch.utils.data import DataLoader

from corebehrt.constants.paths import CHECKPOINTS_DIR, ENCODINGS_FILE, PID_FILE
from corebehrt.functional.setup.model import get_last_checkpoint_epoch
from corebehrt.functional.trainer.collate import dynamic_padding
from corebehrt.modules.model.model import BertForFineTuning
from corebehrt.modules.monitoring.logger import get_tqdm
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.modules.setup.loader import ModelLoader


def encode_loop(
    logger,
    model_dir: str,
    write_dir: str,
    data: PatientDataset,
    folds: List[dict],
    loader_cfg: dict,
):
    """Encode patient data using trained models from each fold.

    Args:
        logger: Logger instance
        model_dir: Directory containing model checkpoints
        write_dir: Directory to save encoded data
        data: PatientDataset containing patient data
        folds: List of fold indices for cross-validation
        loader_cfg: Configuration for DataLoader
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodings, pids = [], []

    for i, fold in enumerate(folds):
        logger.info(f"Encoding fold {i+1}")
        fold_pids, fold_encodings = encode_fold(
            model_dir, data, fold, i, device, loader_cfg
        )
        pids.extend(fold_pids)
        encodings.append(fold_encodings)

    encodings = torch.cat(encodings, dim=0)

    save_encodings(write_dir, encodings, pids)


def encode_fold(
    model_dir: str,
    data: PatientDataset,
    fold: dict,
    fold_idx: int,
    device: torch.device,
    loader_cfg: dict,
):
    """Encodes a single fold."""
    val_data = data.filter_by_pids(fold["val"])
    val_dataset = BinaryOutcomeDataset(val_data.patients)
    val_loader = DataLoader(val_dataset, collate_fn=dynamic_padding, **loader_cfg)

    model: BertForFineTuning = load_model(model_dir, fold_idx, device)

    encodings, pids = [], val_data.get_pids()
    with torch.no_grad():
        for batch in get_tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(batch)
            encodings.append(model.cls.pool.last_pooled_output)

    return pids, torch.cat(encodings, dim=0)


def load_model(
    model_dir: str, fold_idx: int, device: torch.device
) -> BertForFineTuning:
    """Loads the trained model for a specific fold."""
    fold_model_dir = join(model_dir, f"fold_{fold_idx+1}")
    epoch = get_last_checkpoint_epoch(join(fold_model_dir, CHECKPOINTS_DIR))

    model = ModelLoader(fold_model_dir, epoch).load_model(BertForFineTuning)
    model.eval().to(device)
    return model


def save_encodings(write_dir: str, encodings: torch.Tensor, pids: List[int]):
    """Saves the encodings and PIDs to disk."""
    torch.save(encodings, join(write_dir, ENCODINGS_FILE))
    torch.save(pids, join(write_dir, PID_FILE))
