from os.path import join
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from corebehrt.constants.causal.paths import ENCODINGS_FILE
from corebehrt.constants.data import VAL_KEY
from corebehrt.constants.paths import PID_FILE
from corebehrt.functional.io_operations.paths import get_fold_folders
from corebehrt.functional.trainer.collate import dynamic_padding
from corebehrt.modules.model.model import CorebehrtForFineTuning
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

    fold_folders = get_fold_folders(model_dir)

    for fold_folder, fold in tqdm(zip(fold_folders, folds), desc="Encoding folds"):
        fold_dir = join(model_dir, fold_folder)
        logger.info(f"Encoding fold {fold_folder}")
        fold_pids, fold_encodings = encode_fold(
            fold_dir, data, fold, device, loader_cfg
        )
        pids.extend(fold_pids)
        encodings.append(fold_encodings)

    encodings = torch.cat(encodings, dim=0)

    save_encodings(write_dir, encodings, pids)


def encode_fold(
    fold_dir: str,
    data: PatientDataset,
    fold: dict,
    device: torch.device,
    loader_cfg: dict,
):
    """Encodes a single fold."""
    val_data = data.filter_by_pids(fold[VAL_KEY])
    val_dataset = BinaryOutcomeDataset(val_data.patients)
    val_loader = DataLoader(val_dataset, collate_fn=dynamic_padding, **loader_cfg)

    model: CorebehrtForFineTuning = load_model(fold_dir, device)

    encodings, pids = [], val_data.get_pids()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Encoding fold"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(batch)
            encodings.append(model.cls.pool.last_pooled_output)

    return pids, torch.cat(encodings, dim=0)


def load_model(fold_dir: str, device: torch.device) -> CorebehrtForFineTuning:
    """Loads the trained model for a specific fold."""
    model = ModelLoader(fold_dir).load_model(CorebehrtForFineTuning)
    model.eval().to(device)
    return model


def save_encodings(write_dir: str, encodings: torch.Tensor, pids: List[int]):
    """Saves the encodings and PIDs to disk."""
    torch.save(encodings, join(write_dir, ENCODINGS_FILE))
    torch.save(pids, join(write_dir, PID_FILE))
