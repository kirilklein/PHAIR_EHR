import os
from os.path import join

from corebehrt.constants.paths import CHECKPOINTS_DIR


def get_fold_folders(finetune_folder: str) -> list[str]:
    """Get the fold folders inside the finetune folder."""
    return [f for f in os.listdir(finetune_folder) if f.startswith("fold_")]


def get_checkpoint_predictions_path(
    file_type: str, fold_dir: str, mode: str, epoch: int
) -> str:
    """Get the path to the file for a given fold and mode."""
    return join(fold_dir, CHECKPOINTS_DIR, f"{file_type}_{mode}_{epoch}.npz")
