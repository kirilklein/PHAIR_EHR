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
        write_dir: Directory to save encoded data
        data: PatientDataset containing patient data
        folds: List of fold indices for cross-validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodings = []
    pids = []
    for i, fold in enumerate(folds):
        logger.info(f"Encoding fold {i+1}")
        val_data = data.filter_by_pids(fold["val"])
        pids.extend(val_data.get_pids())
        val_data = BinaryOutcomeDataset(val_data.patients)
        val_loader = DataLoader(val_data, collate_fn=dynamic_padding, **loader_cfg)
        fold_model_dir = join(model_dir, f"fold_{i+1}")
        epoch = get_last_checkpoint_epoch(join(fold_model_dir, CHECKPOINTS_DIR))

        model = ModelLoader(fold_model_dir, epoch).load_model(BertForFineTuning)
        model.eval()
        model.to(device)

        with torch.no_grad():
            for batch in get_tqdm(val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                model(batch)
                pooled_rep = model.cls.pool.last_pooled_output
                encodings.append(pooled_rep)
    encodings = torch.cat(encodings, dim=0)

    torch.save(encodings, join(write_dir, ENCODINGS_FILE))
    torch.save(pids, join(write_dir, PID_FILE))
    return encodings
