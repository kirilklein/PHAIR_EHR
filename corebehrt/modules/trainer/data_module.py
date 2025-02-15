from datetime import datetime
from os.path import join
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from corebehrt.constants.data import ABSPOS_COL, TIMESTAMP_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import DATA_CFG, FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.functional.causal.load import (
    load_encodings_and_pids_from_encoded_dir,
    load_exposure_from_predictions,
)
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.utils.filter import filter_folds_by_pids
from corebehrt.functional.utils.time import get_abspos_from_origin_point
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.trainer.dataset import SimpleDataset


class EncodedDataModule:
    """Data module for handling encoded patient data and exposures for model training.

    This class manages the loading and preprocessing of encoded patient data, including:
    - Loading patient encodings and exposure predictions
    - Managing train/validation splits via folds
    - Loading and processing outcome data
    - Creating PyTorch DataLoaders for training

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing:
            - paths: Directory paths for data files
            - outcome: Parameters for outcome processing
            - trainer_args: DataLoader configuration
        logger (Any): Logger instance for tracking setup progress

    Attributes:
        X (torch.Tensor): Combined tensor of patient encodings and exposures
        y (torch.Tensor): Binary outcome tensor
        pids (List[str]): List of patient IDs
        folds (List[Dict]): Train/validation split configurations
        input_dim (int): Dimension of the input features
    """

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        self.X = None
        self.y = None
        self.pids: List[str] = []
        self.folds: List[Dict] = []
        self.input_dim = 0

    def setup(self) -> None:
        self.logger.info("Loading encodings and exposures")
        encodings, pids = load_encodings_and_pids_from_encoded_dir(
            self.cfg.paths.encoded_data
        )
        exposure = load_exposure_from_predictions(
            self.cfg.paths.calibrated_predictions, pids
        )
        self.X = torch.cat(
            [encodings, exposure.unsqueeze(1) - 0.5], dim=1
        )  # center exposure at 0
        self.input_dim = self.X.shape[1]

        self.logger.info("Loading index dates and folds")
        index_dates, folds, pids = self._load_index_dates_and_folds()
        self.folds = filter_folds_by_pids(folds, pids)
        self.pids = pids

        self.logger.info("Loading outcomes")
        self.y = self._load_outcomes(index_dates, pids)

    def _load_index_dates_and_folds(self) -> Tuple[pd.DataFrame, List[Dict], List[str]]:
        index_dates = pd.read_csv(
            join(self.cfg.paths.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        origin_point = load_config(
            join(self.cfg.paths.encoded_data, DATA_CFG)
        ).features.origin_point
        index_dates[ABSPOS_COL] = get_abspos_from_origin_point(
            index_dates[TIMESTAMP_COL], datetime(**origin_point)
        )
        folds = torch.load(join(self.cfg.paths.cohort, FOLDS_FILE))
        _, pids = load_encodings_and_pids_from_encoded_dir(self.cfg.paths.encoded_data)
        return index_dates, folds, pids

    def _load_outcomes(
        self, index_dates: pd.DataFrame, pids: List[str]
    ) -> torch.Tensor:
        outcomes = pd.read_csv(self.cfg.paths.outcomes)
        binary_outcomes = get_binary_outcomes(
            index_dates,
            outcomes,
            self.cfg.outcome.n_hours_start_follow_up,
            self.cfg.outcome.n_hours_end_follow_up,
        )
        binary_outcomes = binary_outcomes.loc[pids]
        return torch.tensor(binary_outcomes.values, dtype=torch.float32)

    def get_fold_dataloaders(self, fold: Dict) -> Tuple[DataLoader, DataLoader]:
        val_fold_pids = fold[VAL_KEY]
        train_fold_pids = fold[TRAIN_KEY]
        val_fold_ids = [i for i, pid in enumerate(self.pids) if pid in val_fold_pids]
        train_fold_ids = [
            i for i, pid in enumerate(self.pids) if pid in train_fold_pids
        ]

        print("X", len(self.X))
        print("y", len(self.y))
        print("train_fold_ids", len(train_fold_ids))
        print("val_fold_ids", len(val_fold_ids))
        print("validation pids", len(val_fold_pids))
        print("train pids", len(train_fold_pids))

        X_train = self.X[train_fold_ids]
        X_val = self.X[val_fold_ids]
        y_train = self.y[train_fold_ids]
        y_val = self.y[val_fold_ids]

        train_dataset = SimpleDataset(X_train, y_train)
        val_dataset = SimpleDataset(X_val, y_val)

        train_loader = DataLoader(
            dataset=train_dataset,
            **self.cfg.trainer_args.train_loader_kwargs,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            shuffle=False,  # Never shuffle validation set
            **self.cfg.trainer_args.val_loader_kwargs,
        )
        return train_loader, val_loader
