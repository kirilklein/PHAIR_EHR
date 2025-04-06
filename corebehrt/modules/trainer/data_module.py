import logging
from os.path import join
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from corebehrt.constants.data import ABSPOS_COL, TIMESTAMP_COL, TRAIN_KEY, VAL_KEY
from corebehrt.constants.paths import FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.functional.causal.load import (
    load_encodings_and_pids_from_encoded_dir,
    load_exposure_from_predictions,
)
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.utils.filter import filter_folds_by_pids
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.trainer.dataset import SimpleDataset
from corebehrt.modules.trainer.utils import get_sampler


def load_and_prepare_data(
    encoded_data_path: str, calibrated_predictions_path: str, logger: logging.Logger
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load and prepare factual and counterfactual features.

    Args:
        encoded_data_path: Path to encoded data
        calibrated_predictions_path: Path to calibrated predictions
        logger: Logger instance

    Returns:
        Tuple containing:
        - X: Factual features
        - X_cf: Counterfactual features
        - pids: List of patient IDs
    """
    logger.info("Loading encodings and exposures")
    encodings, pids = load_encodings_and_pids_from_encoded_dir(encoded_data_path)
    exposure = load_exposure_from_predictions(calibrated_predictions_path, pids)

    # Create factual and counterfactual features
    X, X_cf = create_factual_counterfactual_features(encodings, exposure)

    return X, X_cf, pids


def create_factual_counterfactual_features(
    encodings: torch.Tensor, exposure: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create factual and counterfactual feature tensors.

    Args:
        encodings: Encoded features tensor
        exposure: Exposure tensor

    Returns:
        Tuple of (factual features, counterfactual features)
    """
    centered_exposure = exposure.unsqueeze(1) - 0.5
    X = torch.cat([encodings, centered_exposure], dim=1)
    X_cf = torch.cat([encodings, -centered_exposure], dim=1)

    return X, X_cf


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
        pid_to_idx (dict): Mapping from patient ID to index
    """

    def __init__(self, cfg: Dict[str, Any], logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        self.X = None
        self.y = None
        self.pids: List[str] = []
        self.folds: List[Dict] = []
        self.input_dim = 0
        self.pid_to_idx = {}

    def setup(self) -> None:
        """Setup the data module by loading and preparing data."""
        self.X, self.X_cf, self.pids = load_and_prepare_data(
            self.cfg.paths.encoded_data,
            self.cfg.paths.calibrated_predictions,
            self.logger,
        )
        # Remove GPU transfer, keep everything on CPU
        self.input_dim = self.X.shape[1]

        self.logger.info("Loading index dates and folds")
        index_dates, folds, pids = self._load_index_dates_and_folds()
        self.folds = filter_folds_by_pids(folds, pids)
        self.pids = pids

        self.logger.info("Loading outcomes")
        self.y = self._load_outcomes(index_dates, pids)

        # Create PID mapping once
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.pids)}

    def _load_index_dates_and_folds(self) -> Tuple[pd.DataFrame, List[Dict], List[str]]:
        index_dates = pd.read_csv(
            join(self.cfg.paths.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])
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

    def get_fold_data(
        self, fold: Dict[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use mapping for faster indexing
        val_fold_ids = torch.tensor([self.pid_to_idx[pid] for pid in fold[VAL_KEY]])
        train_fold_ids = torch.tensor([self.pid_to_idx[pid] for pid in fold[TRAIN_KEY]])

        return (
            self.X[train_fold_ids],
            self.X[val_fold_ids],
            self.X_cf[val_fold_ids],
            self.y[train_fold_ids],
            self.y[val_fold_ids],
        )

    def get_fold_dataloaders(
        self, fold: Dict[str, List[str]]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get dataloaders for a given fold.
        Args:
            fold (Dict): A dictionary containing training and validation patient IDs

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: A tuple of three DataLoader objects
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - val_counter_loader: DataLoader for counterfactual validation data
        """
        X_train, X_val, X_val_counter, y_train, y_val = self.get_fold_data(fold)
        train_dataset = SimpleDataset(X_train, y_train)
        val_dataset = SimpleDataset(X_val, y_val)
        val_counter_dataset = SimpleDataset(X_val_counter, y_val)
        # Get the sampler from your function. It returns None if sampling is not enabled.
        sampler = get_sampler(self.cfg, y_train.tolist())

        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            # Disable shuffling if a sampler is provided.
            shuffle=False if sampler is not None else True,
            **self.cfg.trainer_args.train_loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **self.cfg.trainer_args.val_loader_kwargs,
        )

        val_cf_loader = DataLoader(
            val_counter_dataset,
            shuffle=False,
            **self.cfg.trainer_args.val_loader_kwargs,
        )
        return train_loader, val_loader, val_cf_loader
