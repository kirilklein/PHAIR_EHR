import logging
from datetime import datetime
from os.path import join

import pandas as pd
import torch

from corebehrt.constants.data import ABSPOS_COL, TIMESTAMP_COL
from corebehrt.constants.paths import DATA_CFG, FOLDS_FILE, INDEX_DATES_FILE
from corebehrt.functional.causal.load import (
    load_encodings_and_pids_from_encoded_dir,
    load_exposure_from_predictions,
)
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.utils.time import get_abspos_from_origin_point
from corebehrt.main.helper.causal.train_mlp import (
    check_val_fold_pids,
    combine_encodings_and_exposures,
)
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/causal/train_mlp.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_train_mlp()

    # Logger
    logger = logging.getLogger("train_mlp")

    logger.info("Load data")

    # Encodings and exposure
    encodings, pids = load_encodings_and_pids_from_encoded_dir(cfg.paths.encoded_data)
    exposure = load_exposure_from_predictions(cfg.paths.calibrated_predictions, pids)
    X = combine_encodings_and_exposures(encodings, exposure)
    # load index_dates, and cohort pids
    cohort_dir = cfg.paths.cohort
    index_dates = pd.read_csv(
        join(cohort_dir, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
    )
    origin_point = load_config(
        join(cfg.paths.encoded_data, DATA_CFG)
    ).features.origin_point
    index_dates[ABSPOS_COL] = get_abspos_from_origin_point(
        index_dates[TIMESTAMP_COL], datetime(**origin_point)
    )
    folds = torch.load(join(cohort_dir, FOLDS_FILE))

    check_val_fold_pids(folds, pids)

    # Load outcomes
    outcomes = pd.read_csv(cfg.paths.outcomes)
    binary_outcomes = get_binary_outcomes(
        index_dates,
        outcomes,
        cfg.outcome.n_hours_start_follow_up,
        cfg.outcome.n_hours_end_follow_up,
    )
    print(binary_outcomes.shape)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_train(args.config_path)
