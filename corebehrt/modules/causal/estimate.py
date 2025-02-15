from os.path import join
from typing import Any

import pandas as pd
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table

from corebehrt.constants.causal import (
    CALIBRATED_PREDICTIONS_FILE,
    EXPERIMENT_DATA_FILE,
    EXPERIMENT_STATS_FILE,
    EXPOSURE_COL,
    PROBAS,
    PS_COL,
    TARGETS,
)
from corebehrt.constants.data import PID_COL
from corebehrt.modules.setup.config import Config


class EffectEstimator:
    def __init__(self, cfg: Config, logger: Any):
        self.cfg = cfg
        self.logger = logger
        self.exp_dir = self.cfg.paths.estimate

    def run(self):
        self.logger.info("Loading data")
        df = self._load_data()

        df.to_parquet(join(self.exp_dir, EXPERIMENT_DATA_FILE), index=True)
        stats_table = compute_treatment_outcome_table(df, EXPOSURE_COL, TARGETS)
        stats_table.to_csv(join(self.exp_dir, EXPERIMENT_STATS_FILE))

    def _load_data(self) -> pd.DataFrame:
        """Load exposure and outcome predictions and combine them."""
        exposure_preds = pd.read_csv(
            join(self.cfg.paths.exposure_predictions, CALIBRATED_PREDICTIONS_FILE)
        )
        exposure_preds = exposure_preds.rename(
            columns={
                TARGETS: EXPOSURE_COL,
                PROBAS: PS_COL,
            }
        )

        # Load outcome predictions
        outcome_preds = pd.read_csv(
            join(self.cfg.paths.outcome_predictions, CALIBRATED_PREDICTIONS_FILE)
        )

        # Combine data
        df = pd.merge(exposure_preds, outcome_preds, on=PID_COL)
        return df
