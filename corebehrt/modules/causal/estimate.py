import os
from dataclasses import dataclass
from os.path import join
from typing import Any

import pandas as pd
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from corebehrt.constants.data import (
    PID_COL,
)
from corebehrt.constants.causal import (
    PROBAS,
    TARGETS,
    CALIBRATED_PREDICTIONS_FILE,
    EXPOSURE_COL,
    PS_COL,
)
from corebehrt.constants.paths import (
    OUTCOMES_FILE,
    PROCESSED_DATA_DIR,
)
from corebehrt.modules.setup.config import Config


@dataclass
class EffectEstimator:
    cfg: Config
    logger: Any

    def run(self):
        self.logger.info("Loading data")
        df = self._load_data()
        print(df.head())

    def _load_data(self) -> pd.DataFrame:
        # Load exposure predictions (propensity scores)
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
