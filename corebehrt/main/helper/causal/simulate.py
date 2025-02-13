from typing import Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal import TARGETS
from corebehrt.functional.causal.simulate import simulate_outcome_from_encodings


def simulate(
    logger, encodings: np.ndarray, predictions: pd.DataFrame, simulate_cfg: dict
) -> Tuple[np.ndarray, np.ndarray]:

    exposure = predictions[TARGETS]

    logger.info("simulate actual outcome")
    outcome = simulate_outcome_from_encodings(encodings, exposure, **simulate_cfg)

    logger.info("simulate under exposure")
    all_exposed = np.ones_like(exposure)
    all_exposed_outcome = simulate_outcome_from_encodings(
        encodings, all_exposed, **simulate_cfg
    )

    logger.info("simulate under control")
    all_control = np.zeros_like(exposure)
    all_control_outcome = simulate_outcome_from_encodings(
        encodings, all_control, **simulate_cfg
    )

    logger.info("combine into cf outcome")
    cf_outcome = np.where(exposure == 1, all_control_outcome, all_exposed_outcome)

    return outcome, cf_outcome
