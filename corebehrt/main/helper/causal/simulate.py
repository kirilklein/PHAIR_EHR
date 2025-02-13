from typing import Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal import CF_OUTCOMES, CF_PROBAS, OUTCOMES, PROBAS, TARGETS
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.causal.simulate import simulate_outcome_from_encodings

DATE_FUTURE = pd.Timestamp("2100-01-01")


def simulate(
    logger, encodings: np.ndarray, predictions: pd.DataFrame, simulate_cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    exposure = predictions[TARGETS]

    logger.info("simulate actual outcome")
    outcome, proba = simulate_outcome_from_encodings(
        encodings, exposure, **simulate_cfg
    )

    logger.info("simulate under exposure")
    all_exposed = np.ones_like(exposure)
    all_exposed_outcome, all_exposed_proba = simulate_outcome_from_encodings(
        encodings, all_exposed, **simulate_cfg
    )

    logger.info("simulate under control")
    all_control = np.zeros_like(exposure)
    all_control_outcome, all_control_proba = simulate_outcome_from_encodings(
        encodings, all_control, **simulate_cfg
    )

    logger.info("combine into cf outcome")
    cf_outcome = combine_counterfactuals(
        exposure, all_exposed_outcome, all_control_outcome
    )
    cf_proba = combine_counterfactuals(exposure, all_exposed_proba, all_control_proba)

    results_df = pd.DataFrame(
        {
            PID_COL: predictions[PID_COL],
            OUTCOMES: outcome,
            CF_OUTCOMES: cf_outcome,
            PROBAS: proba,
            CF_PROBAS: cf_proba,
        }
    )

    timestamp_df = get_timestamp_df(results_df)
    return results_df, timestamp_df


def get_timestamp_df(results_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PID_COL: results_df[results_df[OUTCOMES] == 1][PID_COL],
            TIMESTAMP_COL: DATE_FUTURE,
        }
    )


def combine_counterfactuals(exposure, exposed_values, control_values):
    """Combines counterfactual values based on exposure status.

    For each individual, returns the opposite of their actual exposure:
    - If exposed (exposure=1), returns their control value
    - If not exposed (exposure=0), returns their exposed value

    Args:
        exposure (numpy.ndarray): Binary array indicating exposure status (1=exposed, 0=control)
        exposed_values (numpy.ndarray): Values under exposure condition
        control_values (numpy.ndarray): Values under control condition

    Returns:
        numpy.ndarray: Combined array where each element is the counterfactual value
            based on the opposite of the actual exposure status
    """
    return np.where(exposure == 1, control_values, exposed_values)
