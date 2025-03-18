from datetime import datetime
from typing import Tuple

import pandas as pd
import torch

from corebehrt.constants.causal import (
    EXPOSURE_COL,
    OUTCOMES,
    PROBAS,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
)
from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.causal.counterfactuals import get_true_outcome
from corebehrt.functional.causal.simulate import simulate_outcome_from_encodings
from corebehrt.functional.utils.time import get_hours_since_epoch

DATE_FUTURE = pd.Timestamp("2100-01-01")


def simulate(
    logger,
    pids: list,
    encodings: torch.Tensor,
    exposure: torch.Tensor,
    simulate_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates outcomes under both exposed and control conditions for a set of patients.

    This function takes patient encodings and exposure status and simulates counterfactual
    outcomes - what would have happened if each patient was exposed vs not exposed.

    Args:
        logger: Logger object for tracking simulation progress and debugging
        pids (list): List of patient IDs
        encodings (torch.Tensor): Encoded patient features/characteristics
        exposure (torch.Tensor): Binary tensor indicating actual exposure status
        simulate_cfg (dict): Configuration parameters for the simulation model

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two dataframes containing:
            1. Results dataframe with simulated outcomes and probabilities
            2. Timestamp dataframe with temporal information
    """
    logger.info("simulate under exposure")
    all_exposed = torch.ones_like(exposure)
    all_exposed_outcome, all_exposed_proba = simulate_outcome_from_encodings(
        encodings, all_exposed, **simulate_cfg
    )

    logger.info("simulate under control")
    all_control = torch.zeros_like(exposure)
    all_control_outcome, all_control_proba = simulate_outcome_from_encodings(
        encodings, all_control, **simulate_cfg
    )
    probas = get_true_outcome(exposure, all_exposed_proba, all_control_proba)
    outcomes = get_true_outcome(exposure, all_exposed_outcome, all_control_outcome)
    results_df = pd.DataFrame(
        {
            PID_COL: pids,
            SIMULATED_OUTCOME_EXPOSED: all_exposed_outcome,
            SIMULATED_OUTCOME_CONTROL: all_control_outcome,
            SIMULATED_PROBAS_EXPOSED: all_exposed_proba,
            SIMULATED_PROBAS_CONTROL: all_control_proba,
            EXPOSURE_COL: exposure,
            PROBAS: probas,
            OUTCOMES: outcomes,
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
