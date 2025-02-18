from datetime import datetime
from typing import Tuple

import pandas as pd
import torch

from corebehrt.constants.causal import (
    EXPOSURE_COL,
    OUTCOME_CONTROL,
    OUTCOME_EXPOSED,
    OUTCOMES,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
)
from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.causal.counterfactuals import get_true_outcome
from corebehrt.functional.causal.simulate import simulate_outcome_from_encodings
from corebehrt.functional.utils.time import get_abspos_from_origin_point

DATE_FUTURE = pd.Timestamp("2100-01-01")


def simulate(
    logger,
    pids: list,
    encodings: torch.Tensor,
    exposure: torch.Tensor,
    simulate_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

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
            OUTCOME_EXPOSED: all_exposed_outcome,
            OUTCOME_CONTROL: all_control_outcome,
            PROBAS_EXPOSED: all_exposed_proba,
            PROBAS_CONTROL: all_control_proba,
            EXPOSURE_COL: exposure,
            PROBAS: probas,
            OUTCOMES: outcomes,
        }
    )
    timestamp_df = get_timestamp_df(results_df)

    return results_df, timestamp_df


def add_abspos_to_df(df: pd.DataFrame, origin_point: dict) -> pd.DataFrame:
    """Add abspos to df. Use origin point."""
    df[ABSPOS_COL] = get_abspos_from_origin_point(
        df[TIMESTAMP_COL], datetime(**origin_point)
    )
    return df


def get_timestamp_df(results_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PID_COL: results_df[results_df[OUTCOMES] == 1][PID_COL],
            TIMESTAMP_COL: DATE_FUTURE,
        }
    )
