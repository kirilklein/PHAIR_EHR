from typing import Tuple

import pandas as pd
import torch
from corebehrt.constants.causal import CF_OUTCOMES, CF_PROBAS, OUTCOMES, PROBAS, TARGETS
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.causal.simulate import (
    combine_counterfactuals,
    simulate_outcome_from_encodings,
)

DATE_FUTURE = pd.Timestamp("2100-01-01")


def simulate(
    logger,
    pids: list,
    encodings: torch.Tensor,
    exposure: torch.Tensor,
    simulate_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("simulate actual outcome")
    outcome, proba = simulate_outcome_from_encodings(
        encodings, exposure, **simulate_cfg
    )

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

    logger.info("combine into cf outcome")
    cf_outcome = combine_counterfactuals(
        exposure, all_exposed_outcome, all_control_outcome
    )
    cf_proba = combine_counterfactuals(exposure, all_exposed_proba, all_control_proba)

    results_df = pd.DataFrame(
        {
            PID_COL: pids,
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
