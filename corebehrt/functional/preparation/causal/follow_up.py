import pandas as pd
import numpy as np

from corebehrt.constants.causal.data import (
    GROUP_COL,
    START_COL,
    END_COL,
    DEATH_COL,
    NON_COMPLIANCE_COL,
)
from corebehrt.constants.data import PID_COL, ABSPOS_COL


def prepare_follow_ups_simple(
    index_dates: pd.DataFrame, n_hours_start_follow_up: int, n_hours_end_follow_up: int
) -> pd.DataFrame:
    """
    Prepare the follow-ups for the patients.
    """
    if n_hours_end_follow_up is None:
        n_hours_end_follow_up = np.inf
    index_dates = index_dates.copy()
    index_dates[START_COL] = index_dates[ABSPOS_COL] + n_hours_start_follow_up
    index_dates[END_COL] = index_dates[ABSPOS_COL] + n_hours_end_follow_up
    index_dates = index_dates.drop(columns=[ABSPOS_COL])
    return index_dates


def prepare_follow_ups_adjusted(
    follow_ups: pd.DataFrame,
    non_compliance_abspos: dict,
    deaths: dict,
    group_dict: dict,
) -> pd.DataFrame:
    """
    Prepare the follow-ups for the patients.
    """
    follow_ups = follow_ups.copy()

    follow_ups[NON_COMPLIANCE_COL] = follow_ups[PID_COL].map(non_compliance_abspos)
    follow_ups[DEATH_COL] = follow_ups[PID_COL].map(deaths)
    follow_ups[GROUP_COL] = follow_ups[PID_COL].map(group_dict)

    follow_ups[END_COL] = follow_ups[[END_COL, NON_COMPLIANCE_COL, DEATH_COL]].min(
        axis=1
    )
    follow_ups[END_COL] = follow_ups.groupby(GROUP_COL)[END_COL].transform("min")
    return follow_ups
