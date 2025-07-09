import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    DEATH_COL,
    END_COL,
    END_TIME_COL,
    GROUP_COL,
    NON_COMPLIANCE_COL,
    START_COL,
    START_TIME_COL,
)
from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import get_hours_since_epoch


def prepare_follow_ups_simple(
    index_dates: pd.DataFrame,
    n_hours_start_follow_up: int,
    n_hours_end_follow_up: int,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Prepare the follow-ups for the patients.
    If n_hours_end_follow_up is not None, then the follow-up time is the time between the index date and the follow-up time.
    """
    index_dates = index_dates.copy()

    index_dates[START_TIME_COL] = index_dates[TIMESTAMP_COL] + pd.Timedelta(
        hours=n_hours_start_follow_up
    )
    if n_hours_end_follow_up is not None:
        index_dates[END_TIME_COL] = index_dates[TIMESTAMP_COL] + pd.Timedelta(
            hours=n_hours_end_follow_up
        )
    else:
        index_dates[END_TIME_COL] = data_end
    index_dates[START_COL] = get_hours_since_epoch(index_dates[START_TIME_COL])
    index_dates[END_COL] = get_hours_since_epoch(index_dates[END_TIME_COL])
    index_dates = index_dates.drop(columns=[ABSPOS_COL])
    return index_dates


def prepare_follow_ups_adjusted(
    follow_ups: pd.DataFrame,
    non_compliance_abspos: pd.Series,
    deaths: dict,
    group_dict: dict,
) -> pd.DataFrame:
    """
    Prepare the follow-ups for the patients.
    The follow-up time for each patient is the minimum of the follow-up time, non-compliance time, and death time.
    You can set non-compliance to a large value to ensure that the follow-up time is the follow-up time.

    Args:
        follow_ups: DataFrame with follow-up information
        non_compliance_abspos: Series with non-compliance times
        deaths: Dictionary mapping patient IDs to death times
        group_dict: Dictionary mapping patient IDs to groups
    """
    follow_ups = follow_ups.copy()

    follow_ups[NON_COMPLIANCE_COL] = follow_ups[PID_COL].map(non_compliance_abspos)
    follow_ups[DEATH_COL] = follow_ups[PID_COL].map(deaths)
    follow_ups[GROUP_COL] = follow_ups[PID_COL].map(group_dict)

    follow_ups[END_COL] = follow_ups[[END_COL, NON_COMPLIANCE_COL, DEATH_COL]].min(
        axis=1
    )
    follow_ups[END_COL] = follow_ups[END_COL].fillna(follow_ups[END_COL].max())
    return follow_ups
