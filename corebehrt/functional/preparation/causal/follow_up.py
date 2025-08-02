import pandas as pd

from corebehrt.constants.causal.data import (
    DEATH_COL,
    END_COL,
    END_TIME_COL,
    NON_COMPLIANCE_COL,
    START_COL,
    START_TIME_COL,
    GROUP_COL,
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
    deaths: pd.Series,
    delay_death_hours: int = 0,
) -> pd.DataFrame:
    """
    Prepare the follow-ups for the patients.
    The follow-up time for each patient is the minimum of the follow-up time, non-compliance time, and death time.
    You can set non-compliance to a large value to ensure that the follow-up time is the follow-up time.

    Args:
        follow_ups: DataFrame with follow-up information
        non_compliance_abspos: Series with non-compliance times
        deaths: Series with death times
        delay_death_hours: Hours to add to death time for outcomes that are coded with a delay
    """
    follow_ups = follow_ups.copy()

    follow_ups[NON_COMPLIANCE_COL] = follow_ups[PID_COL].map(non_compliance_abspos)
    follow_ups[DEATH_COL] = follow_ups[PID_COL].map(deaths)
    follow_ups["delayed_death"] = (
        follow_ups[DEATH_COL] + delay_death_hours
    )  # for outcomes that are coded with a delay
    follow_ups[END_COL] = follow_ups[
        [END_COL, NON_COMPLIANCE_COL, "delayed_death"]
    ].min(
        axis=1
    )  # end follow-up if patient dies, non-complies, or the follow-up period ends
    if follow_ups[END_COL].isna().all():
        follow_ups[END_COL] = get_hours_since_epoch(
            pd.Timestamp.now()
        )  # just a safeguard
    follow_ups[END_COL] = follow_ups[END_COL].fillna(
        follow_ups[END_COL].max()
    )  # if all none for a patient (if no follow-up end is set) and patient is alive and compliant, then set to the overall max

    return follow_ups


def minimize_end_by_group(follow_ups: pd.DataFrame) -> pd.DataFrame:
    """
    Minimize the end time by group.
    We get shorter follow-up times for patients in the same (index date) group.
    """
    follow_ups[END_COL] = follow_ups.groupby(GROUP_COL)[END_COL].transform("min")
    return follow_ups
