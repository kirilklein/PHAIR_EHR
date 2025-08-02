import pandas as pd

from corebehrt.constants.causal.data import (
    DEATH_COL,
    END_COL,
    END_TIME_COL,
    NON_COMPLIANCE_COL,
    START_COL,
    START_TIME_COL,
    GROUP_COL,
    CONTROL_PID_COL,
)
from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.functional.preparation.causal.utils import (
    filter_df_by_unique_values,
    get_non_compliance_abspos,
    get_group_dict,
)
from corebehrt.modules.preparation.causal.config import OutcomeConfig


def get_combined_follow_ups(
    index_dates: pd.DataFrame,
    index_date_matching: pd.DataFrame,
    deaths: pd.Series,
    exposures: pd.DataFrame,
    data_end: pd.Timestamp,
    cfg: OutcomeConfig,
) -> pd.DataFrame:
    """
    Create follow-up windows.

    This method accounts for multiple censoring events:
    1. Death events (from deaths parameter)
    2. Non-compliance periods (last exposure + n_hours_compliance)
    3. End of follow-up periods

    The final follow-up period for each patient is the minimum of these three values.

    Args:
        index_dates: DataFrame with columns 'subject_id', 'abspos' (index dates)
        n_hours_start_follow_up: Hours after index date to start follow-up
        n_hours_end_follow_up: Hours after index date to end follow-up
        n_hours_compliance: Hours to add to last exposure for non-compliance cutoff
        index_date_matching: DataFrame defining matched groups (control_subject_id, exposed_subject_id)
        deaths: Series with death times
        exposures: DataFrame with columns 'subject_id', 'abspos' (exposure events)
        data_end: Timestamp of the end of the data
        group_wise_follow_up: Whether to group-wise follow-up
        delay_death_hours: Hours to add to death time for outcomes that are coded with a delay

    Returns:
        - adjusted_follow_ups: pd.DataFrame with final follow-up periods with following column
    """
    follow_ups = prepare_follow_ups_simple(
        index_dates, cfg.n_hours_start_follow_up, cfg.n_hours_end_follow_up, data_end
    )  # simply based on n_hours_start_follow_up and n_hours_end_follow_up

    non_compliance_abspos = get_non_compliance_abspos(exposures, cfg.n_hours_compliance)
    follow_ups = prepare_follow_ups_adjusted(
        follow_ups,
        non_compliance_abspos,
        deaths,
        cfg.delay_death_hours,
    )  # based on non-compliance, death, and group

    if cfg.group_wise_follow_up:  # make group-wise follow-up times shorter
        if index_date_matching is None:
            raise ValueError("index_date_matching is required for group-wise follow-up")
        index_date_matching = filter_df_by_unique_values(
            index_date_matching, index_dates, CONTROL_PID_COL, PID_COL
        )
        all_pids = index_dates[PID_COL].unique()
        group_dict = get_group_dict(index_date_matching)
        for pid in all_pids:
            if pid not in group_dict:
                group_dict[pid] = len(group_dict)

        follow_ups[GROUP_COL] = follow_ups[PID_COL].map(group_dict)
        follow_ups[GROUP_COL] = follow_ups[GROUP_COL].astype(int)
        follow_ups = minimize_end_by_group(follow_ups)

    return follow_ups


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
