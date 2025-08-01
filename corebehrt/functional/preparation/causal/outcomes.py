import pandas as pd

from corebehrt.constants.causal.data import CONTROL_PID_COL
from corebehrt.constants.data import PID_COL
from corebehrt.functional.preparation.causal.convert import abspos_to_binary_outcome
from corebehrt.functional.preparation.causal.follow_up import (
    prepare_follow_ups_adjusted,
    prepare_follow_ups_simple,
)
from corebehrt.functional.preparation.causal.utils import (
    filter_df_by_unique_values,
    get_non_compliance_abspos,
)


def get_binary_exposure(
    exposures: pd.DataFrame,
    index_dates: pd.DataFrame,
    n_hours_start_follow_up: int,
    n_hours_end_follow_up: int,
    data_end: pd.Timestamp,
) -> pd.Series:
    """
    Create binary exposure indicators for patients based on exposure events within follow-up periods.

    Since index dates were determined using exposure criteria, this method uses simple
    follow-up windows without additional adjustments for compliance or deaths.

    Args:
        exposures: DataFrame with columns 'subject_id', 'abspos' (exposure events)
        index_dates: DataFrame with columns 'subject_id', 'abspos' (index dates)
        n_hours_start_follow_up: Hours after index date to start follow-up
        n_hours_end_follow_up: Hours after index date to end follow-up

    Returns:
        pd.Series: Binary exposure indicator (1 if exposed during follow-up, 0 otherwise)
    """
    follow_ups = prepare_follow_ups_simple(
        index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
    )
    return abspos_to_binary_outcome(follow_ups, exposures)


def get_binary_outcome(
    index_dates: pd.DataFrame,
    outcomes: pd.DataFrame,
    n_hours_start_follow_up: int,
    n_hours_end_follow_up: int,
    n_hours_compliance: int,
    index_date_matching: pd.DataFrame,
    deaths: pd.Series,
    exposures: pd.DataFrame,
    data_end: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Create binary outcome indicators for patients using adjusted follow-up periods.

    This method accounts for multiple censoring events:
    1. Death events (from deaths parameter)
    2. Non-compliance periods (last exposure + n_hours_compliance)
    3. End of follow-up periods

    The final follow-up period for each patient is the minimum of these three values.

    Args:
        index_dates: DataFrame with columns 'subject_id', 'abspos' (index dates)
        outcomes: DataFrame with columns 'subject_id', 'abspos' (outcome events)
        n_hours_start_follow_up: Hours after index date to start follow-up
        n_hours_end_follow_up: Hours after index date to end follow-up
        n_hours_compliance: Hours to add to last exposure for non-compliance cutoff
        index_date_matching: DataFrame defining matched groups (control_subject_id, exposed_subject_id)
        deaths: Series mapping patient IDs to death times (NaN if no death)
        exposures: DataFrame with columns 'subject_id', 'abspos' (exposure events)

    Returns:
        tuple: (binary_outcomes, adjusted_follow_ups)
            - binary_outcomes: pd.Series with binary outcome indicators
            - adjusted_follow_ups: pd.DataFrame with final follow-up periods
    """
    if index_date_matching is not None:
        index_date_matching = filter_df_by_unique_values(
            index_date_matching, index_dates, CONTROL_PID_COL, PID_COL
        )
    non_compliance_abspos = get_non_compliance_abspos(exposures, n_hours_compliance)
    follow_ups = prepare_follow_ups_simple(
        index_dates, n_hours_start_follow_up, n_hours_end_follow_up, data_end
    )
    follow_ups = prepare_follow_ups_adjusted(
        follow_ups,
        non_compliance_abspos,
        deaths,
    )
    return abspos_to_binary_outcome(follow_ups, outcomes), follow_ups
