from typing import List, Tuple

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL
from corebehrt.constants.data import DEATHDATE_COL, PID_COL, TIMESTAMP_COL


def select_time_eligible_exposed(index_dates: pd.DataFrame, time_windows: dict) -> list:
    """
    Filter exposed patients by time window eligibility.

    Patients must have sufficient lookback and follow-up periods within the data
    availability window to be included in the study.

    Parameters
    ----------
    index_dates : pd.DataFrame
        DataFrame with patient IDs and their index dates (first exposure)
    time_windows : dict
        Dictionary containing:
        - min_follow_up: minimum time after index date
        - min_lookback: minimum time before index date
        - data_start/data_end: data availability boundaries

    Returns
    -------
    list
        Patient IDs meeting all time window requirements

    Raises
    ------
    ValueError
        If duplicate patient IDs are found in index_dates
    """
    if index_dates.duplicated(subset=PID_COL).any():
        raise ValueError("Duplicate patient IDs found in index_dates")
    if index_dates.empty:
        return []
    min_follow_up = pd.Timedelta(**time_windows["min_follow_up"])
    min_lookback = pd.Timedelta(**time_windows["min_lookback"])
    data_end = pd.Timestamp(**time_windows["data_end"])
    data_start = pd.Timestamp(**time_windows["data_start"])

    sufficient_follow_up = index_dates[TIMESTAMP_COL] + min_follow_up <= data_end
    sufficient_lookback = index_dates[TIMESTAMP_COL] - min_lookback >= data_start
    filtered_index_dates = index_dates[sufficient_follow_up & sufficient_lookback]

    return filtered_index_dates[PID_COL].unique().tolist()


def draw_index_dates_for_control_with_redraw(
    control_pids: List[str],
    exposed_index_dates: pd.DataFrame,
    patients_info: pd.DataFrame,
    redraw_attempts: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draw index dates for unexposed patients by randomly sampling from exposed patients' index dates.

    This function assigns index dates to unexposed (control) patients by randomly sampling
    from the index dates of exposed patients. It ensures that assigned index dates do not
    occur after a patient's death date. If an invalid date is drawn (after death), the
    function will attempt to redraw up to 2 additional times before excluding the patient.

    Parameters
    ----------
    control_pids : List[str]
        List of patient IDs for unexposed/control patients who need index dates assigned.
    exposed_index_dates : pd.DataFrame
        DataFrame with exposed patient data, must include PID_COL and TIMESTAMP_COL columns.
    patients_info : pd.DataFrame
        DataFrame containing patient information, must include PID_COL and DEATHDATE_COL columns.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - pd.DataFrame: Index dates for valid unexposed patients with columns [PID_COL, TIMESTAMP_COL]
        - pd.DataFrame: Matching information with columns ['exposed_pid', 'index_date'] showing
          which exposed patient each unexposed patient was matched to and their assigned index date

    Notes
    -----
    - Patients who die before their assigned index date are excluded after 2 failed attempts
    - The matching process uses random sampling with replacement from exposed patients
    - The function prioritizes validity over maintaining exact sample size
    """

    # Get death dates for unexposed patients
    death_info = patients_info.set_index(PID_COL)[DEATHDATE_COL].reindex(control_pids)

    # Convert exposed info to arrays for faster sampling
    exposed_dates_array = exposed_index_dates[TIMESTAMP_COL].values
    exposed_pids_array = exposed_index_dates[PID_COL].values
    n_exposed = len(exposed_dates_array)
    n_unexposed = len(control_pids)
    if n_exposed == 0:
        raise ValueError("Cannot draw index dates: no exposed patients available")
    # Draw initial random indices
    sampled_indices = np.random.choice(n_exposed, size=n_unexposed, replace=True)

    # Create DataFrame with all necessary info
    temp_df = pd.DataFrame(
        {
            TIMESTAMP_COL: exposed_dates_array[sampled_indices],
            DEATHDATE_COL: death_info.values,
            EXPOSED_PID_COL: exposed_pids_array[sampled_indices],
            CONTROL_PID_COL: control_pids,
        },
        index=control_pids,
    )

    for _ in range(redraw_attempts):
        temp_df, resampled = _resample_invalid_dates(
            temp_df, exposed_dates_array, exposed_pids_array
        )
        if not resampled:
            break

    return _finalize_control(temp_df)


def _resample_invalid_dates(
    temp_df: pd.DataFrame,
    exposed_dates_array: np.ndarray,
    exposed_pids_array: np.ndarray,
) -> Tuple[pd.DataFrame, bool]:
    """
    Resample invalid dates.
    Return temp_df with resampled dates and a boolean indicating if any invalid dates were resampled.
    """
    invalid = (temp_df[TIMESTAMP_COL] > temp_df[DEATHDATE_COL]) & pd.notna(
        temp_df[DEATHDATE_COL]
    )
    if not invalid.any():
        return temp_df, False
    new_idx = np.random.choice(
        len(exposed_dates_array), size=invalid.sum(), replace=True
    )
    temp_df.loc[invalid, TIMESTAMP_COL] = exposed_dates_array[new_idx]
    temp_df.loc[invalid, EXPOSED_PID_COL] = exposed_pids_array[new_idx]
    return temp_df, True


def _finalize_control(temp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finalize control index dates and matching.
    Return index dates and matching.
    """
    mask = (temp_df[TIMESTAMP_COL] > temp_df[DEATHDATE_COL]) & pd.notna(
        temp_df[DEATHDATE_COL]
    )
    valid = temp_df.loc[~mask]
    index_dates = pd.DataFrame(
        {PID_COL: valid.index, TIMESTAMP_COL: valid[TIMESTAMP_COL]}
    )
    matching = valid[[EXPOSED_PID_COL, TIMESTAMP_COL]].reset_index(
        drop=False, names=CONTROL_PID_COL
    )
    return index_dates, matching
