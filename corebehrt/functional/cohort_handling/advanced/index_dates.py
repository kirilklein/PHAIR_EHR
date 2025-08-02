from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    CONTROL_PID_COL,
    EXPOSED_PID_COL,
    BIRTH_YEAR_COL,
)
from corebehrt.constants.data import (
    BIRTHDATE_COL,
    DEATHDATE_COL,
    PID_COL,
    TIMESTAMP_COL,
)


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
    control_pids: List[int],
    cases_df: pd.DataFrame,
    patients_info: pd.DataFrame,
    birth_year_tolerance: int = 10,
    redraw_attempts: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draws index dates for controls by sampling from cases matched on birth year.

    This function uses a vectorized approach to assign index dates from cases to controls,
    ensuring that controls have a similar age distribution. It redraws for controls whose
    assigned date is invalid (e.g., after their death).

    Parameters
    ----------
    control_pids : List of patient IDs for unexposed/control patients who need index dates assigned.
    cases_df : DataFrame of cases index dates, must include PID_COL, TIMESTAMP_COL.
    patients_info : DataFrame with patient info, including PID_COL, BIRTHDATE_COL, and DEATHDATE_COL.
    birth_year_tolerance : int, default=3
        The +/- range for matching birth years.
    redraw_attempts : int, default=2
        Number of redraw attempts for invalid dates.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
        - DataFrame of valid control index dates.
        - DataFrame of matching information.

    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    # 1. Prepare data and add birth year information
    if cases_df.empty:
        raise ValueError("Cannot draw index dates: cases_df is empty.")

    cases_info = _prepare_cases_df(cases_df, patients_info)
    controls_info = _prepare_controls_info(patients_info, control_pids)

    case_pools = _create_stratified_pools(
        cases_info, controls_info, birth_year_tolerance
    )

    control_byears = controls_info.set_index(CONTROL_PID_COL).loc[
        control_pids, BIRTH_YEAR_COL
    ]

    sampled_case_indices = _sample_stratified_indices(control_byears, case_pools)

    temp_df = _construct_initial_matching_df(
        cases_info, controls_info, control_byears, sampled_case_indices
    )

    # 5. Iteratively redraw for invalid matches
    for i in range(redraw_attempts):
        print(f"Performing redraw attempt {i + 1}...")
        temp_df, resampled = _resample_invalid_dates_stratified(
            temp_df, case_pools, cases_info
        )
        if not resampled:
            print("No invalid dates to resample. Stopping.")
            break
    result = _finalize_control(temp_df)
    return result


def _resample_invalid_dates_stratified(
    temp_df: pd.DataFrame, case_pools: Dict[int, np.ndarray], cases_info: pd.DataFrame
) -> Tuple[pd.DataFrame, bool]:
    """Resamples invalid dates using a safe, targeted update."""
    invalid_mask = _get_invalid_mask(temp_df)
    if not invalid_mask.any():
        return temp_df, False

    invalid_byears = temp_df.loc[invalid_mask, "control_birth_year"]
    new_case_indices = _sample_stratified_indices(invalid_byears, case_pools)

    # --- This is the safer update logic ---
    # Update the case index for all invalid rows
    temp_df.loc[invalid_mask, "case_idx"] = new_case_indices

    # Get the new case information. This will raise a KeyError on -1, so handle it.
    # We create a temporary series to map new indices to new data safely.
    new_info_map = cases_info.loc[
        np.unique(new_case_indices[new_case_indices != -1]),
        [TIMESTAMP_COL, EXPOSED_PID_COL],
    ]

    new_timestamps = pd.Series(new_case_indices, index=invalid_byears.index).map(
        new_info_map[TIMESTAMP_COL]
    )
    new_pids = pd.Series(new_case_indices, index=invalid_byears.index).map(
        new_info_map[EXPOSED_PID_COL]
    )

    # Update only the invalid rows with the new data
    temp_df.loc[invalid_mask, TIMESTAMP_COL] = new_timestamps
    temp_df.loc[invalid_mask, EXPOSED_PID_COL] = new_pids

    return temp_df, True


def _construct_initial_matching_df(
    cases_info: pd.DataFrame,
    controls_info: pd.DataFrame,
    control_byears: pd.Series,
    sampled_case_indices: np.ndarray,
) -> pd.DataFrame:
    """Constructs the initial matching DataFrame, ensuring data alignment."""

    # Create the base DataFrame using the index from control_byears to guarantee alignment
    temp_df = pd.DataFrame(
        {"control_birth_year": control_byears.values, "case_idx": sampled_case_indices},
        index=control_byears.index,
    )

    # Merge control birth/death dates using the index
    temp_df = temp_df.merge(
        controls_info.set_index(CONTROL_PID_COL)[[BIRTHDATE_COL, DEATHDATE_COL]],
        left_index=True,
        right_index=True,
        how="left",
    )

    # Merge case info using the 'case_idx' column
    temp_df = temp_df.merge(
        cases_info[[TIMESTAMP_COL, EXPOSED_PID_COL]].rename_axis("case_idx"),
        on="case_idx",
        how="left",
    )
    return temp_df


def _create_stratified_pools(
    cases_info: pd.DataFrame, controls_info: pd.DataFrame, birth_year_tolerance: int
) -> Dict[int, np.ndarray]:
    """Creates stratified pools of case indices efficiently."""

    # 1. Group cases by birth year once to avoid re-scanning
    case_indices_by_year = {
        year: group.index.to_numpy()
        for year, group in cases_info.groupby(BIRTH_YEAR_COL)
    }

    # 2. Build the final pools by combining the pre-grouped case years
    unique_control_byears = controls_info[BIRTH_YEAR_COL].dropna().unique().astype(int)
    final_case_pools = {}

    for year in unique_control_byears:
        pool_list = [
            case_indices_by_year[case_year]
            for case_year in range(
                year - birth_year_tolerance, year + birth_year_tolerance + 1
            )
            if case_year in case_indices_by_year
        ]

        if pool_list:
            final_case_pools[year] = np.concatenate(pool_list)

    return final_case_pools


def _sample_stratified_indices(
    control_birth_years: pd.Series, case_pools: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Performs stratified sampling based on birth year.

    For each control birth year provided, it randomly selects a case index
    from the corresponding pre-calculated pool.
    Args:
        control_birth_years (pd.Series): A Series of birth years for the controls to sample for.
        case_pools (Dict[int, np.ndarray]): A dictionary mapping a birth year to an
                                             array of valid case indices.
    Returns: np.ndarray: An array of sampled case indices, with -1 for unmatchable years.
    """
    # The .get(y, [-1]) safely handles controls whose birth year has no matching case pool
    return np.array(
        [np.random.choice(case_pools.get(y, [-1])) for y in control_birth_years]
    )


def _prepare_controls_info(patients_info: pd.DataFrame, control_pids) -> pd.DataFrame:
    """
    Prepare controls info for index date matching.
    Add birth year column and drop rows with missing birth year.
    Return controls DataFrame with birth year column.
    """
    controls_info = patients_info[patients_info[PID_COL].isin(control_pids)].copy()
    controls_info[BIRTH_YEAR_COL] = pd.to_datetime(controls_info[BIRTHDATE_COL]).dt.year
    controls_info.rename(columns={PID_COL: CONTROL_PID_COL}, inplace=True)
    return controls_info


def _prepare_cases_df(
    cases_df: pd.DataFrame, patients_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare cases DataFrame for index date matching.
    Add birth year column and drop rows with missing birth year.
    Return cases DataFrame with birth year column.
    """
    cases_info = pd.merge(cases_df, patients_info[[PID_COL, BIRTHDATE_COL]], on=PID_COL)
    cases_info[BIRTH_YEAR_COL] = pd.to_datetime(cases_info[BIRTHDATE_COL]).dt.year
    cases_info.dropna(subset=[BIRTH_YEAR_COL], inplace=True)
    cases_info[BIRTH_YEAR_COL] = cases_info[BIRTH_YEAR_COL].astype(int)
    cases_info.rename(columns={PID_COL: EXPOSED_PID_COL}, inplace=True)
    return cases_info


def _finalize_control(temp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finalize control index dates and matching.
    Return index dates and matching.
    """
    invalid_mask = _get_invalid_mask(temp_df)
    valid = temp_df.loc[~invalid_mask]
    if not valid.empty:
        valid[EXPOSED_PID_COL] = valid[EXPOSED_PID_COL].astype(int)

    index_dates = pd.DataFrame(
        {PID_COL: valid.index, TIMESTAMP_COL: valid[TIMESTAMP_COL]}
    )
    matching = valid[[EXPOSED_PID_COL, TIMESTAMP_COL]].reset_index(
        drop=False, names=CONTROL_PID_COL
    )
    return index_dates, matching


def _get_invalid_mask(temp_df: pd.DataFrame) -> pd.Series:
    """Gets an invalid mask, also checking for missing index dates."""

    # A row is invalid if the index date is missing (no match found)
    invalid_match = temp_df[TIMESTAMP_COL].isna()

    # Or if the date is after death
    after_death = (temp_df[TIMESTAMP_COL] > temp_df[DEATHDATE_COL]) & pd.notna(
        temp_df[DEATHDATE_COL]
    )

    # Or if the date is before birth (and birth date is known)
    before_birth = (temp_df[TIMESTAMP_COL] < temp_df[BIRTHDATE_COL]) & pd.notna(
        temp_df[BIRTHDATE_COL]
    )

    return invalid_match | after_death | before_birth
