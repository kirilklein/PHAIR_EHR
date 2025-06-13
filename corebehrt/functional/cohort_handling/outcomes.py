import pandas as pd

from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL
from corebehrt.constants.data import ABSPOS_COL, PID_COL


def get_binary_outcomes(
    index_dates: pd.DataFrame,
    outcomes: pd.DataFrame,
    n_hours_start_follow_up: float = 0,
    n_hours_end_follow_up: float = None,
    n_hours_compliance: float = None,
    index_date_matching: pd.DataFrame = None,
    exposures: pd.DataFrame = None,
) -> pd.Series:
    """Get binary outcomes for each patient.

    Args:
        index_dates: DataFrame with PID_COL and abspos columns
        outcomes: DataFrame with PID_COL and abspos columns
        n_hours_start_follow_up: Hours after index date to start follow-up
        n_hours_end_follow_up: Hours after index date to end follow-up (None for no end)
        n_hours_compliance: Exclude the case and corresponding control as soon as there is no exposure for n_hours_compliance hours
        index_date_matching: DataFrame with PID_COL and index_date columns. Only needed if n_hours_compliance is not None
        exposures: DataFrame with PID_COL and abspos columns
    Returns:
        Series with PID index and int (0 or 1) values indicating if outcome occurred in window
    """
    # Create a mask for outcomes within the follow-up window
    merged = pd.merge(
        outcomes[[PID_COL, ABSPOS_COL]],
        index_dates[[PID_COL, ABSPOS_COL]].rename(columns={ABSPOS_COL: "index_abspos"}),
        on=PID_COL,
    )

    # Calculate relative position from index date
    merged["rel_pos"] = merged[ABSPOS_COL] - merged["index_abspos"]

    # Check if outcome is within window
    in_window = merged["rel_pos"] >= n_hours_start_follow_up
    if n_hours_end_follow_up is not None:
        in_window &= merged["rel_pos"] <= n_hours_end_follow_up
    if n_hours_compliance is not None:
        compliance_mask = adjust_windows_for_compliance(
            merged, exposures, index_date_matching, n_hours_compliance
        )
        in_window &= compliance_mask
    # Group by patient and check if any outcome is within window
    has_outcome = merged[in_window].groupby(PID_COL).size() > 0

    # Ensure all patients from index_dates are included with False for those without outcomes
    result = pd.Series(
        False, index=index_dates[PID_COL].unique(), name="has_outcome", dtype=bool
    )
    result.index_name = PID_COL
    result[has_outcome.index] = has_outcome
    return result.astype(int)


def adjust_windows_for_compliance(
    merged: pd.DataFrame,
    exposures: pd.DataFrame,
    index_date_matching: pd.DataFrame,
    n_hours_compliance: float,
) -> pd.Series:
    """
    Adjust the follow-up windows for compliance.
    Set based on last exposure + n_hours_compliance for exposed
    For controls we use the linking that was used to create the control index dates.

    Args:
        merged: DataFrame containing outcome and index date information
        exposures: DataFrame with exposure information
        index_date_matching: DataFrame containing exposed-control subject pairs
        n_hours_compliance: Hours to add after last exposure for end of follow-up

    Returns:
        Boolean series with True for outcomes within compliance window, False otherwise
    """
    if index_date_matching is None:
        raise ValueError(
            "index_date_matching is required if n_hours_compliance is not None"
        )
    if exposures is None:
        raise ValueError("exposures is required if n_hours_compliance is not None")
    last_exposure_time_col = "last_exposure_time"
    # Get last exposure time for each exposed subject
    last_exposures = exposures.groupby(PID_COL)[ABSPOS_COL].max().reset_index()
    last_exposures.columns = [EXPOSED_PID_COL, last_exposure_time_col]

    # Create mapping for control subjects
    control_mapping = pd.merge(
        index_date_matching[[EXPOSED_PID_COL, CONTROL_PID_COL]],
        last_exposures,
        on=EXPOSED_PID_COL,
        how="left",
    )

    # Create combined mapping for all subjects (both exposed and controls)
    exposed_mapping = last_exposures.rename(columns={EXPOSED_PID_COL: PID_COL})
    control_mapping_clean = (
        control_mapping[[CONTROL_PID_COL, last_exposure_time_col]]
        .rename(columns={CONTROL_PID_COL: PID_COL})
        .dropna()
    )

    all_last_exposures = pd.concat(
        [exposed_mapping, control_mapping_clean], ignore_index=True
    )
    all_last_exposures = all_last_exposures.drop_duplicates(subset=[PID_COL])

    # Create a mapping series that maintains the original index
    last_exposure_map = pd.Series(
        index=merged.index,
        data=merged[PID_COL].map(
            all_last_exposures.set_index(PID_COL)[last_exposure_time_col]
        ),
    )

    # Create compliance mask using the original index
    # If a subject has no exposure mapping (NaN), they should be included (True)
    compliance_mask = pd.Series(True, index=merged.index)
    has_exposure = ~last_exposure_map.isna()
    compliance_mask[has_exposure] = merged.loc[has_exposure, ABSPOS_COL] <= (
        last_exposure_map[has_exposure] + n_hours_compliance
    )

    return compliance_mask
