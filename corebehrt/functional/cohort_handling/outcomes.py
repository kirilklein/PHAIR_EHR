from corebehrt.constants.data import PID_COL, ABSPOS_COL
import pandas as pd


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
        in_window = adjust_windows_for_compliance(
            in_window, merged, exposures, index_date_matching, n_hours_compliance
        )
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
    in_window: pd.Series,
    merged: pd.DataFrame,
    exposures: pd.DataFrame,
    index_date_matching: pd.DataFrame,
    n_hours_compliance: float,
) -> pd.Series:
    """
    Adjust the windows for compliance by setting end of follow-up based on last exposure.

    Args:
        in_window: Boolean series indicating if outcomes are within initial window
        merged: DataFrame containing outcome and index date information
        exposures: DataFrame with exposure information
        index_date_matching: DataFrame containing exposed-control subject pairs
        n_hours_compliance: Hours to add after last exposure for end of follow-up

    Returns:
        Boolean series with adjusted windows
    """
    if index_date_matching is None:
        raise ValueError(
            "index_date_matching is required if n_hours_compliance is not None"
        )
    if exposures is None:
        raise ValueError("exposures is required if n_hours_compliance is not None")

    # Get last exposure time for each exposed subject
    last_exposures = exposures.groupby(PID_COL)[ABSPOS_COL].max()

    # Create mapping of exposed subjects to their controls
    exposed_to_controls = (
        index_date_matching.groupby("exposed_subject_id")["control_subject_id"]
        .apply(list)
        .to_dict()
    )

    # Map last exposures to both exposed and control subjects
    merged["last_exposure"] = merged[PID_COL].map(last_exposures)
    for exposed_id, control_ids in exposed_to_controls.items():
        if exposed_id in last_exposures:
            merged.loc[merged[PID_COL].isin(control_ids), "last_exposure"] = (
                last_exposures[exposed_id]
            )

    # Create compliance mask
    compliance_mask = merged[ABSPOS_COL] <= (
        merged["last_exposure"] + n_hours_compliance
    )

    return in_window & compliance_mask
