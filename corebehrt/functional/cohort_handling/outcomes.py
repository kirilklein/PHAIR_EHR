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
        Boolean series with adjusted windows
    """
    if index_date_matching is None:
        raise ValueError(
            "index_date_matching is required if n_hours_compliance is not None"
        )
    if exposures is None:
        raise ValueError("exposures is required if n_hours_compliance is not None")

    # Get last exposure time for each exposed subject
    last_exposures = exposures.groupby(PID_COL)[ABSPOS_COL].max().reset_index()
    last_exposures.columns = ["exposed_subject_id", "last_exposure_time"]

    # Create mapping for control subjects
    control_mapping = pd.merge(
        index_date_matching[["exposed_subject_id", "control_subject_id"]],
        last_exposures,
        on="exposed_subject_id",
        how="left",
    )

    # Create combined mapping for all subjects
    all_last_exposures = pd.concat(
        [
            # Map exposed subjects to their own last exposure
            last_exposures.rename(columns={"exposed_subject_id": PID_COL}),
            # Map control subjects to their exposed subject's last exposure
            control_mapping[["control_subject_id", "last_exposure_time"]].rename(
                columns={"control_subject_id": PID_COL}
            ),
        ]
    ).drop_duplicates(subset=[PID_COL])

    # Create a mapping series that maintains the original index
    last_exposure_map = pd.Series(
        index=merged.index,
        data=merged[PID_COL].map(
            all_last_exposures.set_index(PID_COL)["last_exposure_time"]
        ),
    )

    # Create compliance mask using the original index
    compliance_mask = merged[ABSPOS_COL] <= (last_exposure_map + n_hours_compliance)

    # Handle cases where last_exposure_time is NaN (subjects with no exposures)
    # For these, we keep the original window (no compliance restriction)
    compliance_mask = compliance_mask.fillna(True)

    return compliance_mask
