import pandas as pd
import numpy as np

from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL, GROUP_COL
from corebehrt.constants.data import ABSPOS_COL, PID_COL


def assign_groups_to_followups(
    follow_ups: pd.DataFrame, index_date_matching: pd.DataFrame
) -> pd.DataFrame:
    """
    Assigns a group ID to each patient based on the exposed patient ID.

    The group ID for all patients in a matched set (one exposed patient and all
    their controls) is the ID of the exposed patient. Patients not in a matched
    group (i.e., unmatched exposed patients) will have their own ID as their group ID.

    This function is a direct, vectorized replacement for the combination of
    get_group_dict and the subsequent loop for unmatched patients.

    Args:
        follow_ups: DataFrame containing a PID_COL for all patients in the cohort.
        index_date_matching: DataFrame with CONTROL_PID_COL and EXPOSED_PID_COL.

    Returns:
        The `follow_ups` DataFrame with an added GROUP_COL.
    """
    if index_date_matching is None or index_date_matching.empty:
        # If there's no matching, every patient is in their own group.
        follow_ups[GROUP_COL] = follow_ups[PID_COL]
        return follow_ups

    # Guard against duplicate control IDs to prevent ambiguous mapping
    dupes = index_date_matching[CONTROL_PID_COL].duplicated()
    if dupes.any():
        raise ValueError(
            "Duplicate control PIDs detected in index_date_matching: "
            f"{index_date_matching.loc[dupes, CONTROL_PID_COL].unique()}"
        )
    # 1. Create a map from a control's ID to their matched exposed patient's ID.
    #    This series will have control_pids as the index and exposed_pids as values.#
    control_to_group_map = pd.Series(
        index_date_matching[EXPOSED_PID_COL].values,
        index=index_date_matching[CONTROL_PID_COL],
    )

    # 2. Use this map to assign group IDs to all patients in the follow_ups DataFrame.
    #    - If a patient is a control, it will be mapped to its exposed_pid.
    #    - If a patient is an exposed patient, it won't be in the map's index,
    #      resulting in `NaN`.
    follow_ups[GROUP_COL] = follow_ups[PID_COL].map(control_to_group_map)

    # 3. Fill the `NaN` values. The NaNs correspond to exposed patients.
    #    We fill the NaN with the patient's own ID, making the exposed_pid the group ID.
    follow_ups[GROUP_COL] = follow_ups[GROUP_COL].fillna(follow_ups[PID_COL])

    # Ensure the group column is an integer type.
    follow_ups[GROUP_COL] = follow_ups[GROUP_COL].astype(int)

    return follow_ups


def get_non_compliance_abspos(
    exposures: pd.DataFrame, n_hours_compliance: float
) -> dict:
    """
    Get the last exposure for each patient and add n_hours_compliance.

    Args:
        exposures: DataFrame with columns 'subject_id', 'abspos'
        n_hours_compliance: Hours to add to the last exposure time

    Returns:
        dict: Mapping from patient ID to non-compliance time (last_exposure + n_hours_compliance)

    Example:
        >>> exposures = pd.DataFrame({
        ...     'subject_id': [1, 1, 2, 2, 3],
        ...     'abspos': [100, 200, 150, 300, 250]
        ... })
        >>> n_hours_compliance = 24
        >>> get_non_compliance_abspos(exposures, n_hours_compliance)
        {1: 224, 2: 324, 3: 274}
    """
    if n_hours_compliance is None:
        n_hours_compliance = np.inf
    return {
        pid: last_exposure + n_hours_compliance
        for pid, last_exposure in exposures.groupby(PID_COL)[ABSPOS_COL]
        .max()
        .to_dict()
        .items()
    }


def filter_df_by_unique_values(
    df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str
) -> pd.DataFrame:
    """
    Filter df1 to only include rows where the values in col1 are in the unique values of col2 in df2.
    """
    return df1[df1[col1].isin(df2[col2].unique())]
