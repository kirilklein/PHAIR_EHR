import pandas as pd
import numpy as np

from corebehrt.constants.causal.data import CONTROL_PID_COL, EXPOSED_PID_COL, GROUP_COL
from corebehrt.constants.data import ABSPOS_COL, PID_COL


def get_group_dict(index_date_matching: pd.DataFrame) -> dict:
    """
    In a cohort with matched index_dates, each exposed subject and their corresponding unexposed subjects
    form a group. This function assigns a unique integer ID to each of these groups.
    Both exposed and unexposed subjects within the same group will be mapped to the
    same group ID.

    Example:
        >>> import pandas as pd
        >>> matching_df = pd.DataFrame({
        ...     'control_subject_id': [1, 2, 3, 5],
        ...     'exposed_subject_id': [10, 10, 20, 30]
        ... })
        >>> # The returned dictionary is not guaranteed to be in a specific order.
        >>> get_group_dict(matching_df)
        {1: 0, 2: 0, 10: 0, 3: 1, 20: 1, 5: 2, 30: 2}
    """
    index_date_matching = index_date_matching.copy()
    index_date_matching[GROUP_COL] = index_date_matching.groupby(
        EXPOSED_PID_COL
    ).ngroup()
    # Melt the dataframe to get a single column of subject_ids with their corresponding group
    id_to_group = pd.melt(
        index_date_matching,
        id_vars=[GROUP_COL],
        value_vars=[CONTROL_PID_COL, EXPOSED_PID_COL],
        value_name=PID_COL,
    )
    # Create a map from subject_id to group
    subject_to_group_map = (
        id_to_group.drop(columns="variable")
        .drop_duplicates()
        .set_index(PID_COL)[GROUP_COL]
    )
    return subject_to_group_map.to_dict()


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
