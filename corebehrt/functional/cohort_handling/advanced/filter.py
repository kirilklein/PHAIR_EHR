from corebehrt.constants.data import PID_COL
import pandas as pd


def filter_by_compliant(
    patients_info: pd.DataFrame, exposures: pd.DataFrame, min_exposures: int
):
    """
    Filter patients based on the number of exposures.
    Patients with at least min_exposures are kept.
    Patients with no exposures are also included by default.
    """
    # Get patients with enough exposures
    counts = exposures.groupby(PID_COL).size().reset_index(name="exposure_count")
    compliant = counts[counts["exposure_count"] >= min_exposures]

    # Get patients with no exposures
    no_exposures = patients_info[~patients_info[PID_COL].isin(counts[PID_COL])]

    # Combine compliant patients and those with no exposures
    return pd.concat(
        [patients_info[patients_info[PID_COL].isin(compliant[PID_COL])], no_exposures]
    )
