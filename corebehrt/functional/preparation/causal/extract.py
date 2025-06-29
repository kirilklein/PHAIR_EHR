import numpy as np

from corebehrt.modules.preparation.causal.dataset import CausalPatientData


def extract_death(patient: CausalPatientData, death_token: int) -> int:
    """
    Extracts the death absolute position from a patient.
    If the patient does not have a death, returns None.
    Args:
        patient: CausalPatientData
        death_token: int
    Returns:
        int: The death absolute position or None if the patient does not have a death.
    """
    if death_token in patient.concepts:
        return patient.abspos[patient.concepts.index(death_token)]
    return np.nan
