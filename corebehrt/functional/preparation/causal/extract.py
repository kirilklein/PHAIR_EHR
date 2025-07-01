import numpy as np

from corebehrt.modules.preparation.causal.dataset import CausalPatientData


def extract_death(patient: CausalPatientData, death_token: int) -> float:
    """
    Extracts the death absolute position from a patient.
    Args:
        patient: CausalPatientData
        death_token: int
    Returns:
        float: The death absolute position or np.nan if the patient does not have a death.
    """
    if death_token in patient.concepts:
        return patient.abspos[patient.concepts.index(death_token)]
    return np.nan
