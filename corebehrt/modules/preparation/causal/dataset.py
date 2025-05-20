from dataclasses import dataclass
from typing import List

import pandas as pd
import torch

from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.modules.preparation.dataset import (
    BinaryOutcomeDataset,
    PatientData,
    PatientDataset,
)


@dataclass
class CausalPatientData(PatientData):
    exposure: int = None


class CausalPatientDataset(PatientDataset):
    """
    Provides additional functionality for assigning attributes to patients.
    See PatientDataset for more details.
    """

    patients: List[CausalPatientData]

    def assign_attributes(self, attribute_name: str, values: pd.Series):
        """Assigns binary attributes (outcomes or exposures) to each patient in the dataset.

        Takes a pandas Series mapping patient IDs to attributes and assigns them to each patient in the dataset.

        Args:
            attribute_name (str): Name of the attribute to assign ('outcome' or 'exposure')
            values (pd.Series): Series with patient IDs as index and attributes as values.
                The actual attribute values are not used, only whether they are null or not.

        Returns:
            PatientDataset: Returns self for method chaining.
        """
        for p in self.patients:
            setattr(p, attribute_name, values[p.pid])

        return self

    def get_exposures(self):
        return [p.exposure for p in self.patients]


class ExposureOutcomeDataset(BinaryOutcomeDataset):
    """
    outcomes: absolute position when outcome occured for each patient
    exposures: absolute position when exposure occured for each patient
    For details on the dataset structure, see corebehrt.modules.preparation.dataset.BinaryOutcomeDataset.
    """

    def __init__(self, patients: List[CausalPatientData]):
        super().__init__(patients)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample[EXPOSURE_TARGET] = torch.tensor(
            self.patients[index].exposure, dtype=torch.float
        )
        return sample
