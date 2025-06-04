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

    def __init__(self, patients):
        super().__init__(patients)
        self.patients: List[CausalPatientData] = patients

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
        valid_attributes = {"outcome", "exposure"}
        if attribute_name not in valid_attributes:
            raise ValueError(
                f"Invalid attribute name: {attribute_name}. Must be one of {valid_attributes}"
            )

        for p in self.patients:
            setattr(p, attribute_name, values[p.pid])

        return self

    def filter_by_pids(self, pids: List[str]) -> "CausalPatientDataset":
        pids_set = set(pids)
        return CausalPatientDataset([p for p in self.patients if p.pid in pids_set])

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
