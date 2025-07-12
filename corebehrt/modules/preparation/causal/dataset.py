from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch

from corebehrt.constants.causal.data import EXPOSURE_TARGET, OUTCOME_PREFIX
from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
)
from corebehrt.modules.preparation.dataset import PatientData, PatientDataset


@dataclass
class CausalPatientData(PatientData):
    exposure: int = None
    outcomes: dict[str, int] = None


class CausalPatientDataset(PatientDataset):
    """
    Provides additional functionality for assigning attributes to patients.
    See PatientDataset for more details.
    """

    def __init__(self, patients):
        super().__init__(patients)
        self.patients: List[CausalPatientData] = patients

    def assign_attributes(self, attribute_name: str, values: pd.Series) -> None:
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

    def filter_by_pids(self, pids: List[str]) -> "CausalPatientDataset":
        pids_set = set(pids)
        return CausalPatientDataset([p for p in self.patients if p.pid in pids_set])

    def get_exposures(self):
        return [p.exposure for p in self.patients]

    def assign_outcomes(self, outcomes: dict[str, pd.Series]):
        for p in self.patients:
            setattr(p, "outcomes", {k: v[p.pid] for k, v in outcomes.items()})

    def get_outcomes(self) -> Dict[str, List[int]]:
        outcomes = self.patients[0].outcomes.keys()
        return {
            outcome: [p.outcomes[outcome] for p in self.patients]
            for outcome in outcomes
        }

    def get_outcome_names(self) -> List[str]:
        return list(self.patients[0].outcomes.keys())


class ExposureOutcomesDataset:
    """
    Dataset for causal inference with exposure and multiple outcome targets.

    Returns samples with exposure target (single tensor) and outcome targets
    (dictionary mapping outcome names to tensors). Patient data includes concepts,
    absolute positions, segments, ages, and attention masks.

    For base dataset structure details, see corebehrt.modules.preparation.dataset.BinaryOutcomeDataset.
    """

    def __init__(self, patients: List[CausalPatientData]):
        self.patients = patients

    def __getitem__(self, index: int) -> dict:
        patient = self.patients[index]
        attention_mask = torch.ones(
            len(patient.concepts), dtype=torch.long
        )  # Require attention mask for bi-gru head
        sample = {
            CONCEPT_FEAT: torch.tensor(patient.concepts, dtype=torch.long),
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,
        }
        sample[EXPOSURE_TARGET] = torch.tensor(patient.exposure, dtype=torch.float)
        for outcome_name, outcome_value in patient.outcomes.items():
            sample[f"{OUTCOME_PREFIX}{outcome_name}"] = torch.tensor(
                outcome_value, dtype=torch.float
            )
        return sample

    def __len__(self):
        return len(self.patients)
