# config.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

COMMON_INDEX_DATE = pd.Timestamp(
    "2010-01-01"
)  # for simplicity pick the same index_date for all


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    data: str
    splits: List[str]
    outcomes: str


@dataclass
class UnobservedConfounderConfig:
    """Configuration for unobserved confounder."""

    p_occurrence: float
    exposure_effect: float
    outcome_effects: Dict[str, float]


@dataclass
class ModelWeightsConfig:
    """Configuration for sampling model weights."""

    mean: float
    scale: float


@dataclass
class SimulationModelConfig:
    """Configuration for the simulation model."""

    linear: ModelWeightsConfig
    interaction: ModelWeightsConfig
    interaction_subset_size: int
    linear_subset_size: int  # New parameter


@dataclass
class ExposureConfig:
    """Configuration for simulating exposure probability."""

    p_base: float


@dataclass
class OutcomeConfig:
    """Configuration for simulating a single outcome."""

    run_in_days: int
    p_base: float
    exposure_effect: float


@dataclass
class SimulationConfig:
    """Top-level configuration for the entire causal simulation."""

    paths: PathsConfig
    simulation_model: SimulationModelConfig
    exposure: ExposureConfig
    outcomes: Dict[str, OutcomeConfig]
    # trigger_codes is no longer needed here, will be derived from vocabulary
    index_date: pd.Timestamp = COMMON_INDEX_DATE
    unobserved_confounder: Optional[UnobservedConfounderConfig] = None
    seed: int = 42

    def __init__(self, config: dict):
        self.paths = PathsConfig(**config["paths"])
        # self.trigger_codes is removed

        self.simulation_model = SimulationModelConfig(
            linear=ModelWeightsConfig(**config["simulation_model"]["linear"]),
            interaction=ModelWeightsConfig(**config["simulation_model"]["interaction"]),
            interaction_subset_size=config["simulation_model"][
                "interaction_subset_size"
            ],
            linear_subset_size=config["simulation_model"]["linear_subset_size"],
        )

        self.exposure = ExposureConfig(**config["exposure"])

        self.outcomes = {
            outcome_key: OutcomeConfig(**outcome_data)
            for outcome_key, outcome_data in config.get("outcomes", {}).items()
        }

        if "unobserved_confounder" in config:
            self.unobserved_confounder = UnobservedConfounderConfig(
                **config["unobserved_confounder"]
            )
        else:
            self.unobserved_confounder = None

        self._validate_config()

    def _validate_config(self):
        """Validate that the configuration is consistent."""
        if self.exposure.p_base < 0 or self.exposure.p_base > 1:
            raise ValueError(
                f"Exposure p_base must be between 0 and 1, got {self.exposure.p_base}"
            )

        if self.simulation_model.interaction_subset_size <= 1:
            raise ValueError(
                f"interaction_subset_size must be greater than 1, got {self.simulation_model.interaction_subset_size}"
            )
        if self.simulation_model.linear_subset_size < 1:
            raise ValueError(
                f"linear_subset_size must be greater than 0, got {self.simulation_model.linear_subset_size}"
            )
        for outcome_key, outcome in self.outcomes.items():
            if outcome.p_base < 0 or outcome.p_base > 1:
                raise ValueError(
                    f"Outcome '{outcome_key}' p_base must be between 0 and 1, got {outcome.p_base}"
                )
