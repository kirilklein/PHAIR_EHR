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
    std: float


@dataclass
class SimulationModelConfig:
    """Configuration for the simulation model."""

    linear: ModelWeightsConfig
    interaction: ModelWeightsConfig


@dataclass
class ExposureConfig:
    """Configuration for simulating exposure probability.

    The probability of exposure is determined by a combination of a base
    probability, linear effects from historical codes, and non-linear effects
    (interaction effects).

    Attributes:
        p_base: The base probability of exposure.
    """

    p_base: float


@dataclass
class OutcomeConfig:
    """Configuration for simulating a single outcome.

    The probability of an outcome is determined by a combination of a base
    probability, linear effects from historical codes, non-linear effects
    (interaction effects), and a direct effect from exposure.

    Attributes:
        run_in_days: Time in days after the index date to assess for the outcome.
        p_base: The base probability of the outcome.
        exposure_effect: The direct causal effect (logit) of the exposure on the outcome.
    """

    run_in_days: int
    p_base: float
    exposure_effect: float


@dataclass
class SimulationConfig:
    """Top-level configuration for the entire causal simulation."""

    paths: PathsConfig
    trigger_codes: List[str]
    simulation_model: SimulationModelConfig
    exposure: ExposureConfig
    outcomes: Dict[str, OutcomeConfig]
    index_date: pd.Timestamp = COMMON_INDEX_DATE
    unobserved_confounder: Optional[UnobservedConfounderConfig] = None

    def __init__(self, config: dict):
        self.paths = PathsConfig(**config["paths"])
        self.trigger_codes = config["trigger_codes"]

        self.simulation_model = SimulationModelConfig(
            linear=ModelWeightsConfig(**config["simulation_model"]["linear"]),
            interaction=ModelWeightsConfig(**config["simulation_model"]["interaction"]),
        )

        self.exposure = ExposureConfig(**config["exposure"])

        self.outcomes = {}
        for outcome_key, outcome_data in config.get("outcomes", {}).items():
            self.outcomes[outcome_key] = OutcomeConfig(**outcome_data)

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

        for outcome_key, outcome in self.outcomes.items():
            if outcome.p_base < 0 or outcome.p_base > 1:
                raise ValueError(
                    f"Outcome '{outcome_key}' p_base must be between 0 and 1, got {outcome.p_base}"
                )

    def get_all_trigger_codes(self) -> List[str]:
        """Get all trigger codes from exposure and outcomes."""
        return self.trigger_codes

    def get_confounder_codes(self) -> Dict[str, List[str]]:
        """DEPRECATED: All trigger codes are now potential confounders for all outcomes."""
        confounders_by_outcome = {}
        for outcome_key in self.outcomes:
            confounders_by_outcome[outcome_key] = self.trigger_codes
        return confounders_by_outcome

    def get_all_confounder_codes(self) -> List[str]:
        """Get all unique confounder codes across all outcomes."""
        return self.trigger_codes
