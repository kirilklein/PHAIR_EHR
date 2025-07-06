from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    source_dir: str
    write_dir: str
    splits: List[str] = field(default_factory=lambda: ["train", "tuning", "held_out"])


@dataclass
class ExposureConfig:
    """Configuration for exposure simulation."""

    code: str
    run_in_days: int
    compliance_interval_days: int
    daily_stop_prob: float
    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]


@dataclass
class OutcomeConfig:
    """Configuration for a single outcome simulation."""

    code: str
    run_in_days: int
    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]
    exposure_effect: float


@dataclass
class SimulationConfig:
    """Top-level configuration for the entire causal simulation."""

    paths: PathsConfig
    exposure: ExposureConfig
    outcomes: Dict[str, OutcomeConfig]

    def __init__(self, config: dict):
        # Parse paths
        self.paths = PathsConfig(**config["paths"])

        # Parse exposure
        self.exposure = ExposureConfig(**config["exposure"])

        # Parse outcomes
        self.outcomes = {}
        for outcome_key, outcome_data in config.get("outcomes", {}).items():
            self.outcomes[outcome_key] = OutcomeConfig(**outcome_data)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate that the configuration is consistent."""
        # Validate exposure section
        if len(self.exposure.trigger_codes) != len(self.exposure.trigger_weights):
            raise ValueError(
                f"Exposure section: trigger_codes length ({len(self.exposure.trigger_codes)}) "
                f"must match trigger_weights length ({len(self.exposure.trigger_weights)})"
            )

        # Validate each outcome section
        for outcome_key, outcome in self.outcomes.items():
            if len(outcome.trigger_codes) != len(outcome.trigger_weights):
                raise ValueError(
                    f"Outcome '{outcome_key}': trigger_codes length ({len(outcome.trigger_codes)}) "
                    f"must match trigger_weights length ({len(outcome.trigger_weights)})"
                )

        # Validate required fields are present and reasonable
        if self.exposure.p_base < 0 or self.exposure.p_base > 1:
            raise ValueError(
                f"Exposure p_base must be between 0 and 1, got {self.exposure.p_base}"
            )

        for outcome_key, outcome in self.outcomes.items():
            if outcome.p_base < 0 or outcome.p_base > 1:
                raise ValueError(
                    f"Outcome '{outcome_key}' p_base must be between 0 and 1, got {outcome.p_base}"
                )
