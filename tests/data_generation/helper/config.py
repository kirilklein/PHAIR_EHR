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
    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]
    run_in_days: int = 365
    compliance_interval_days: int = 30
    min_compliance_days: int = 365


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

    def get_all_trigger_codes(self) -> List[str]:
        """Get all trigger codes from exposure and outcomes."""
        all_codes = set(self.exposure.trigger_codes)
        for outcome in self.outcomes.values():
            all_codes.update(outcome.trigger_codes)
        return list(all_codes)

    def get_confounder_codes(self) -> Dict[str, List[str]]:
        """Get confounder codes for each outcome (codes that appear in both exposure and that outcome)."""
        confounders_by_outcome = {}
        exposure_codes = set(self.exposure.trigger_codes)

        for outcome_key, outcome_cfg in self.outcomes.items():
            outcome_codes = set(outcome_cfg.trigger_codes)
            confounders = exposure_codes.intersection(outcome_codes)
            confounders_by_outcome[outcome_key] = list(confounders)

        return confounders_by_outcome

    def get_all_confounder_codes(self) -> List[str]:
        """Get all unique confounder codes across all outcomes."""
        all_confounders = set()
        for confounders in self.get_confounder_codes().values():
            all_confounders.update(confounders)
        return list(all_confounders)
