from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
class ExposureConfig:
    """Configuration for simulating a single outcome.

    The probability of an outcome is determined by a combination of a base
    probability, linear effects from historical codes, non-linear effects
    (quadratic and combination effects), and a direct effect from exposure.

    Attributes:
        p_base: The base probability of the outcome.
        trigger_codes: A list of medical codes that can influence the outcome probability.
        trigger_weights: A list of weights (logits) corresponding to each code in `trigger_codes`.
                         A code's presence adds its weight to the logit probability.
        quadratic_weights: An optional list of weights (logits) for non-linear effects.
                           These are applied in the same way as `trigger_weights`. If shorter
                           than `trigger_codes`, it will be padded with zeros. Defaults to None.
        combinations: An optional list of dictionaries to specify interaction effects.
                      Each dictionary should have two keys:
                      - `codes`: A list of code strings.
                      - `weight`: A float (logit) to be added if all `codes` are present
    """

    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]
    quadratic_weights: Optional[List[float]] = None
    combinations: Optional[List[Dict[str, Any]]] = None


@dataclass
class OutcomeConfig:
    """Configuration for simulating a single outcome.

    The probability of an outcome is determined by a combination of a base
    probability, linear effects from historical codes, non-linear effects
    (quadratic and combination effects), and a direct effect from exposure.

    Attributes:
        run_in_days: Time in days after the index date to assess for the outcome.
        p_base: The base probability of the outcome.
        trigger_codes: A list of medical codes that can influence the outcome probability.
        trigger_weights: A list of weights (logits) corresponding to each code in `trigger_codes`.
                         A code's presence adds its weight to the logit probability.
        exposure_effect: The direct causal effect (logit) of the exposure on the outcome.
        quadratic_weights: An optional list of weights (logits) for non-linear effects.
                           These are applied in the same way as `trigger_weights`. If shorter
                           than `trigger_codes`, it will be padded with zeros. Defaults to None.
        combinations: An optional list of dictionaries to specify interaction effects.
                      Each dictionary should have two keys:
                      - `codes`: A list of code strings.
                      - `weight`: A float (logit) to be added if all `codes` are present
                                in the patient's history.
                      Example: `[{"codes": ["D/CODE1", "D/CODE2"], "weight": 0.5}]`. Defaults to None.
    """

    run_in_days: int
    p_base: float
    trigger_codes: List[str]
    trigger_weights: List[float]
    exposure_effect: float
    quadratic_weights: Optional[List[float]] = None
    combinations: Optional[List[Dict[str, Any]]] = None


@dataclass
class SimulationConfig:
    """Top-level configuration for the entire causal simulation."""

    paths: PathsConfig
    exposure: ExposureConfig
    outcomes: Dict[str, OutcomeConfig]
    index_date: pd.Timestamp = COMMON_INDEX_DATE

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
