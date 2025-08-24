from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class PathsConfig:
    """
    Configuration for file paths used in the simulation.

    Attributes:
        data: Path to the input directory containing real patient sequence data (MEDS format)
        splits: List of data splits to use (e.g., ["train", "tuning", "held_out"])
        outcomes: Path to the output directory where simulation results will be saved
    """

    data: str
    splits: List[str]
    outcomes: str


@dataclass
class ModelWeightsConfig:
    """
    Configuration for sampling model weights that map medical codes to latent factors.

    This controls how individual medical codes in a patient's history contribute to
    the underlying latent health factors that drive exposure and outcome probabilities.

    Attributes:
        mean: Mean of the normal distribution used to sample code-to-factor weights
        scale: Standard deviation of the normal distribution for sampling weights.
               Smaller values mean individual codes have smaller effects on factors.
        sparsity_factor: Probability that a code-factor weight is set to zero (0.0-1.0).
                        Higher values create sparser connections, meaning each code
                        only influences a few factors rather than all factors.
    """

    mean: float
    scale: float
    sparsity_factor: float = 0.0


@dataclass
class InfluenceScalesConfig:
    """
    Configuration for scaling the influence of each latent factor group.

    This controls how much each latent factor group contributes to the overall
    exposure and outcome probabilities.

    Attributes:
        shared_to_exposure: Scaling factor for the shared latent factors' influence on exposure
        shared_to_outcome: Scaling factor for the shared latent factors' influence on outcomes
        exposure_only_to_exposure: Scaling factor for the exposure-only latent factors' influence on exposure
        outcome_only_to_outcome: Scaling factor for the outcome-only latent factors' influence on outcomes
    """

    shared_to_exposure: float = 1.0
    shared_to_outcome: float = 1.0
    exposure_only_to_exposure: float = 1.0
    outcome_only_to_outcome: float = 1.0


@dataclass
class ExposureConfig:
    """
    Configuration for simulating exposure probability.

    Attributes:
        p_base: Base probability of exposure (before any effects)
        age_effect: Optional effect of age on exposure probability (if None, no age effect)
    """

    p_base: float
    age_effect: Optional[float] = None


@dataclass
class OutcomeConfig:
    """
    Configuration for simulating a single outcome.

    Attributes:
        run_in_days: Number of days between the index date and the first day of the outcome
        p_base: Base probability of the outcome (before any effects)
        exposure_effect: Effect of exposure on the outcome probability
        age_effect: Optional effect of age on the outcome probability (if None, no age effect)
    """

    run_in_days: int
    p_base: float
    exposure_effect: float
    age_effect: Optional[float] = None


@dataclass
class UnobservedConfounderConfig:
    """
    Configuration for an optional unobserved confounder.

    Attributes:
        p_occurrence: Probability of the confounder occurring in a patient
        exposure_effect: Effect of the confounder on exposure probability
        outcome_effects: Dictionary mapping outcome names to their effects on that outcome
    """

    p_occurrence: float
    exposure_effect: float
    outcome_effects: Dict[str, float]


@dataclass
class RealisticSimulationModelConfig:
    """
    Configuration for the realistic simulation model based on latent factors.

    Attributes:
        num_shared_factors: Number of shared latent factors
        num_exposure_only_factors: Number of exposure-only latent factors
        num_outcome_only_factors: Number of outcome-only latent factors
        factor_mapping: Configuration for how medical codes map to latent factors
        influence_scales: Configuration for scaling the influence of each latent factor group
        time_decay_halflife_days: Optional half-life for time-dependent effects (if None, no time decay)
    """

    num_shared_factors: int
    num_exposure_only_factors: int
    num_outcome_only_factors: int
    factor_mapping: ModelWeightsConfig
    influence_scales: InfluenceScalesConfig = field(
        default_factory=InfluenceScalesConfig
    )
    time_decay_halflife_days: Optional[float] = 365


@dataclass
class SimulationConfig:
    """Top-level configuration for the entire causal simulation."""

    paths: PathsConfig
    simulation_model: RealisticSimulationModelConfig
    exposure: ExposureConfig
    outcomes: Dict[str, OutcomeConfig]
    index_date_str: str = "2010-01-01"
    unobserved_confounder: Optional[UnobservedConfounderConfig] = None
    include_code_prefixes: Optional[List[str]] = None
    seed: int = 42
    debug: bool = False
    min_num_codes: int = 5
    index_date: pd.Timestamp = field(init=False)

    def __post_init__(self):
        """Convert string date to timestamp after initialization."""
        self.index_date = pd.Timestamp(self.index_date_str)


def create_simulation_config(cfg: dict) -> SimulationConfig:
    """
    Parses a raw config dictionary and constructs the nested SimulationConfig dataclass.

    This acts as a factory function to abstract away the details of config parsing,
    providing a single, validated configuration object.

    Args:
        cfg: A dictionary loaded from a YAML configuration file.

    Returns:
        A fully constructed and validated SimulationConfig object.
    """
    # 1. Parse nested dictionary sections into their respective dataclasses
    paths_config = PathsConfig(**cfg["paths"])
    model_weights_config = ModelWeightsConfig(
        **cfg["simulation_model"]["factor_mapping"]
    )
    influence_scales_config = InfluenceScalesConfig(
        **cfg["simulation_model"].get("influence_scales", {})
    )

    sim_model_config = RealisticSimulationModelConfig(
        num_shared_factors=cfg["simulation_model"]["num_shared_factors"],
        num_exposure_only_factors=cfg["simulation_model"]["num_exposure_only_factors"],
        num_outcome_only_factors=cfg["simulation_model"]["num_outcome_only_factors"],
        factor_mapping=model_weights_config,
        influence_scales=influence_scales_config,
        time_decay_halflife_days=cfg["simulation_model"]["time_decay_halflife_days"],
    )

    exposure_config = ExposureConfig(**cfg["exposure"])
    outcomes_config = {
        name: OutcomeConfig(**data) for name, data in cfg["outcomes"].items()
    }

    # 2. Handle optional sections gracefully
    unobserved_confounder_config = None
    if "unobserved_confounder" in cfg and cfg["unobserved_confounder"] is not None:
        unobserved_confounder_config = UnobservedConfounderConfig(
            **cfg["unobserved_confounder"]
        )

    # 3. Construct the top-level SimulationConfig object
    # Use .get() for optional top-level keys for robustness
    simulation_config = SimulationConfig(
        paths=paths_config,
        simulation_model=sim_model_config,
        exposure=exposure_config,
        outcomes=outcomes_config,
        index_date_str=cfg["index_date_str"],
        unobserved_confounder=unobserved_confounder_config,
        include_code_prefixes=cfg.get("include_code_prefixes"),
        seed=cfg.get("seed", 42),
        debug=cfg.get("debug", False),
        min_num_codes=cfg.get("min_num_codes", 5),
    )

    return simulation_config
