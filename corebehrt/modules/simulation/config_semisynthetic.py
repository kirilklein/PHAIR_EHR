from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathsConfig:
    """File paths for the semi-synthetic simulation."""

    data: str
    splits: List[str]
    outcomes: str


@dataclass
class CodePrefixConfig:
    """Maps concept types to their code prefixes in the MEDS data."""

    diagnosis: str = "D/"
    medication: str = "M/"
    procedure: str = "P/"
    admission: str = "ADM/"


@dataclass
class FeatureConfig:
    """Controls oracle feature extraction parameters."""

    code_prefixes: CodePrefixConfig = field(default_factory=CodePrefixConfig)
    lookback_days: int = 365
    recent_window_days: int = 90
    burst_window_days: int = 30
    motif_window_days: int = 30
    standardize: bool = True


@dataclass
class OutcomeModelConfig:
    """Outcome model: eta^(0) = beta_0 + f_B(r_B) + f_L(r_L)."""

    run_in_days: int = 1
    beta_0: float = -2.0
    baseline_coefficients: Dict[str, float] = field(default_factory=dict)
    longitudinal_coefficients: Dict[str, float] = field(default_factory=dict)
    interactions: List[Dict] = field(default_factory=list)
    noise_scale: float = 0.0


@dataclass
class TreatmentEffectConfig:
    """Treatment effect: constant (tau=delta) or heterogeneous (tau=delta_0 + g(r_B))."""

    mode: str = "constant"
    delta: float = 1.0
    delta_0: float = 0.5
    heterogeneous_coefficients: Dict[str, float] = field(default_factory=dict)


@dataclass
class SemiSyntheticOutcomeConfig:
    """Bundles outcome model and treatment effect for one outcome."""

    outcome_model: OutcomeModelConfig
    treatment_effect: TreatmentEffectConfig


@dataclass
class SemiSyntheticSimulationConfig:
    """Top-level configuration for the semi-synthetic simulation."""

    paths: PathsConfig
    features: FeatureConfig
    outcomes: Dict[str, SemiSyntheticOutcomeConfig]
    seed: int = 42
    debug: bool = False
    min_num_codes: int = 5
    exposure_code: str = "EXPOSURE"


def create_semisynthetic_config(cfg) -> SemiSyntheticSimulationConfig:
    """Parse a config object/dict into a SemiSyntheticSimulationConfig."""
    paths_config = PathsConfig(**cfg["paths"])

    prefix_cfg = CodePrefixConfig(**cfg.get("features", {}).get("code_prefixes", {}))
    feature_dict = dict(cfg.get("features", {}))
    feature_dict.pop("code_prefixes", None)
    feature_config = FeatureConfig(code_prefixes=prefix_cfg, **feature_dict)

    outcomes_config = {}
    for name, outcome_data in cfg["outcomes"].items():
        om_data = dict(outcome_data.get("outcome_model", {}))
        outcome_model = OutcomeModelConfig(**om_data)

        te_data = dict(outcome_data.get("treatment_effect", {}))
        treatment_effect = TreatmentEffectConfig(**te_data)

        outcomes_config[name] = SemiSyntheticOutcomeConfig(
            outcome_model=outcome_model,
            treatment_effect=treatment_effect,
        )

    return SemiSyntheticSimulationConfig(
        paths=paths_config,
        features=feature_config,
        outcomes=outcomes_config,
        seed=cfg.get("seed", 42),
        debug=cfg.get("debug", False),
        min_num_codes=cfg.get("min_num_codes", 5),
        exposure_code=cfg.get("exposure_code", "EXPOSURE"),
    )
