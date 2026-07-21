"""Aggregate nested semi-synthetic study estimates."""

from pathlib import Path

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import EffectColumns as E
from corebehrt.constants.causal.data import OUTCOME
from corebehrt.constants.causal.paths import (
    BOOTSTRAP_RESULTS_FILE,
    ESTIMATE_RESULTS_FILE,
)

MODEL_NAMES = ("bert", "baseline")
GROUP_COLUMNS = ["model", "method", OUTCOME]
REPLICATE_GROUP_COLUMNS = ["model", "run_id", "method", OUTCOME, E.effect_type]


def _tag_from_path(path: Path) -> dict:
    """Infer model and nested-resampling IDs from a result path."""
    parts = [part.lower() for part in path.parts]
    model = next((name for name in MODEL_NAMES if name in parts), "unknown")
    run_id = next((part for part in parts if part.startswith("run_")), "run_01")
    inner_id = next((part for part in parts if part.startswith("k_")), "k_01")
    return {"model": model, "run_id": run_id, "inner_id": inner_id}


def load_tagged_results(study_dir: Path, filename: str) -> pd.DataFrame:
    """Recursively load result files and add model/run/refit identifiers."""
    files = sorted(study_dir.rglob(filename))
    if not files:
        raise FileNotFoundError(f"No {filename} found under {study_dir}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        for key, value in _tag_from_path(path).items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_results(study_dir: Path) -> pd.DataFrame:
    return load_tagged_results(study_dir, ESTIMATE_RESULTS_FILE)


def load_bootstrap_results(study_dir: Path) -> pd.DataFrame:
    return load_tagged_results(study_dir, BOOTSTRAP_RESULTS_FILE)


def aggregate_replicates(results: pd.DataFrame) -> pd.DataFrame:
    """Combine K bootstrap-refit point estimates into one estimate per run."""
    results = results.dropna(subset=[E.true_effect, E.effect_type])
    rows = []

    for keys, group in results.groupby(REPLICATE_GROUP_COLUMNS):
        tags = dict(zip(REPLICATE_GROUP_COLUMNS, keys))
        effect_type = tags[E.effect_type]
        point = group[E.effect].mean()
        true_effect = group[E.true_effect].mean()

        if effect_type in {"RR", "RRT"}:
            uncertainty = _risk_ratio_uncertainty(group)
            point = uncertainty.pop("effect")
        else:
            uncertainty = _difference_uncertainty(point, group[E.effect])

        lower = uncertainty[E.CI95_lower]
        upper = uncertainty[E.CI95_upper]
        covered = (
            lower <= true_effect <= upper
            if np.isfinite(lower) and np.isfinite(upper)
            else np.nan
        )

        rows.append(
            {
                **tags,
                "n_refits": group["inner_id"].nunique(),
                E.true_effect: true_effect,
                E.effect: point,
                **uncertainty,
                "covered": covered,
            }
        )

    return pd.DataFrame(rows).sort_values([OUTCOME, "model", "method", "run_id"])


def _difference_uncertainty(point: float, samples: pd.Series) -> dict:
    standard_error = samples.std(ddof=1)
    margin = 1.96 * standard_error
    return {
        E.std_err: standard_error,
        "std_err_log": np.nan,
        E.CI95_lower: point - margin,
        E.CI95_upper: point + margin,
    }


def _risk_ratio_uncertainty(estimates: pd.DataFrame) -> dict:
    p1 = estimates[E.effect_1].mean()
    p0 = estimates[E.effect_0].mean()
    if not 0 < p1 < 1 or not 0 < p0 < 1:
        raise ValueError(f"Risk-ratio probabilities must lie in (0, 1): {p1=}, {p0=}")

    sample_p1 = estimates[E.effect_1]
    sample_p0 = estimates[E.effect_0]
    if (
        not sample_p1.between(0, 1, inclusive="neither").all()
        or not sample_p0.between(0, 1, inclusive="neither").all()
    ):
        raise ValueError("Bootstrap risk-ratio probabilities must lie in (0, 1)")

    eta_1 = np.log(sample_p1 / (1 - sample_p1))
    eta_0 = np.log(sample_p0 / (1 - sample_p0))
    variance_log_rr = (1 - p1) ** 2 * eta_1.var(ddof=1) + (1 - p0) ** 2 * eta_0.var(
        ddof=1
    )
    std_err_log = np.sqrt(variance_log_rr)
    risk_ratio = p1 / p0
    margin = 1.96 * std_err_log
    return {
        E.effect: risk_ratio,
        E.std_err: risk_ratio * std_err_log,
        "std_err_log": std_err_log,
        E.CI95_lower: np.exp(np.log(risk_ratio) - margin),
        E.CI95_upper: np.exp(np.log(risk_ratio) + margin),
    }


def summarize_performance(replicates: pd.DataFrame) -> pd.DataFrame:
    """Summarize bias, empirical SD, SE calibration, and coverage across runs."""
    rows = []
    for keys, group in replicates.groupby(GROUP_COLUMNS + ["effect_type"]):
        tags = dict(zip(GROUP_COLUMNS + ["effect_type"], keys))
        empirical_sd = group[E.effect].std(ddof=1) if len(group) > 1 else np.nan
        mean_se = group[E.std_err].mean()
        rows.append(
            {
                **tags,
                "n": len(group),
                E.true_effect: group[E.true_effect].mean(),
                "mean_effect": group[E.effect].mean(),
                "bias": (group[E.effect] - group[E.true_effect]).mean(),
                "sd_emp": empirical_sd,
                "mean_se": mean_se,
                "se_calibration": (
                    empirical_sd / mean_se if len(group) > 1 and mean_se > 0 else np.nan
                ),
                "coverage": group["covered"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values([OUTCOME, "model", "method"])
