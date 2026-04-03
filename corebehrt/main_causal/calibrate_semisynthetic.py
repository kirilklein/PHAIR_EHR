"""Calibration diagnostics for the semi-synthetic simulation.

Runs the feature extraction and probability computation pipeline
without Bernoulli sampling, then prints a calibration report and
saves diagnostic plots.
"""

import logging
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corebehrt.functional.setup.args import get_args
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.simulation.config_semisynthetic import (
    create_semisynthetic_config,
)
from corebehrt.modules.simulation.plot import plot_probability_distributions
from corebehrt.modules.simulation.semisynthetic_simulator import (
    SemiSyntheticCausalSimulator,
)

logger = logging.getLogger("calibrate")

CONFIG_PATH = "./corebehrt/configs/causal/simulate_semisynthetic.yaml"


def main_calibrate(config_path):
    cfg = load_config(config_path)
    CausalDirectoryPreparer(cfg).setup_simulate_from_sequence()

    shard_loader = ShardLoader(cfg.paths.data, cfg.paths.splits)
    sim_config = create_semisynthetic_config(cfg)
    simulator = SemiSyntheticCausalSimulator(sim_config)

    # Accumulate across shards
    all_features = []
    all_is_exposed = []
    all_probas = {}  # outcome_name -> {"P0": [], "P1": []}
    all_tau = {}  # outcome_name -> []

    for shard, _ in shard_loader():
        result = simulator.extract_features_and_probabilities(shard)
        if result is None:
            continue
        features_df, _, is_exposed, probas, tau = result
        all_features.append(features_df.assign(is_exposed=is_exposed))
        all_is_exposed.append(is_exposed)
        for outcome_name in sim_config.outcomes:
            all_probas.setdefault(outcome_name, {"P0": [], "P1": []})
            all_probas[outcome_name]["P0"].append(probas[outcome_name]["P0"])
            all_probas[outcome_name]["P1"].append(probas[outcome_name]["P1"])
            all_tau.setdefault(outcome_name, []).append(tau[outcome_name])

    if not all_features:
        logger.error("No patients found across shards.")
        return

    features_combined = pd.concat(all_features, ignore_index=True)
    is_exposed_combined = np.concatenate(all_is_exposed)
    output_dir = sim_config.paths.outcomes
    figs_dir = join(output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # --- Feature diagnostics ---
    _print_feature_diagnostics(features_combined, is_exposed_combined)

    # --- Probability and causal effect diagnostics ---
    probas_for_plot = {}
    for outcome_name in sim_config.outcomes:
        p0 = np.concatenate(all_probas[outcome_name]["P0"])
        p1 = np.concatenate(all_probas[outcome_name]["P1"])
        tau_arr = np.concatenate(all_tau[outcome_name])
        probas_for_plot[outcome_name] = {"P0": p0, "P1": p1}

        _print_probability_diagnostics(outcome_name, p0, p1, is_exposed_combined)
        _print_causal_diagnostics(outcome_name, p0, p1, is_exposed_combined)
        _plot_ite_histogram(tau_arr, outcome_name, figs_dir)

    plot_probability_distributions(probas_for_plot, figs_dir)

    # --- SMD love plot ---
    feature_cols = [c for c in features_combined.columns if c != "is_exposed"]
    _plot_smd_love_plot(features_combined, feature_cols, is_exposed_combined, figs_dir)

    logger.info("Calibration complete. Plots saved to %s", figs_dir)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_feature_diagnostics(features_df: pd.DataFrame, is_exposed: np.ndarray):
    feature_cols = [c for c in features_df.columns if c != "is_exposed"]
    print("\n" + "=" * 80)
    print("FEATURE DIAGNOSTICS")
    print("=" * 80)

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    header = f"{'Feature':<30} {'mean':>8} {'std':>8} {'min':>8}"
    for q in quantiles:
        header += f" {'p' + str(int(q * 100)):>6}"
    header += f" {'max':>8} {'SMD':>8}"
    print(header)
    print("-" * len(header))

    for col in feature_cols:
        vals = features_df[col].values
        q_vals = np.quantile(vals, quantiles)
        smd = _compute_smd(vals[is_exposed], vals[~is_exposed])
        row = (
            f"{col:<30} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} {np.min(vals):>8.3f}"
        )
        for qv in q_vals:
            row += f" {qv:>6.3f}"
        row += f" {np.max(vals):>8.3f} {smd:>8.3f}"
        print(row)
    print()


def _print_probability_diagnostics(
    outcome_name: str, p0: np.ndarray, p1: np.ndarray, is_exposed: np.ndarray
):
    print(f"\n--- Probability diagnostics: {outcome_name} ---")
    for label, probs in [("P(Y(0))", p0), ("P(Y(1))", p1)]:
        print(
            f"  {label}: mean={np.mean(probs):.4f}, std={np.std(probs):.4f}, "
            f"min={np.min(probs):.4f}, max={np.max(probs):.4f}, "
            f"median={np.median(probs):.4f}"
        )
    frac_extreme_p0 = np.mean((p0 < 0.01) | (p0 > 0.80))
    frac_extreme_p1 = np.mean((p1 < 0.01) | (p1 > 0.80))
    print(
        f"  Extreme probabilities (<0.01 or >0.80): P0={frac_extreme_p0:.3f}, P1={frac_extreme_p1:.3f}"
    )

    # Expected prevalence under factual assignment
    factual_prob = np.where(is_exposed, p1, p0)
    print(f"  Expected factual prevalence: {np.mean(factual_prob):.4f}")


def _print_causal_diagnostics(
    outcome_name: str, p0: np.ndarray, p1: np.ndarray, is_exposed: np.ndarray
):
    ite = p1 - p0
    ate = np.mean(ite)
    att = np.mean(ite[is_exposed]) if np.any(is_exposed) else float("nan")
    atc = np.mean(ite[~is_exposed]) if np.any(~is_exposed) else float("nan")
    rr = np.mean(p1) / np.mean(p0) if np.mean(p0) > 0 else float("nan")

    print(f"\n--- Causal effect diagnostics: {outcome_name} ---")
    print(f"  True ATE  = {ate:.4f}")
    print(f"  True ATT  = {att:.4f}")
    print(f"  True ATC  = {atc:.4f}")
    print(f"  True RR   = {rr:.4f}")


def _compute_smd(treated: np.ndarray, control: np.ndarray) -> float:
    """Standardized mean difference."""
    pooled_std = np.sqrt((np.var(treated) + np.var(control)) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(treated) - np.mean(control)) / pooled_std


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_ite_histogram(tau: np.ndarray, outcome_name: str, figs_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(tau, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(f"ITE Distribution: {outcome_name}")
    ax.set_xlabel("Individual Treatment Effect (logit scale)")
    ax.set_ylabel("Count")
    ax.axvline(
        np.mean(tau), color="red", linestyle="--", label=f"Mean={np.mean(tau):.3f}"
    )
    ax.legend()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    save_figure_with_azure_copy(
        fig, join(figs_dir, f"ite_histogram_{outcome_name}.png")
    )


def _plot_smd_love_plot(
    features_df: pd.DataFrame,
    feature_cols,
    is_exposed: np.ndarray,
    figs_dir: str,
):
    smds = []
    for col in feature_cols:
        vals = features_df[col].values
        smds.append(_compute_smd(vals[is_exposed], vals[~is_exposed]))

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4)))
    y_pos = np.arange(len(feature_cols))
    ax.barh(y_pos, smds, color="#3498db", edgecolor="black", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Feature Balance (Treated vs Control)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.1, color="red", linestyle="--", alpha=0.5, label="SMD=0.1")
    ax.axvline(-0.1, color="red", linestyle="--", alpha=0.5)
    ax.legend()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    save_figure_with_azure_copy(fig, join(figs_dir, "smd_love_plot.png"))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_calibrate(args.config_path)
