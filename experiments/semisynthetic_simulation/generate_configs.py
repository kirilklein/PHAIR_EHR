#!/usr/bin/env python3
"""Generate per-run config files for multi-replicate semisynthetic simulation.

For each run, produces a YAML config with a unique seed and output directory.
Seed = base_seed + run_number (e.g., base_seed=42, run_01 -> seed=43).

Usage:
    python generate_configs.py my_scenario --n_runs 50 --base_seed 42
"""

import argparse
from pathlib import Path

import yaml


def generate_configs(
    scenario_name,
    base_config_path,
    n_runs,
    base_seed=42,
    experiments_dir="./outputs/causal/semisynthetic_study/runs",
    meds_data=None,
):
    """Generate one config file per run from a base config."""
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    output_dir = Path("generated_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_number in range(1, n_runs + 1):
        run_id = f"run_{run_number:02d}"
        seed = base_seed + run_number

        config = _deep_copy_config(base_config)
        config["seed"] = seed
        config["paths"]["outcomes"] = f"{experiments_dir}/{run_id}/{scenario_name}"
        if meds_data is not None:
            config["paths"]["data"] = meds_data

        output_path = output_dir / f"{scenario_name}_{run_id}.yaml"
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Generated: {output_path} (seed={seed})")

    print(f"\n{n_runs} configs generated for scenario: {scenario_name}")


def _deep_copy_config(config):
    """Deep copy a config dict (handles nested dicts and lists)."""
    if isinstance(config, dict):
        return {k: _deep_copy_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_deep_copy_config(item) for item in config]
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-run configs for semisynthetic simulation"
    )
    parser.add_argument("scenario_name", help="Scenario identifier")
    parser.add_argument(
        "--base_config",
        default="corebehrt/configs/causal/simulate_semisynthetic.yaml",
        help="Path to base config (default: simulate_semisynthetic.yaml)",
    )
    parser.add_argument(
        "--n_runs", type=int, default=50, help="Number of runs (default: 50)"
    )
    parser.add_argument(
        "--base_seed", type=int, default=42, help="Base seed (default: 42)"
    )
    parser.add_argument(
        "--experiments_dir",
        default="./outputs/causal/semisynthetic_study/runs",
        help="Base output directory for runs",
    )
    parser.add_argument("--meds", default=None, help="Override MEDS data path")
    args = parser.parse_args()

    generate_configs(
        args.scenario_name,
        args.base_config,
        args.n_runs,
        args.base_seed,
        args.experiments_dir,
        args.meds,
    )


if __name__ == "__main__":
    main()
