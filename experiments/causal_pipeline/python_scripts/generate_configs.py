#!/usr/bin/env python3
"""
Script to generate experiment-specific config files from base templates.
Replaces placeholders like {{EXPERIMENT_NAME}} with actual values.
"""

import yaml
from pathlib import Path
import argparse


def merge_dicts(base_dict, override_dict):
    """Recursively merge two dictionaries."""
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def replace_placeholders(config_dict, experiment_name, run_id="run_01"):
    """Recursively replace {{EXPERIMENT_NAME}} and {{RUN_ID}} placeholders in config."""
    if isinstance(config_dict, dict):
        return {
            key: replace_placeholders(value, experiment_name, run_id)
            for key, value in config_dict.items()
        }
    elif isinstance(config_dict, list):
        return [
            replace_placeholders(item, experiment_name, run_id) for item in config_dict
        ]
    elif isinstance(config_dict, str):
        result = config_dict.replace("{{EXPERIMENT_NAME}}", experiment_name)
        result = result.replace("{{RUN_ID}}", run_id)
        return result
    else:
        return config_dict


def generate_experiment_configs(experiment_name, script_dir, run_id="run_01"):
    """Generate all config files for a specific experiment."""

    # Define paths relative to script directory
    base_configs_dir = script_dir / "base_configs"
    experiment_configs_dir = script_dir / "experiment_configs"
    output_dir = script_dir / "generated_configs" / experiment_name

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment-specific config
    experiment_config_path = experiment_configs_dir / f"{experiment_name}.yaml"
    if not experiment_config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {experiment_config_path}"
        )

    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)

    # Config files to generate
    config_mappings = {
        # Shared data preparation
        "simulation.yaml": "simulation.yaml",
        "select_cohort.yaml": "select_cohort.yaml",
        "prepare_finetune.yaml": "prepare_finetune.yaml",
        # Baseline pipeline
        "train_baseline.yaml": "train_baseline.yaml",
        "calibrate.yaml": "calibrate.yaml",
        "estimate.yaml": "estimate.yaml",
        # BERT pipeline
        "finetune_bert.yaml": "finetune_bert.yaml",
        "calibrate_bert.yaml": "calibrate_bert.yaml",
        "estimate_bert.yaml": "estimate_bert.yaml",
    }

    for base_file, output_file in config_mappings.items():
        # Load base config
        base_config_path = base_configs_dir / base_file
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)

        # Merge with experiment-specific overrides (if any)
        if base_file == "simulation.yaml":
            # For simulation config, merge with experiment settings
            final_config = merge_dicts(base_config, experiment_config)
        else:
            # For other configs, just use base config
            final_config = base_config

        # Replace placeholders
        final_config = replace_placeholders(final_config, experiment_name, run_id)

        # Write output config
        output_path = output_dir / output_file
        with open(output_path, "w") as f:
            yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)

        print(f"Generated: {output_path}")

    print(f"All configs generated for experiment: {experiment_name}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configs")
    parser.add_argument("experiment_name", help="Name of the experiment")
    parser.add_argument(
        "--run_id", default="run_01", help="Run ID for output paths (default: run_01)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    generate_experiment_configs(args.experiment_name, script_dir, args.run_id)


if __name__ == "__main__":
    main()
