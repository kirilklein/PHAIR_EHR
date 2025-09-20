#!/usr/bin/env python3
"""
Script to create a new experiment configuration template.
"""

import yaml
import argparse
from pathlib import Path


def create_experiment_template(experiment_name, script_dir):
    """Create a new experiment config template."""
    
    experiment_configs_dir = script_dir / "experiment_configs"
    experiment_configs_dir.mkdir(exist_ok=True)
    
    output_path = experiment_configs_dir / f"{experiment_name}.yaml"
    
    if output_path.exists():
        response = input(f"Experiment '{experiment_name}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Template configuration
    template_config = {
        'experiment_name': experiment_name,
        'description': f"Experiment: {experiment_name}",
        'simulation_model': {
            'num_shared_factors': 10,
            'num_exposure_only_factors': 10,
            'num_outcome_only_factors': 10,
            'factor_mapping': {
                'mean': 0.0,
                'scale': 1.4,
                'sparsity_factor': 0.8
            },
            'influence_scales': {
                'shared_to_exposure': 5.0,
                'shared_to_outcome': 5.0,
                'exposure_only_to_exposure': 0.4,
                'outcome_only_to_outcome': 0.4
            }
        },
        'outcomes': {
            'OUTCOME': {
                'run_in_days': 1,
                'p_base': 0.2,
                'exposure_effect': 3.0,
                'age_effect': -0.005
            },
            'OUTCOME_NULL': {
                'run_in_days': 1,
                'p_base': 0.2,
                'exposure_effect': 0.0
            }
        }
    }
    
    with open(output_path, 'w') as f:
        f.write(f"# {experiment_name.replace('_', ' ').title()} Experiment\n")
        f.write(f"# TODO: Describe your experiment here\n\n")
        yaml.dump(template_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created experiment template: {output_path}")
    print(f"Edit this file to customize your experiment settings.")
    print(f"Then run: run_experiment.bat {experiment_name}")


def main():
    parser = argparse.ArgumentParser(description="Create a new experiment template")
    parser.add_argument("experiment_name", help="Name of the experiment (use underscores, no spaces)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    create_experiment_template(args.experiment_name, script_dir)


if __name__ == "__main__":
    main()

