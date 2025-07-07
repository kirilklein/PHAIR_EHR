"""
Enhanced causal effect simulation module for EHR data.

USAGE:
    python induce_causal_effect.py --config /path/to/config.yaml

DESCRIPTION:
    This script simulates binary exposure and multiple outcome events in EHR data based on causal relationships.
    It uses a simplified configuration structure where:

    1. Exposure section defines trigger codes that influence exposure probability
    2. Multiple outcomes sections define trigger codes that influence outcome probabilities
    3. Causal relationships are implicit based on code placement:
       - Codes in both exposure and outcome sections act as confounders
       - Codes only in exposure section act as instruments
       - Codes only in outcome sections act as prognostic factors

    The simulation uses a logistic model: P(event) = expit(logit(p_base) + Σ(effect_i * trigger_i))

CONFIGURATION STRUCTURE:
    The YAML config file should contain:
    - paths: source_dir and write_dir for input/output data
    - exposure: code, probabilities, and trigger_codes with trigger_weights
    - outcomes: multiple outcomes, each with code, probabilities, trigger_codes, trigger_weights, and exposure_effect

EXAMPLE USAGE:
    # Using a configuration file (recommended)
    python induce_causal_effect.py --config ./configs/induce_causal_effect.yaml

PARAMETER GUIDANCE:
    - Effect sizes (trigger_weights): Typical range [-3.0, 3.0]. Positive = increases probability, negative = decreases
    - Base probabilities (p_base): Should reflect realistic event rates (0.01-0.4 often reasonable)
    - Exposure effects: Direct causal effect of exposure on each outcome (0.0 = no effect, 2.0 = strong effect)

OUTPUT:
    - Parquet files with original data + simulated EXPOSURE and OUTCOME events
    - .ite.csv file containing Individual Treatment Effects for each outcome (columns: ite_{outcome_code})
    - .ate.txt file containing Average Treatment Effects for each outcome
    - .ate.json file with ATE results in JSON format
    - config.yaml file with the exact configuration used
    - simulation_parameters.json file with all parameters for reproducibility

NOTES:
    - Input directory must contain .parquet shard files
    - Each shard should have columns: subject_id, time, code, numeric_value
    - Simulated events are added with temporal ordering preserved
    - Multiple outcomes are supported, each with their own ITE column
"""

import argparse
import os

import yaml

from tests.data_generation.helper.analytics import SimulationReporter
from tests.data_generation.helper.config import SimulationConfig
from tests.data_generation.helper.induce_causal_effect import CausalSimulator
from tests.data_generation.helper.io import DataManager, save_ite_data


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser with comprehensive help."""
    parser = argparse.ArgumentParser(
        description="""
        Generate enhanced simulated causal data for EHR analysis.
        
        This script adds simulated EXPOSURE and OUTCOME(s) events to existing EHR data
        based on a realistic causal structure with confounding relationships.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration file option
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file.",
    )
    return parser


def parse_codes_and_weights(
    codes_str: str, weights_str: str
) -> tuple[list[str], list[float]]:
    """
    Parse comma-separated codes and weights strings.

    Args:
        codes_str: Comma-separated string of codes
        weights_str: Comma-separated string of weights

    Returns:
        Tuple of (codes_list, weights_list)

    Raises:
        ValueError: If codes and weights have different lengths
    """
    codes = [code.strip() for code in codes_str.split(",")]
    weights = [float(w.strip()) for w in weights_str.split(",")]

    if len(codes) != len(weights):
        raise ValueError(
            f"Number of codes ({len(codes)}) must match number of weights ({len(weights)})"
        )

    return codes, weights


def main() -> None:
    """Main function to run the enhanced causal simulation."""
    args = create_parser().parse_args()

    config = load_config(args.config)
    simulation_config = SimulationConfig(
        config=config,
    )
    write_dir = simulation_config.paths.write_dir
    os.makedirs(write_dir, exist_ok=True)
    with open(os.path.join(write_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Initialize simulator with all parameters
    simulator = CausalSimulator(simulation_config)

    for split in simulation_config.paths.splits:
        print(f"Processing split: {split}")
        os.makedirs(os.path.join(write_dir, split), exist_ok=True)
        df, shards = DataManager.load_shards(
            os.path.join(simulation_config.paths.source_dir, split)
        )
        reporter = SimulationReporter()
        reporter.print_trigger_stats(df, simulation_config)

        simulated_df, ite_df = simulator.simulate_dataset(df)

        reporter.print_simulation_results(simulated_df, simulation_config)
        split_write_dir = os.path.join(write_dir, split)
        os.makedirs(split_write_dir, exist_ok=True)

        simulated_df.reset_index(drop=True, inplace=True)
        DataManager.write_shards(simulated_df, split_write_dir, shards)
        reporter.save_report(os.path.join(split_write_dir, ".report.txt"))
        save_ite_data(ite_df, simulation_config, split_write_dir)


if __name__ == "__main__":
    main()
