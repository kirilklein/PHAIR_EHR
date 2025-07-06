"""
Enhanced causal effect simulation module for EHR data.

USAGE:
    python induce_causal_effect.py --source_dir /path/to/input --write_dir /path/to/output [OPTIONS]

DESCRIPTION:
    This script simulates binary exposure and outcome events in EHR data based on causal relationships.
    It creates a realistic causal structure where:
    
    1. Trigger codes in the original data influence exposure probability
    2. Exposure events influence outcome probability  
    3. Some trigger codes act as confounders (affecting both exposure and outcome)
    
    The simulation uses a logistic model: P(event) = expit(logit(p_base) + Σ(effect_i * trigger_i))

CAUSAL STRUCTURE:
    - Confounder codes: Affect both exposure and outcome (creates confounding bias)
    - Instrument codes: Only affect exposure probability (like instrumental variables)
    - Prognostic codes: Only affect outcome probability (prognostic factors)
    - Exposure events: Directly affect outcome probability (the causal effect of interest)

EXAMPLE USAGE:
    # Basic usage with default parameters
    python induce_causal_effect.py \
        --source_dir ./data/input_shards \
        --write_dir ./data/simulated_shards

    # Custom causal effects with multiple codes
    python induce_causal_effect.py \
        --source_dir ./data/input_shards \
        --write_dir ./data/simulated_shards \
        --confounder_codes "DDZ32,LAB8" \
        --confounder_exposure_weights "1.2,0.8" \
        --confounder_outcome_weights "-0.3,0.5" \
        --instrument_codes "MME01,DRUG_A" \
        --instrument_weights "0.8,1.1"

PARAMETER GUIDANCE:
    - Effect sizes: Typical range [-2.0, 2.0]. Positive = increases probability, negative = decreases
    - Base probabilities: Should reflect realistic event rates (0.1-0.4 often reasonable)
    - Daily probabilities: Should be much smaller (0.001-0.01) as they represent daily risk
    
OUTPUT:
    - Parquet files with original data + simulated EXPOSURE and OUTCOME events
    - .ate.txt file containing the Average Treatment Effect for validation
    - simulation_parameters.json file with all parameters for reproducibility
    - reproduce_command.txt file with exact command line for easy copy-pasting

NOTES:
    - Input directory must contain .parquet shard files
    - Each shard should have columns: subject_id, time, code, numeric_value
    - Simulated events are added with temporal ordering preserved
"""

import os
import argparse
import json
import sys
from datetime import datetime

from tests.data_generation.helper.induce_causal_effect import (
    CausalSimulator,
    DataManager,
    SimulationReporter,
)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser with comprehensive help."""
    parser = argparse.ArgumentParser(
        description="""
        Generate enhanced simulated causal data for EHR analysis.
        
        This script adds simulated EXPOSURE and OUTCOME events to existing EHR data
        based on a realistic causal structure with confounding relationships.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python %(prog)s --source_dir ./input --write_dir ./output
  
  # Custom parameters with multiple codes 
  python %(prog)s --source_dir ./input --write_dir ./output \\
    --confounder_codes "DDZ32,LAB8" \\
    --confounder_exposure_weights "1.2,0.8" \\
    --confounder_outcome_weights "-0.3,0.5" \\
    --instrument_codes "MME01,DRUG_A" --instrument_weights "0.8,1.1"
        """,
    )

    # Required arguments
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Directory containing source data shards (.parquet files)",
    )
    parser.add_argument(
        "--write_dir",
        required=True,
        help="Directory to write output shards (will be created if needed)",
    )

    # Trigger codes
    trigger_group = parser.add_argument_group(
        "Trigger Codes", "Medical codes that influence event probabilities"
    )
    trigger_group.add_argument(
        "--confounder_codes",
        default="DDZ32",
        help="Comma-separated codes that affect both exposure and outcome. Default: DDZ32",
    )
    trigger_group.add_argument(
        "--confounder_exposure_weights",
        default="1.2",
        help="Comma-separated weights for confounder effects on exposure. Default: 1.2",
    )
    trigger_group.add_argument(
        "--confounder_outcome_weights",
        default="-0.3",
        help="Comma-separated weights for confounder effects on outcome. Default: -0.3",
    )
    trigger_group.add_argument(
        "--instrument_codes",
        default="MME01",
        help="Comma-separated codes that only affect exposure (instrumental variables). Default: MME01",
    )
    trigger_group.add_argument(
        "--instrument_weights",
        default="0.8",
        help="Comma-separated weights for instrument codes (exposure effects). Default: 0.8",
    )
    trigger_group.add_argument(
        "--prognostic_codes",
        default="DE11",
        help="Comma-separated codes that only affect outcome (prognostic factors). Default: DE11",
    )
    trigger_group.add_argument(
        "--prognostic_weights",
        default="0.5",
        help="Comma-separated weights for prognostic codes (outcome effects). Default: 0.5",
    )

    # Base probabilities
    prob_group = parser.add_argument_group(
        "Base Probabilities", "Baseline event rates without any triggers"
    )
    prob_group.add_argument(
        "--p_base_exposure",
        type=float,
        default=0.2,
        help="Base probability for exposure events (0-1). Default: 0.2",
    )
    prob_group.add_argument(
        "--p_base_outcome",
        type=float,
        default=0.2,
        help="Base probability for outcome events (0-1). Default: 0.2",
    )
    prob_group.add_argument(
        "--p_daily_base_exposure",
        type=float,
        default=0.005,
        help="Base daily probability for exposure events (0-1). Default: 0.005",
    )
    prob_group.add_argument(
        "--p_daily_base_outcome",
        type=float,
        default=0.003,
        help="Base daily probability for outcome events (0-1). Default: 0.003",
    )

    # Effect sizes
    effects_group = parser.add_argument_group(
        "Effect Sizes", "Logistic regression coefficients for causal relationships"
    )
    effects_group.add_argument(
        "--exposure_outcome_effect",
        type=float,
        default=2.0,
        help="Causal effect of exposure on outcome (main effect of interest). Default: 2.0",
    )

    # Other parameters
    other_group = parser.add_argument_group("Other Parameters")
    other_group.add_argument(
        "--simulate_outcome",
        action="store_true",
        default=True,
        help="Whether to simulate outcome events. Default: True",
    )
    other_group.add_argument(
        "--no_simulate_outcome",
        action="store_false",
        dest="simulate_outcome",
        help="Disable outcome simulation (exposure-only mode).",
    )
    other_group.add_argument(
        "--exposure_name",
        default="EXPOSURE",
        help="Code name for simulated exposure events. Default: EXPOSURE",
    )
    other_group.add_argument(
        "--outcome_name",
        default="OUTCOME",
        help="Code name for simulated outcome events. Default: OUTCOME",
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


def save_simulation_parameters(args: argparse.Namespace, output_dir: str) -> None:
    """
    Save simulation parameters to JSON and command line to text file for reproducibility.

    Args:
        args: Parsed command line arguments
        output_dir: Directory to save parameter files
    """
    # Parse codes and weights
    confounder_codes, confounder_exposure_weights = parse_codes_and_weights(
        args.confounder_codes, args.confounder_exposure_weights
    )
    _, confounder_outcome_weights = parse_codes_and_weights(
        args.confounder_codes, args.confounder_outcome_weights
    )
    instrument_codes, instrument_weights = parse_codes_and_weights(
        args.instrument_codes, args.instrument_weights
    )
    prognostic_codes, prognostic_weights = parse_codes_and_weights(
        args.prognostic_codes, args.prognostic_weights
    )

    # Convert args to dictionary
    params = {
        "description": "Simulation of a causal effect on the outcome using the script tests/data_generation/induce_causal_effect.py",
        "simulation_timestamp": datetime.now().isoformat(),
        "source_dir": args.source_dir,
        "write_dir": args.write_dir,
        "trigger_codes": {
            "confounder_codes": confounder_codes,
            "confounder_exposure_weights": confounder_exposure_weights,
            "confounder_outcome_weights": confounder_outcome_weights,
            "instrument_codes": instrument_codes,
            "instrument_weights": instrument_weights,
            "prognostic_codes": prognostic_codes,
            "prognostic_weights": prognostic_weights,
        },
        "base_probabilities": {
            "p_base_exposure": args.p_base_exposure,
            "p_base_outcome": args.p_base_outcome,
            "p_daily_base_exposure": args.p_daily_base_exposure,
            "p_daily_base_outcome": args.p_daily_base_outcome,
        },
        "effect_sizes": {
            "exposure_outcome_effect": args.exposure_outcome_effect,
        },
        "other_parameters": {
            "simulate_outcome": args.simulate_outcome,
            "exposure_name": args.exposure_name,
            "outcome_name": args.outcome_name,
        },
    }

    # Save parameters to JSON file
    params_file = os.path.join(output_dir, "simulation_parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Simulation parameters saved to: {params_file}")

    # Save command line for easy copy-pasting
    command_file = os.path.join(output_dir, "reproduce_command.txt")
    command_line = " ".join(sys.argv)

    with open(command_file, "w") as f:
        f.write("# Command used to generate this simulation:\n")
        f.write(f"# Run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Working directory: {os.getcwd()}\n\n")
        f.write(f"{command_line}\n\n")
        f.write("# To reproduce this simulation, copy and paste the command above.\n")
        f.write(
            "# Make sure you're in the correct working directory and have the same input data.\n"
        )

    print(f"Command line saved to: {command_file}")


def main() -> None:
    """Main function to run the enhanced causal simulation."""
    args = create_parser().parse_args()

    # Parse all codes and weights
    confounder_codes, confounder_exposure_weights = parse_codes_and_weights(
        args.confounder_codes, args.confounder_exposure_weights
    )
    _, confounder_outcome_weights = parse_codes_and_weights(
        args.confounder_codes, args.confounder_outcome_weights
    )
    instrument_codes, instrument_weights = parse_codes_and_weights(
        args.instrument_codes, args.instrument_weights
    )
    prognostic_codes, prognostic_weights = parse_codes_and_weights(
        args.prognostic_codes, args.prognostic_weights
    )

    # Initialize simulator with all parameters
    simulator = CausalSimulator(
        confounder_codes=confounder_codes,
        confounder_exposure_weights=confounder_exposure_weights,
        confounder_outcome_weights=confounder_outcome_weights,
        instrument_codes=instrument_codes,
        instrument_weights=instrument_weights,
        prognostic_codes=prognostic_codes,
        prognostic_weights=prognostic_weights,
        p_base_exposure=args.p_base_exposure,
        p_base_outcome=args.p_base_outcome,
        p_daily_base_exposure=args.p_daily_base_exposure,
        p_daily_base_outcome=args.p_daily_base_outcome,
        exposure_outcome_effect=args.exposure_outcome_effect,
        exposure_name=args.exposure_name,
        outcome_name=args.outcome_name,
        simulate_outcome=args.simulate_outcome,
    )

    # Load data
    df, shards = DataManager.load_shards(args.source_dir)

    # Print initial statistics
    SimulationReporter.print_trigger_stats(df, simulator, args.simulate_outcome)

    # Run simulation (now simplified - no parameters needed)
    simulated_df, ite_df = simulator.simulate_dataset(df)

    # Print results
    SimulationReporter.print_simulation_results(
        simulated_df, simulator, args.simulate_outcome
    )

    # Save results
    os.makedirs(args.write_dir, exist_ok=True)

    # Save simulation parameters for reproducibility
    save_simulation_parameters(args, args.write_dir)

    # Save ITE data separately if outcomes were simulated
    if args.simulate_outcome and not ite_df.empty:
        ite_df.to_csv(os.path.join(args.write_dir, ".ite.csv"), index=False)
        ate = ite_df["ite"].mean()
        with open(os.path.join(args.write_dir, ".ate.txt"), "w") as f:
            f.write(f"ATE: {ate}")

    # Save main simulation results
    simulated_df.reset_index(drop=True, inplace=True)
    DataManager.write_shards(simulated_df, args.write_dir, shards)


if __name__ == "__main__":
    main()
