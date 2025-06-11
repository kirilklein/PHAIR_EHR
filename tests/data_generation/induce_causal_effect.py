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
    - Exposure-only codes: Only affect exposure probability
    - Outcome-only codes: Only affect outcome probability
    - Exposure events: Directly affect outcome probability (the causal effect of interest)

EXAMPLE USAGE:
    # Basic usage with default parameters
    python induce_causal_effect.py \
        --source_dir ./data/input_shards \
        --write_dir ./data/simulated_shards

    # Custom causal effects
    python induce_causal_effect.py \
        --source_dir ./data/input_shards \
        --write_dir ./data/simulated_shards \
        --confounder_exposure_effect 0.5 \
        --exposure_outcome_effect 1.5 \
        --p_base_exposure 0.25 \
        --p_base_outcome 0.15

    # Using custom trigger codes
    python induce_causal_effect.py \
        --source_dir ./data/input_shards \
        --write_dir ./data/simulated_shards \
        --confounder_code "LAB_GLUCOSE" \
        --exposure_only_code "DRUG_STATINS" \
        --outcome_only_code "DIAG_DIABETES"

PARAMETER GUIDANCE:
    - Effect sizes: Typical range [-2.0, 2.0]. Positive = increases probability, negative = decreases
    - Base probabilities: Should reflect realistic event rates (0.1-0.4 often reasonable)
    - Days offset: Time delay between trigger and simulated event (30-90 days typical)
    
OUTPUT:
    - Parquet files with original data + simulated EXPOSURE and OUTCOME events
    - .ate.txt file containing the Average Treatment Effect for validation

NOTES:
    - Input directory must contain .parquet shard files
    - Each shard should have columns: subject_id, time, code, numeric_value
    - Simulated events are added with temporal ordering preserved
"""

import os
import argparse

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
  
  # Custom parameters  
  python %(prog)s --source_dir ./input --write_dir ./output \\
    --p_base_exposure 0.25 --exposure_outcome_effect 1.5
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
        "--confounder_code",
        default="DDZ32",
        help="Code that affects both exposure and outcome (creates confounding). Default: LAB8",
    )
    trigger_group.add_argument(
        "--exposure_only_code",
        default="MME01",
        help="Code that only affects exposure probability. Default: DDZ32",
    )
    trigger_group.add_argument(
        "--outcome_only_code",
        default="DE11",
        help="Code that only affects outcome probability. Default: DE11",
    )

    # Base probabilities
    prob_group = parser.add_argument_group(
        "Base Probabilities", "Baseline event rates without any triggers"
    )
    prob_group.add_argument(
        "--p_base_exposure",
        type=float,
        default=0.2,
        help="Base probability for exposure events (0-1). Default: 0.3",
    )
    prob_group.add_argument(
        "--p_base_outcome",
        type=float,
        default=0.2,
        help="Base probability for outcome events (0-1). Default: 0.2",
    )

    # Effect sizes
    effects_group = parser.add_argument_group(
        "Effect Sizes", "Logistic regression coefficients for causal relationships"
    )
    effects_group.add_argument(
        "--confounder_exposure_effect",
        type=float,
        default=1.2,
        help="Effect of confounder on exposure (positive increases probability). Default: 0.3",
    )
    effects_group.add_argument(
        "--confounder_outcome_effect",
        type=float,
        default=-0.3,
        help="Effect of confounder on outcome (negative decreases probability). Default: -0.3",
    )
    effects_group.add_argument(
        "--exposure_only_effect",
        type=float,
        default=0.8,
        help="Effect of exposure-only trigger on exposure. Default: 1.5",
    )
    effects_group.add_argument(
        "--outcome_only_effect",
        type=float,
        default=0.5,
        help="Effect of outcome-only trigger on outcome. Default: 0.5",
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
        type=bool,
        default=True,
        help="Whether to simulate outcome events (set False for exposure-only). Default: True",
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


def main() -> None:
    """Main function to run the enhanced causal simulation."""
    args = create_parser().parse_args()

    # Initialize components
    simulator = CausalSimulator(
        args.confounder_code,
        args.exposure_only_code,
        args.outcome_only_code,
        args.exposure_name,
        args.outcome_name,
    )

    # Load data
    df, shards = DataManager.load_shards(args.source_dir)

    # Print initial statistics
    SimulationReporter.print_trigger_stats(df, simulator, args.simulate_outcome)

    # Run simulation
    simulated_df, ite_df = simulator.simulate_dataset(
        df,
        args.p_base_exposure,
        args.p_base_outcome,
        args.confounder_exposure_effect,
        args.confounder_outcome_effect,
        args.exposure_only_effect,
        args.outcome_only_effect,
        args.exposure_outcome_effect,
        args.simulate_outcome,
    )

    # Print results
    SimulationReporter.print_simulation_results(
        simulated_df, simulator, args.simulate_outcome
    )

    # Save results
    os.makedirs(args.write_dir, exist_ok=True)

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
