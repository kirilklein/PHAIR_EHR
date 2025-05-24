"""
Enhanced causal effect simulation module for EHR data.

Simulates binary exposure and outcome events based on multiple trigger conditions:
- Confounder: affects both exposure and outcome
- Exposure-only trigger: affects only exposure
- Outcome-only trigger: affects only outcome

Uses logistic model: P(event) = expit(logit(p_base) + Σ(effect_i * trigger_i))
"""

import os
import argparse

from tests.data_generation.helper.induce_causal_effect import (
    CausalSimulator,
    DataManager,
    SimulationReporter,
)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced simulated causal data"
    )

    # Required arguments
    parser.add_argument(
        "--source_dir", required=True, help="Directory containing source data shards"
    )
    parser.add_argument(
        "--write_dir", required=True, help="Directory to write output shards"
    )

    # Trigger codes
    parser.add_argument(
        "--confounder_code",
        default="LAB8",
        help="Code that affects both exposure and outcome",
    )
    parser.add_argument(
        "--exposure_only_code", default="DDZ32", help="Code that only affects exposure"
    )
    parser.add_argument(
        "--outcome_only_code", default="DE11", help="Code that only affects outcome"
    )

    # Base probabilities
    parser.add_argument(
        "--p_base_exposure",
        type=float,
        default=0.3,
        help="Base probability for exposure",
    )
    parser.add_argument(
        "--p_base_outcome", type=float, default=0.2, help="Base probability for outcome"
    )

    # Effect sizes
    parser.add_argument(
        "--confounder_exposure_effect",
        type=float,
        default=-0.5,
        help="Effect of confounder on exposure",
    )
    parser.add_argument(
        "--confounder_outcome_effect",
        type=float,
        default=-0.3,
        help="Effect of confounder on outcome",
    )
    parser.add_argument(
        "--exposure_only_effect",
        type=float,
        default=1.0,
        help="Effect of exposure-only trigger",
    )
    parser.add_argument(
        "--outcome_only_effect",
        type=float,
        default=0.5,
        help="Effect of outcome-only trigger",
    )
    parser.add_argument(
        "--exposure_outcome_effect",
        type=float,
        default=2.0,
        help="Effect of exposure on outcome",
    )

    # Other parameters
    parser.add_argument(
        "--days_offset",
        type=int,
        default=30,
        help="Days offset for temporal relationships",
    )
    parser.add_argument(
        "--simulate_outcome",
        type=bool,
        default=True,
        help="Whether to simulate outcome",
    )
    parser.add_argument(
        "--exposure_name", default="EXPOSURE", help="Name of exposure event"
    )
    parser.add_argument(
        "--outcome_name", default="OUTCOME", help="Name of outcome event"
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
    simulated_df = simulator.simulate_dataset(
        df,
        args.p_base_exposure,
        args.p_base_outcome,
        args.confounder_exposure_effect,
        args.confounder_outcome_effect,
        args.exposure_only_effect,
        args.outcome_only_effect,
        args.exposure_outcome_effect,
        args.days_offset,
        args.simulate_outcome,
    )

    # Print results
    SimulationReporter.print_simulation_results(
        simulated_df, simulator, args.simulate_outcome
    )

    # Save results
    os.makedirs(args.write_dir, exist_ok=True)

    if args.simulate_outcome and "ite" in simulated_df.columns:
        ate = simulated_df["ite"].mean()
        with open(os.path.join(args.write_dir, ".ate.txt"), "w") as f:
            f.write(f"ATE: {ate}")
        simulated_df = simulated_df.drop(columns=["ite"])
    simulated_df.reset_index(drop=True, inplace=True)
    DataManager.write_shards(simulated_df, args.write_dir, shards)


if __name__ == "__main__":
    main()
