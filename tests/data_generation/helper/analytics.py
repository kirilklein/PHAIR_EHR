import pandas as pd
from tests.data_generation.helper.induce_causal_effect import CausalSimulator
from tests.data_generation.helper.config import SimulationConfig


class SimulationReporter:
    """Generates simulation statistics and reports."""

    @staticmethod
    def print_trigger_stats(df: pd.DataFrame, config: SimulationConfig) -> None:
        """Print statistics about trigger code presence before simulation."""
        total_subjects = df["subject_id"].nunique()
        subject_codes = df.groupby("subject_id")["code"].apply(set)

        print(f"\nTotal subjects: {total_subjects}")
        print("\nTrigger code presence before simulation:")

        # Get all unique codes and compute their presence once
        all_codes = config.get_all_trigger_codes()
        code_stats = {}
        for code in all_codes:
            count = subject_codes.apply(lambda codes: code in codes).sum()
            percentage = 100 * count / total_subjects
            code_stats[code] = {"count": count, "percentage": percentage}

        # Display exposure trigger codes
        print(f"\nEXPOSURE ({config.exposure.code}) trigger codes:")
        for i, code in enumerate(config.exposure.trigger_codes):
            stats = code_stats[code]
            weight = config.exposure.trigger_weights[i]
            print(
                f"  {code}: {stats['count']} subjects ({stats['percentage']:.1f}%) [weight: {weight:.2f}]"
            )

        # Get confounders by outcome
        confounders_by_outcome = config.get_confounder_codes()

        # Display each outcome and its specific trigger codes
        print(f"\nOUTCOMES:")
        for outcome_key, outcome_cfg in config.outcomes.items():
            print(f"\n  {outcome_cfg.code}:")
            print(f"    Base probability: {outcome_cfg.p_base:.3f}")
            print(f"    Exposure effect: {outcome_cfg.exposure_effect:.2f}")

            # Show confounders for this outcome
            outcome_confounders = confounders_by_outcome.get(outcome_key, [])
            if outcome_confounders:
                print(f"    Confounders for this outcome ({len(outcome_confounders)}):")
                for code in outcome_confounders:
                    stats = code_stats[code]
                    # Find weights in both exposure and outcome
                    exp_weight = config.exposure.trigger_weights[
                        config.exposure.trigger_codes.index(code)
                    ]
                    out_weight = outcome_cfg.trigger_weights[
                        outcome_cfg.trigger_codes.index(code)
                    ]
                    print(
                        f"      {code}: {stats['count']} subjects ({stats['percentage']:.1f}%) [exp: {exp_weight:.2f}, out: {out_weight:.2f}]"
                    )

            # Show prognostic codes for this outcome (only in outcome, not in exposure)
            prognostic_codes = [
                code
                for code in outcome_cfg.trigger_codes
                if code not in config.exposure.trigger_codes
            ]
            if prognostic_codes:
                print(
                    f"    Prognostic codes for this outcome ({len(prognostic_codes)}):"
                )
                for code in prognostic_codes:
                    stats = code_stats[code]
                    weight = outcome_cfg.trigger_weights[
                        outcome_cfg.trigger_codes.index(code)
                    ]
                    print(
                        f"      {code}: {stats['count']} subjects ({stats['percentage']:.1f}%) [weight: {weight:.2f}]"
                    )

        # Summary statistics
        all_confounders = config.get_all_confounder_codes()

        print(f"\nSUMMARY:")
        print(f"  Total unique trigger codes: {len(all_codes)}")
        print(f"  Unique confounders (across all outcomes): {len(all_confounders)}")
        print(f"  Outcomes to simulate: {len(config.outcomes)}")

        # Show confounder breakdown by outcome
        print(f"\nCONFOUNDER BREAKDOWN BY OUTCOME:")
        for outcome_key, outcome_cfg in config.outcomes.items():
            outcome_confounders = confounders_by_outcome.get(outcome_key, [])
            print(f"  {outcome_cfg.code}: {len(outcome_confounders)} confounders")

    @staticmethod
    def print_simulation_results(
        df: pd.DataFrame, simulator: CausalSimulator, simulate_outcome: bool = True
    ) -> None:
        """Print simulation results statistics."""
        total_subjects = df["subject_id"].nunique()

        # Count simulated events
        exposure_subjects = df.groupby("subject_id")["code"].apply(
            lambda codes: (codes == simulator.exposure_name).any()
        )
        exposure_count = exposure_subjects.sum()

        print("\nSimulation results:")
        print(
            f"  EXPOSURE events: {exposure_count} subjects ({100 * exposure_count / total_subjects:.1f}%)"
        )

        if simulate_outcome:
            outcome_subjects = df.groupby("subject_id")["code"].apply(
                lambda codes: (codes == simulator.outcome_name).any()
            )
            outcome_count = outcome_subjects.sum()

            # Conditional probabilities
            outcomes_given_exposure = outcome_subjects[exposure_subjects].mean() * 100
            outcomes_given_no_exposure = (
                outcome_subjects[~exposure_subjects].mean() * 100
            )

            print(
                f"  OUTCOME events: {outcome_count} subjects ({100 * outcome_count / total_subjects:.1f}%)"
            )
            print(f"  P(Outcome | Exposure): {outcomes_given_exposure:.1f}%")
            print(f"  P(Outcome | No Exposure): {outcomes_given_no_exposure:.1f}%")
