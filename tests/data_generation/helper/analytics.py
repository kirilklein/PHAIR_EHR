import pandas as pd

from tests.data_generation.helper.config import SimulationConfig


class SimulationReporter:
    """Generates simulation statistics and reports."""

    def __init__(self):
        self.simulation_text = ""
        self.trigger_text = ""

    def print_trigger_stats(self, df: pd.DataFrame, config: SimulationConfig) -> str:
        """Print statistics about trigger code presence before simulation."""
        output_lines = []

        total_subjects = df["subject_id"].nunique()
        subject_codes = df.groupby("subject_id")["code"].apply(set)

        output_lines.append(f"\nTotal subjects: {total_subjects}")
        output_lines.append("\nTrigger code presence before simulation (no censoring):")

        # Get all unique codes and compute their presence once
        all_codes = config.get_all_trigger_codes()
        code_stats = {}
        for code in all_codes:
            count = subject_codes.apply(lambda codes: code in codes).sum()
            percentage = 100 * count / total_subjects
            code_stats[code] = {"count": count, "percentage": percentage}

        # Display exposure trigger codes
        output_lines.append(f"\nEXPOSURE ({config.exposure.code}) trigger codes:")
        for i, code in enumerate(config.exposure.trigger_codes):
            stats = code_stats[code]
            weight = config.exposure.trigger_weights[i]
            output_lines.append(
                f"  {code}: {stats['count']} subjects ({stats['percentage']:.1f}%) [weight: {weight:.2f}]"
            )

        # Get confounders by outcome
        confounders_by_outcome = config.get_confounder_codes()

        # Display each outcome and its specific trigger codes
        output_lines.append(f"\nOUTCOMES:")
        for outcome_key, outcome_cfg in config.outcomes.items():
            output_lines.append(f"\n  {outcome_cfg.code}:")
            output_lines.append(f"    Base probability: {outcome_cfg.p_base:.3f}")
            output_lines.append(
                f"    Exposure effect: {outcome_cfg.exposure_effect:.2f}"
            )

            # Show confounders for this outcome
            outcome_confounders = confounders_by_outcome.get(outcome_key, [])
            if outcome_confounders:
                output_lines.append(
                    f"    Confounders for this outcome ({len(outcome_confounders)}):"
                )
                for code in outcome_confounders:
                    stats = code_stats[code]
                    # Find weights in both exposure and outcome
                    exp_weight = config.exposure.trigger_weights[
                        config.exposure.trigger_codes.index(code)
                    ]
                    out_weight = outcome_cfg.trigger_weights[
                        outcome_cfg.trigger_codes.index(code)
                    ]
                    output_lines.append(
                        f"      {code}: {stats['count']} subjects ({stats['percentage']:.1f}%) [exp: {exp_weight:.2f}, out: {out_weight:.2f}]"
                    )

            # Show prognostic codes for this outcome (only in outcome, not in exposure)
            prognostic_codes = [
                code
                for code in outcome_cfg.trigger_codes
                if code not in config.exposure.trigger_codes
            ]
            if prognostic_codes:
                output_lines.append(
                    f"    Prognostic codes for this outcome ({len(prognostic_codes)}):"
                )
                for code in prognostic_codes:
                    stats = code_stats[code]
                    weight = outcome_cfg.trigger_weights[
                        outcome_cfg.trigger_codes.index(code)
                    ]
                    output_lines.append(
                        f"      {code}: {stats['count']} subjects ({stats['percentage']:.1f}%) [weight: {weight:.2f}]"
                    )

        # Summary statistics
        all_confounders = config.get_all_confounder_codes()

        output_lines.append(f"\nSUMMARY:")
        output_lines.append(f"  Total unique trigger codes: {len(all_codes)}")
        output_lines.append(
            f"  Unique confounders (across all outcomes): {len(all_confounders)}"
        )
        output_lines.append(f"  Outcomes to simulate: {len(config.outcomes)}")

        # Join all lines and print
        report_text = "\n".join(output_lines)
        print(report_text)

        self.trigger_text = report_text

    def print_simulation_results(
        self, df: pd.DataFrame, simulation_config: SimulationConfig
    ) -> str:
        """Print simulation results statistics."""
        output_lines = []

        total_subjects = df["subject_id"].nunique()

        # Count simulated exposure events
        exposure_subjects = df.groupby("subject_id")["code"].apply(
            lambda codes: (codes == simulation_config.exposure.code).any()
        )
        exposure_count = exposure_subjects.sum()

        output_lines.append("\nSimulation results:")
        output_lines.append(
            f"  EXPOSURE ({simulation_config.exposure.code}): {exposure_count} subjects ({100 * exposure_count / total_subjects:.1f}%)"
        )

        # Process each outcome separately
        if simulation_config.outcomes:
            output_lines.append("\nOUTCOME RESULTS:")

            for outcome_cfg in simulation_config.outcomes.values():
                outcome_code = outcome_cfg.code

                # Count subjects with this outcome
                outcome_subjects = df.groupby("subject_id")["code"].apply(
                    lambda codes: (codes == outcome_code).any()
                )
                outcome_count = outcome_subjects.sum()

                output_lines.append(f"\n  {outcome_code}:")
                output_lines.append(
                    f"    Total subjects with outcome: {outcome_count} ({100 * outcome_count / total_subjects:.1f}%)"
                )

                # Calculate conditional probabilities
                if exposure_count > 0:
                    # P(Outcome | Exposure)
                    outcomes_given_exposure = (
                        outcome_subjects[exposure_subjects].mean() * 100
                    )

                    output_lines.append(
                        f"    P({outcome_code} | Exposure): {outcomes_given_exposure:.1f}%"
                    )
                else:
                    output_lines.append(
                        f"    P({outcome_code} | Exposure): N/A (no exposed subjects)"
                    )

                # Calculate for non-exposed subjects
                no_exposure_count = total_subjects - exposure_count
                if no_exposure_count > 0:
                    # P(Outcome | No Exposure)
                    outcomes_given_no_exposure = (
                        outcome_subjects[~exposure_subjects].mean() * 100
                    )

                    output_lines.append(
                        f"    P({outcome_code} | No Exposure): {outcomes_given_no_exposure:.1f}%"
                    )
                else:
                    output_lines.append(
                        f"    P({outcome_code} | No Exposure): N/A (all subjects exposed)"
                    )

                # Calculate risk difference and relative risk if both groups exist
                if exposure_count > 0 and no_exposure_count > 0:
                    risk_exposed = outcome_subjects[exposure_subjects].mean()
                    risk_unexposed = outcome_subjects[~exposure_subjects].mean()

                    risk_difference = (risk_exposed - risk_unexposed) * 100
                    if risk_unexposed > 0:
                        relative_risk = risk_exposed / risk_unexposed
                        output_lines.append(
                            f"    Risk Difference: {risk_difference:+.1f} percentage points"
                        )
                        output_lines.append(f"    Relative Risk: {relative_risk:.2f}")
                    else:
                        output_lines.append(
                            f"    Risk Difference: {risk_difference:+.1f} percentage points"
                        )
                        output_lines.append(
                            f"    Relative Risk: undefined (no events in unexposed)"
                        )

        # Summary statistics
        output_lines.append(f"\nSUMMARY:")
        output_lines.append(f"  Total subjects: {total_subjects}")
        output_lines.append(
            f"  Exposed subjects: {exposure_count} ({100 * exposure_count / total_subjects:.1f}%)"
        )
        output_lines.append(
            f"  Unexposed subjects: {total_subjects - exposure_count} ({100 * (total_subjects - exposure_count) / total_subjects:.1f}%)"
        )
        output_lines.append(f"  Outcomes simulated: {len(simulation_config.outcomes)}")

        # Join all lines and print
        report_text = "\n".join(output_lines)
        print(report_text)

        self.simulation_text = report_text

    @staticmethod
    def save_report_to_file(report_text: str, filepath: str) -> None:
        """Save a report text to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to: {filepath}")

    def save_report(self, save_path: str = None) -> str:
        """Generate a combined report with both trigger stats and simulation results."""
        if self.trigger_text != "" and self.simulation_text != "":
            combined_report = self.trigger_text + "\n" + self.simulation_text
            self.save_report_to_file(combined_report, save_path)
        else:
            print("No report to save")
