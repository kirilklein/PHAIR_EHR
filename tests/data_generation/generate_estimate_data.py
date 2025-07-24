import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    OUTCOMES,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
)
from corebehrt.constants.causal.paths import (
    SIMULATION_RESULTS_FILE,
)
from corebehrt.constants.data import PID_COL

# Updated paths to match new structure
OUTPUT_DIR = "./outputs/causal/generated/calibrated_predictions"
COUNTERFACTUAL_DIR = "./outputs/causal/generated/simulated_outcome"
COMBINED_PREDICTIONS_FILE = "combined_calibrated_predictions.csv"
COMBINED_COUNTERFACTUAL_FILE = "combined_simulation_results.csv"

# Optional: Add different noise scales for different components
EXPOSURE_NOISE = 0.03  # For propensity scores
OUTCOME_NOISE = 0.05  # For outcome predictions
NUM_SAMPLES = 10_000
OUTCOME_PS_WEIGHT = 0.1  # in logit space
OUTCOME_INTERCEPT = -1  # in logit space
EXPOSURE_EFFECT = 0.5  # in logit space

CLIP_EPS = 0.001
PS_BETA_A = 2
PS_BETA_B = 5

# Define multiple outcomes to match new format
OUTCOME_NAMES = ["OUTCOME", "OUTCOME_2", "OUTCOME_3"]


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [OUTPUT_DIR, COUNTERFACTUAL_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def generate_combined_predictions(
    n_samples=NUM_SAMPLES,
    seed=42,
    exposure_noise=EXPOSURE_NOISE,
    outcome_noise=OUTCOME_NOISE,
    weight=OUTCOME_PS_WEIGHT,
    intercept=OUTCOME_INTERCEPT,
    exposure_effect=EXPOSURE_EFFECT,
):
    """
    Generate combined calibrated predictions data that matches the new format.
    This creates a single CSV with propensity scores, exposure, and multiple outcomes.
    """
    np.random.seed(seed)

    # Generate subject IDs
    subject_ids = np.arange(1, n_samples + 1)

    # Generate propensity scores (probability of treatment)
    exposure_probas = np.random.beta(
        PS_BETA_A, PS_BETA_B, n_samples
    )  # Skewed toward control group

    # Generate binary exposure based on propensity scores
    exposures = np.random.binomial(1, exposure_probas)

    # Add noise to propensity scores for realistic predictions
    ps_scores = np.clip(
        expit(logit(exposure_probas) + np.random.normal(0, exposure_noise, n_samples)),
        CLIP_EPS,
        1 - CLIP_EPS,
    )

    # Initialize the main DataFrame with basic columns
    df_data = {
        "subject_id": subject_ids,
        "ps": ps_scores,
        "exposure": exposures,
    }

    # Store true counterfactual data for the combined counterfactual file
    counterfactual_data = {
        "subject_id": subject_ids,
        "exposure": exposures,
        "ps": ps_scores,
    }

    # Generate data for each outcome
    for outcome_name in OUTCOME_NAMES:
        # Generate potential outcome probabilities for this outcome
        # Each outcome may have slightly different characteristics
        outcome_seed = seed + hash(outcome_name) % 1000
        np.random.seed(outcome_seed)

        # Vary the effect sizes slightly for different outcomes
        outcome_specific_effect = exposure_effect + np.random.normal(0, 0.01)
        outcome_specific_intercept = intercept + np.random.normal(0, 0.01)
        outcome_specific_weight = weight + np.random.normal(0, 0.01)

        # p0: probability of outcome if subject was in control group
        # p1: probability of outcome if subject was in treatment group
        p0 = expit(
            logit(exposure_probas) * outcome_specific_weight
            + outcome_specific_intercept
        )
        p0 = np.clip(p0, CLIP_EPS, 1 - CLIP_EPS)
        p1 = expit(logit(p0) + outcome_specific_effect)
        p1 = np.clip(p1, CLIP_EPS, 1 - CLIP_EPS)

        # Generate potential outcomes (true counterfactuals)
        y0 = np.random.binomial(1, p0)  # Outcomes under control
        y1 = np.random.binomial(1, p1)  # Outcomes under treatment

        # The actual observed outcome matches either y0 or y1 depending on exposure
        observed_outcomes = exposures * y1 + (1 - exposures) * y0

        # Add noise to create "predicted" probabilities
        noise_p0 = np.random.normal(0, outcome_noise, len(p0))
        noise_p1 = np.random.normal(0, outcome_noise, len(p1))

        predicted_p0 = np.clip(expit(logit(p0) + noise_p0), CLIP_EPS, 1 - CLIP_EPS)
        predicted_p1 = np.clip(expit(logit(p1) + noise_p1), CLIP_EPS, 1 - CLIP_EPS)

        # For each subject:
        #   If exposure == 1: factual prediction is predicted_p1, counterfactual is predicted_p0
        #   If exposure == 0: factual prediction is predicted_p0, counterfactual is predicted_p1
        factual_probas = np.where(exposures == 1, predicted_p1, predicted_p0)
        counterfactual_probas = np.where(exposures == 1, predicted_p0, predicted_p1)

        # Add outcome-specific columns to the predictions DataFrame
        df_data[f"probas_{outcome_name}"] = factual_probas
        df_data[f"outcome_{outcome_name}"] = observed_outcomes
        df_data[f"cf_probas_{outcome_name}"] = counterfactual_probas

        # Add true counterfactual data for this outcome
        counterfactual_data[f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"] = y0
        counterfactual_data[f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"] = y1
        counterfactual_data[f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"] = p0
        counterfactual_data[f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"] = p1
        counterfactual_data[f"outcome_{outcome_name}"] = observed_outcomes

    # Create the final DataFrames
    df = pd.DataFrame(df_data)
    counterfactual_df = pd.DataFrame(counterfactual_data)

    # Save to the combined files
    output_path = os.path.join(OUTPUT_DIR, COMBINED_PREDICTIONS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Combined calibrated predictions saved to {output_path}")

    counterfactual_path = os.path.join(COUNTERFACTUAL_DIR, COMBINED_COUNTERFACTUAL_FILE)
    counterfactual_df.to_csv(counterfactual_path, index=False)
    print(f"Combined counterfactual outcomes saved to {counterfactual_path}")

    return df, counterfactual_df


def generate_counterfactual_outcomes_legacy(df, counterfactual_df, seed=42):
    """
    Generate legacy simulation results files for backwards compatibility.
    Creates separate files for each outcome.
    """
    np.random.seed(seed)

    for outcome_name in OUTCOME_NAMES:
        # Extract data for this specific outcome
        legacy_data = {
            PID_COL: counterfactual_df["subject_id"],
            SIMULATED_OUTCOME_EXPOSED: counterfactual_df[
                f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"
            ],
            SIMULATED_OUTCOME_CONTROL: counterfactual_df[
                f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"
            ],
            SIMULATED_PROBAS_EXPOSED: counterfactual_df[
                f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"
            ],
            SIMULATED_PROBAS_CONTROL: counterfactual_df[
                f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"
            ],
            EXPOSURE_COL: counterfactual_df["exposure"],
            "probas": np.where(
                counterfactual_df["exposure"] == 1,
                counterfactual_df[f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"],
                counterfactual_df[f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"],
            ),
            OUTCOMES: counterfactual_df[f"outcome_{outcome_name}"],
        }

        legacy_df = pd.DataFrame(legacy_data)

        # Save legacy file for this outcome
        legacy_filename = f"{outcome_name}_{SIMULATION_RESULTS_FILE}"
        legacy_output_path = os.path.join(COUNTERFACTUAL_DIR, legacy_filename)
        legacy_df.to_csv(legacy_output_path, index=False)
        print(
            f"Legacy simulation results for {outcome_name} saved to {legacy_output_path}"
        )


def main(
    generate_figures=False,
    exposure_noise=EXPOSURE_NOISE,
    outcome_noise=OUTCOME_NOISE,
    num_samples=NUM_SAMPLES,
    weight=OUTCOME_PS_WEIGHT,
    intercept=OUTCOME_INTERCEPT,
    exposure_effect=EXPOSURE_EFFECT,
):
    """Generate combined calibrated predictions and counterfactual data for testing estimate.py"""
    create_directories()

    # Generate the main combined predictions and counterfactual files
    combined_df, counterfactual_df = generate_combined_predictions(
        n_samples=num_samples,
        exposure_noise=exposure_noise,
        outcome_noise=outcome_noise,
        weight=weight,
        intercept=intercept,
        exposure_effect=exposure_effect,
    )

    # Generate legacy files for backwards compatibility
    generate_counterfactual_outcomes_legacy(combined_df, counterfactual_df)

    if not generate_figures:
        return combined_df, counterfactual_df

    # Generate visualization figures if requested
    from tests.data_generation.plot_generated_data import (
        save_comparison_figures,
        save_outcome_probas_by_exposure_figure,
        save_outcome_probas_figures,
        save_predicted_outcome_probas_distribution,
        save_ps_distribution_figure,
    )

    # Use the first outcome for visualization
    outcome_name = OUTCOME_NAMES[0]
    subject_ids = combined_df["subject_id"]
    ps_scores = combined_df["ps"]
    exposures = combined_df["exposure"]

    # Create a simplified outcome_df for visualization compatibility
    outcome_df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            "probas": combined_df[f"probas_{outcome_name}"],
            "cf_probas": combined_df[f"cf_probas_{outcome_name}"],
            "targets": combined_df[f"outcome_{outcome_name}"],
        }
    )

    # Create a simplified cf_data for visualization compatibility
    cf_data = pd.DataFrame(
        {
            PID_COL: subject_ids,
            SIMULATED_OUTCOME_CONTROL: counterfactual_df[
                f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"
            ],
            SIMULATED_OUTCOME_EXPOSED: counterfactual_df[
                f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"
            ],
            SIMULATED_PROBAS_CONTROL: counterfactual_df[
                f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"
            ],
            SIMULATED_PROBAS_EXPOSED: counterfactual_df[
                f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"
            ],
            EXPOSURE_COL: exposures,
            "probas": combined_df[f"probas_{outcome_name}"],
            OUTCOMES: combined_df[f"outcome_{outcome_name}"],
        }
    )

    save_ps_distribution_figure(subject_ids, ps_scores, exposures)
    save_outcome_probas_figures(subject_ids, cf_data, outcome_df)
    save_comparison_figures(subject_ids, cf_data, outcome_df)
    save_predicted_outcome_probas_distribution(cf_data, outcome_df)
    save_outcome_probas_by_exposure_figure(subject_ids, cf_data, outcome_df)
    print("All test data generated successfully!")

    return combined_df, counterfactual_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exposure-noise",
        type=float,
        default=EXPOSURE_NOISE,
        help="Noise in exposure predictions",
    )
    parser.add_argument(
        "--outcome-noise",
        type=float,
        default=OUTCOME_NOISE,
        help="Noise in outcome predictions",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=OUTCOME_PS_WEIGHT,
        help="Weight in the counterfactual outcome model",
    )
    parser.add_argument(
        "--intercept",
        type=float,
        default=OUTCOME_INTERCEPT,
        help="Intercept in the counterfactual outcome model",
    )
    parser.add_argument(
        "--exposure-effect", type=float, default=1, help="Effect of exposure on outcome"
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        default=False,
        help="Generate visualization figures",
    )
    args = parser.parse_args()

    main(
        generate_figures=args.generate_figures,
        exposure_noise=args.exposure_noise,
        outcome_noise=args.outcome_noise,
        num_samples=args.num_samples,
        exposure_effect=args.exposure_effect,
        weight=args.weight,
        intercept=args.intercept,
    )
