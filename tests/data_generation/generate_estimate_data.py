import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from corebehrt.constants.causal.data import (
    CF_PROBAS,
    EXPOSURE_COL,
    OUTCOMES,
    PROBAS,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
    TARGETS,
)
from corebehrt.constants.causal.paths import (
    CALIBRATED_PREDICTIONS_FILE,
    SIMULATION_RESULTS_FILE,
)
from corebehrt.constants.data import PID_COL

# Paths from config
EXPOSURE_PRED_DIR = "./outputs/causal/generated/calibrated_predictions"
OUTCOME_PRED_DIR = "./outputs/causal/generated/trained_mlp_simulated"
COUNTERFACTUAL_OUTCOMES_DIR = "./outputs/causal/generated/simulated_outcome"

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


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [EXPOSURE_PRED_DIR, OUTCOME_PRED_DIR, COUNTERFACTUAL_OUTCOMES_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def generate_exposure_predictions(
    n_samples=NUM_SAMPLES, seed=42, exposure_noise=EXPOSURE_NOISE
):
    """
    Generate exposure predictions data.
    This data contains propensity scores and true exposure values.
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

    ps_scores = np.clip(
        expit(logit(exposure_probas) + np.random.normal(0, exposure_noise, n_samples)),
        CLIP_EPS,
        1 - CLIP_EPS,
    )
    # Create DataFrame
    df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            PROBAS: ps_scores,  # This will be used as the PS distribution
            TARGETS: exposures,  # This column stores exposure (will be labeled accordingly)
        }
    )

    # Save to file
    output_path = os.path.join(EXPOSURE_PRED_DIR, CALIBRATED_PREDICTIONS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Exposure predictions saved to {output_path}")

    return subject_ids, exposure_probas, ps_scores, exposures


def generate_counterfactual_outcomes(
    subject_ids,
    exposures,
    propensities,
    weight=1,
    intercept=0,
    exposure_effect=1,
    seed=42,
):
    """
    Generate counterfactual (true) outcomes data.
    This contains the ground truth potential outcomes under both treatment and control.
    """
    np.random.seed(seed)

    # Generate potential outcome probabilities:
    # p0: probability of outcome if subject was in control group
    # p1: probability of outcome if subject was in treatment group
    p0 = expit(logit(propensities) * weight + intercept)
    p0 = np.clip(p0, CLIP_EPS, 1 - CLIP_EPS)
    p1 = expit(logit(p0) + exposure_effect)
    p1 = np.clip(p1, CLIP_EPS, 1 - CLIP_EPS)  # Ensure probas are in [0.01, 0.99]

    # Generate potential outcomes
    y0 = np.random.binomial(1, p0)  # Outcomes under control
    y1 = np.random.binomial(1, p1)  # Outcomes under treatment

    # The actual observed outcome should match either y0 or y1 depending on the actual exposure
    observed_outcomes = exposures * y1 + (1 - exposures) * y0

    # Create DataFrame with counterfactual outcomes
    df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            SIMULATED_OUTCOME_EXPOSED: y1,  # Outcome under treatment (Y1)
            SIMULATED_OUTCOME_CONTROL: y0,  # Outcome under control (Y0)
            SIMULATED_PROBAS_EXPOSED: p1,  # P1
            SIMULATED_PROBAS_CONTROL: p0,  # P0
            EXPOSURE_COL: exposures,
            PROBAS: np.where(
                exposures == 1, p1, p0
            ),  # Observed probability based on actual exposure
            OUTCOMES: observed_outcomes,
        }
    )

    # Save to file
    output_path = os.path.join(COUNTERFACTUAL_OUTCOMES_DIR, SIMULATION_RESULTS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Counterfactual outcomes saved to {output_path}")

    return df


def generate_outcome_predictions(
    subject_ids, cf_data, noise_scale=OUTCOME_NOISE, seed=42
):
    """
    Generate outcome predictions data.
    This contains predicted probabilities of outcomes and the actual outcomes,
    based on the true counterfactual data but with some noise added.

    Output format:
    - subject_id: Patient identifier
    - probas: Predicted probability under the factual (observed) treatment
    - cf_probas: Predicted probability under the counterfactual (alternative) treatment
    - targets: Actual outcome
    """
    np.random.seed(seed)

    # Extract true counterfactual data
    p0 = cf_data[SIMULATED_PROBAS_CONTROL].values
    p1 = cf_data[SIMULATED_PROBAS_EXPOSED].values
    exposures = cf_data[EXPOSURE_COL].values
    outcomes = cf_data[OUTCOMES].values

    # Add noise to create "predicted" probabilities
    noise_p0 = np.random.normal(0, noise_scale, len(p0))
    noise_p1 = np.random.normal(0, noise_scale, len(p1))

    predicted_p0 = np.clip(expit(logit(p0) + noise_p0), CLIP_EPS, 1 - CLIP_EPS)
    predicted_p1 = np.clip(expit(logit(p1) + noise_p1), CLIP_EPS, 1 - CLIP_EPS)

    # For each subject:
    #   If exposure == 1: factual prediction is predicted_p1, counterfactual is predicted_p0.
    #   If exposure == 0: factual prediction is predicted_p0, counterfactual is predicted_p1.
    factual_probas = np.where(exposures == 1, predicted_p1, predicted_p0)
    counterfactual_probas = np.where(exposures == 1, predicted_p0, predicted_p1)

    # Create DataFrame with the correct format
    df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            PROBAS: factual_probas,  # Predicted probability under factual exposure
            CF_PROBAS: counterfactual_probas,  # Predicted probability under counterfactual exposure
            TARGETS: outcomes,
        }
    )

    # Save to file
    output_path = os.path.join(OUTCOME_PRED_DIR, CALIBRATED_PREDICTIONS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Outcome predictions saved to {output_path}")

    return df


def main(
    generate_figures=False,
    exposure_noise=EXPOSURE_NOISE,
    outcome_noise=OUTCOME_NOISE,
    num_samples=NUM_SAMPLES,
    weight=OUTCOME_PS_WEIGHT,
    intercept=OUTCOME_INTERCEPT,
    exposure_effect=EXPOSURE_EFFECT,
):
    """Generate all necessary data for testing estimate.py"""
    create_directories()

    # Generate data with 1000 samples
    subject_ids, exposure_probas, ps_scores, exposures = generate_exposure_predictions(
        n_samples=num_samples, exposure_noise=exposure_noise
    )

    # Generate counterfactual/true (simulated) outcomes (ground truth)
    cf_data = generate_counterfactual_outcomes(
        subject_ids,
        exposures,
        propensities=exposure_probas,
        weight=weight,
        intercept=intercept,
        exposure_effect=exposure_effect,
    )

    # Generate outcome/cf outcome predictions (with noise) using the counterfactual data
    outcome_df = generate_outcome_predictions(
        subject_ids, cf_data, noise_scale=outcome_noise
    )
    if not generate_figures:
        return
    from tests.data_generation.plot_generated_data import (  # Save visualization figures using the proper data
        save_comparison_figures,
        save_outcome_probas_by_exposure_figure,
        save_outcome_probas_figures,
        save_predicted_outcome_probas_distribution,
        save_ps_distribution_figure,
    )

    save_ps_distribution_figure(subject_ids, ps_scores, exposures)
    save_outcome_probas_figures(subject_ids, cf_data, outcome_df)
    save_comparison_figures(subject_ids, cf_data, outcome_df)
    save_predicted_outcome_probas_distribution(cf_data, outcome_df)
    save_outcome_probas_by_exposure_figure(subject_ids, cf_data, outcome_df)
    print("All test data generated successfully!")


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
    )
