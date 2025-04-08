import os
from pathlib import Path

import numpy as np
import pandas as pd

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    OUTCOMES,
    PROBAS,
    PROBAS_CONTROL,
    PROBAS_EXPOSED,
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
from tests.data_generation.plot_generated_data import (
    save_comparison_figures,
    save_outcome_probas_figures,
    save_predicted_outcome_probas_distribution,
    save_ps_distribution_figure,
    save_outcome_probas_by_exposure_figure,
)

# Paths from config
EXPOSURE_PRED_DIR = "./outputs/causal/generated/calibrated_predictions"
OUTCOME_PRED_DIR = "./outputs/causal/generated/trained_mlp_simulated"
COUNTERFACTUAL_OUTCOMES_DIR = "./outputs/causal/generated/simulated_outcome"


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [EXPOSURE_PRED_DIR, OUTCOME_PRED_DIR, COUNTERFACTUAL_OUTCOMES_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def generate_exposure_predictions(n_samples=1000, seed=42):
    """
    Generate exposure predictions data.
    This data contains propensity scores and true exposure values.
    """
    np.random.seed(seed)

    # Generate subject IDs
    subject_ids = np.arange(1, n_samples + 1)

    # Generate propensity scores (probability of treatment)
    ps_scores = np.random.beta(2, 5, n_samples)  # Skewed toward control group

    # Generate binary exposure based on propensity scores
    exposures = np.random.binomial(1, ps_scores)

    # Create DataFrame
    df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            PROBAS: ps_scores,  # Will be renamed to PS_COL
            TARGETS: exposures,  # Will be renamed to EXPOSURE_COL
        }
    )

    # Save to file
    output_path = os.path.join(EXPOSURE_PRED_DIR, CALIBRATED_PREDICTIONS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Exposure predictions saved to {output_path}")

    return subject_ids, ps_scores, exposures


def generate_counterfactual_outcomes(subject_ids, ps_scores, exposures, seed=42):
    """
    Generate counterfactual outcomes data.
    This contains the ground truth potential outcomes under both treatment and control.
    """
    np.random.seed(seed)

    n_samples = len(subject_ids)

    # Parameters for generating counterfactual outcomes
    baseline_effect = 0.25  # Average treatment effect

    # Generate potential outcome probabilities
    # P0: probability of outcome if subject was in control group
    # P1: probability of outcome if subject was in treatment group
    p0 = np.random.beta(2, 5, n_samples)  # Lower probability for control
    p1 = (
        p0 + baseline_effect + np.random.normal(0, 0.05, n_samples)
    )  # Higher probability for treatment
    p1 = np.clip(p1, 0.01, 0.99)  # Ensure probas are in [0.01, 0.99]

    # Generate potential outcomes
    y0 = np.random.binomial(1, p0)  # Outcomes under control
    y1 = np.random.binomial(1, p1)  # Outcomes under treatment

    # The actual observed outcome should match either Y0 or Y1 depending on actual exposure
    observed_outcomes = exposures * y1 + (1 - exposures) * y0

    # Create DataFrame with counterfactual outcomes
    df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            SIMULATED_OUTCOME_EXPOSED: y1,  # Y1
            SIMULATED_OUTCOME_CONTROL: y0,  # Y0
            SIMULATED_PROBAS_EXPOSED: p1,  # P1
            SIMULATED_PROBAS_CONTROL: p0,  # P0
            EXPOSURE_COL: exposures,
            PROBAS: np.where(exposures == 1, p1, p0),  # True observed probabilities
            OUTCOMES: observed_outcomes,
        }
    )

    # Save to file
    output_path = os.path.join(COUNTERFACTUAL_OUTCOMES_DIR, SIMULATION_RESULTS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Counterfactual outcomes saved to {output_path}")

    return df


def generate_outcome_predictions(subject_ids, cf_data, seed=42):
    """
    Generate outcome predictions data.
    This contains predicted probabilities of outcomes and actual outcomes,
    based on the true counterfactual data but with some noise added.
    """
    np.random.seed(seed)

    # Extract the true counterfactual data
    p0 = cf_data[SIMULATED_PROBAS_CONTROL].values
    p1 = cf_data[SIMULATED_PROBAS_EXPOSED].values
    exposures = cf_data[EXPOSURE_COL].values
    outcomes = cf_data[OUTCOMES].values

    # Add noise to create "predicted" probabilities
    noise_scale = 0.1
    predicted_p0 = np.clip(p0 + np.random.normal(0, noise_scale, len(p0)), 0.01, 0.99)
    predicted_p1 = np.clip(p1 + np.random.normal(0, noise_scale, len(p1)), 0.01, 0.99)

    # Create outcome probabilities for exposed and control groups
    outcome_probas = np.where(exposures == 1, predicted_p1, predicted_p0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            PID_COL: subject_ids,
            PROBAS: outcome_probas,
            PROBAS_EXPOSED: predicted_p1,  # Predicted outcome under treatment
            PROBAS_CONTROL: predicted_p0,  # Predicted outcome under control
            TARGETS: outcomes,
        }
    )

    # Save to file
    output_path = os.path.join(OUTCOME_PRED_DIR, CALIBRATED_PREDICTIONS_FILE)
    df.to_csv(output_path, index=False)
    print(f"Outcome predictions saved to {output_path}")

    return outcome_probas, outcomes


def main():
    """Generate all necessary data for testing estimate.py"""
    create_directories()

    # Generate data with 1000 samples
    subject_ids, ps_scores, exposures = generate_exposure_predictions(n_samples=1000)

    # Generate counterfactual data first (the ground truth)
    cf_data = generate_counterfactual_outcomes(subject_ids, ps_scores, exposures)

    # Now generate the predicted outcomes based on the counterfactual data
    _, _ = generate_outcome_predictions(subject_ids, cf_data)

    # Load the outcome data file for visualization
    outcome_df = pd.read_csv(
        os.path.join(OUTCOME_PRED_DIR, CALIBRATED_PREDICTIONS_FILE)
    )

    # Save visualization figures
    save_ps_distribution_figure(subject_ids, ps_scores, exposures)
    save_outcome_probas_figures(subject_ids, cf_data, outcome_df)
    save_comparison_figures(subject_ids, cf_data, outcome_df)
    # New distribution plot for predicted probabilities colored by exposure
    save_predicted_outcome_probas_distribution(cf_data, outcome_df)
    save_outcome_probas_by_exposure_figure(subject_ids, cf_data, outcome_df)
    print("All test data generated successfully!")


if __name__ == "__main__":
    main()
