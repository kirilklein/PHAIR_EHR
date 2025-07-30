import numpy as np
import json
import pandas as pd
from os.path import join


def f_save_weights(vocabulary: dict, weights, write_dir: str) -> None:
    """Saves the current weights to a file in a human-readable format."""
    vocab_size = len(vocabulary)
    all_indices = np.arange(vocab_size)

    def get_linear_weights_dict(weight_array, indices):
        """Creates a dict of code:weight for linear terms."""
        return {
            vocabulary[i]: round(weight_array[i], 4)
            for i in indices
            if not np.isclose(weight_array[i], 0)
        }

    def get_interaction_weights_dict(weight_matrix, indices):
        """Creates a dict of 'code1___code2':weight for interaction terms."""
        interaction_dict = {}
        # Find the row and column indices of non-zero elements in the upper triangle
        rows, cols = np.where(np.triu(weight_matrix, k=1) != 0)

        for i, j in zip(rows, cols):
            weight = weight_matrix[i, j]
            code1 = vocabulary[indices[i]]
            code2 = vocabulary[indices[j]]
            interaction_dict[f"{code1}___{code2}"] = round(weight, 4)
        return interaction_dict

    state_to_save = {
        "vocabulary_size": len(vocabulary),
        "weights": {
            "exposure": {},
            "_outcomes_shared": {},
        },
    }

    # Linear weights
    state_to_save["weights"]["exposure"]["linear"] = get_linear_weights_dict(
        weights["exposure"]["linear"], all_indices
    )
    state_to_save["weights"]["_outcomes_shared"]["linear"] = get_linear_weights_dict(
        weights["_outcomes_shared"]["linear"], all_indices
    )

    # Interaction weights
    state_to_save["weights"]["exposure"]["interaction"] = get_interaction_weights_dict(
        weights["exposure"]["interaction_joint"],
        all_indices,
    )
    state_to_save["weights"]["_outcomes_shared"]["interaction"] = (
        get_interaction_weights_dict(
            weights["_outcomes_shared"]["interaction_joint"],
            all_indices,
        )
    )

    with open(join(write_dir, "simulation_weights.json"), "w") as f:
        json.dump(state_to_save, f, indent=4)


def debug_patient_history(
    pids: np.ndarray,
    patient_history_matrix: np.ndarray,
    weights: dict,
    vocabulary: list,
    logger,
):
    """
    Identifies patients with a near-zero history effect and prints a
    detailed breakdown of their codes and associated weights for debugging.
    """
    logger.info("--- Debugging patient history effects ---")
    exposure_weights = weights["exposure"]

    linear_effects = patient_history_matrix @ exposure_weights["linear"]
    joint_weights = exposure_weights["interaction_joint"]
    interaction_effects = (
        np.einsum(
            "ij,jk,ik->i", patient_history_matrix, joint_weights, patient_history_matrix
        )
        / 2
    )

    total_history_effect = linear_effects + interaction_effects
    problematic_patient_indices = np.where(
        np.isclose(total_history_effect, 0, atol=0.02)
    )[0]

    if len(problematic_patient_indices) > 0:
        logger.info(
            f"Found {len(problematic_patient_indices)} patients with near-zero history effect."
        )

        # Loop through the first few problematic patients for detailed analysis
        for i, pid_idx in enumerate(problematic_patient_indices[:5]):
            pid = pids[pid_idx]
            logger.info(f"\n--- Analysis for Patient {pid} (index {pid_idx}) ---")

            patient_vec = patient_history_matrix[pid_idx, :]
            present_code_indices = np.where(patient_vec > 0)[0]

            if present_code_indices.size == 0:
                logger.info("  Patient has no relevant codes in their history vector.")
                continue

            # ✅ START: Nicely formatted weight breakdown
            # Create a DataFrame to display weights clearly
            weights_df = pd.DataFrame(
                {
                    "Code": [vocabulary[i] for i in present_code_indices],
                    "Temporal Weight": patient_vec[present_code_indices],
                    "Linear Weight": exposure_weights["linear"][present_code_indices],
                }
            )

            # Calculate the final contribution of each code to the linear effect
            weights_df["Effective Contribution"] = (
                weights_df["Temporal Weight"] * weights_df["Linear Weight"]
            )

            # Filter for codes that actually have a linear weight to reduce noise
            active_weights_df = weights_df[weights_df["Linear Weight"] != 0].copy()

            if not active_weights_df.empty:
                # Sort by the most impactful contribution for easy debugging
                active_weights_df.sort_values(
                    by="Effective Contribution", key=abs, ascending=False, inplace=True
                )
                logger.info(f"  Linear weight contributions for patient {pid}:")
                # Use to_string() for nice table formatting in the log
                # The extra newline improves readability in most log viewers
                logger.info("\n" + active_weights_df.to_string(index=False))
            else:
                logger.info(
                    "  None of this patient's codes have a non-zero linear weight."
                )
            # ✅ END: Nicely formatted weight breakdown

            # Simplified check for interactions for brevity in logging
            interaction_slice = joint_weights[
                np.ix_(present_code_indices, present_code_indices)
            ]
            if np.any(interaction_slice):
                logger.info("  Patient also has codes with active interaction weights.")
    else:
        logger.info("All patients have a non-zero history effect.")
    logger.info("--- End of debugging ---")


def analyze_peak_patients(
    p_exposure: np.ndarray,
    ages: np.ndarray,
    patient_history_matrix: np.ndarray,
    vocabulary: list,
    weights: dict,
    logger,
    peak_width: float = 0.02,
    num_bins: int = 50,
):
    """
    Automatically detects the main probability peak and analyzes the
    characteristics, effect components, and code similarity for patients within it.
    """
    if p_exposure.size == 0:
        logger.info("p_exposure array is empty, skipping peak analysis.")
        return

    # --- 1. Automatically Detect the Peak ---
    counts, bin_edges = np.histogram(p_exposure, bins=num_bins, range=(0, 1))
    peak_bin_index = np.argmax(counts)
    peak_center = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2

    logger.info(
        f"\n--- Analyzing characteristics in auto-detected peak at ~{peak_center:.3f} ---"
    )

    peak_min = peak_center - (peak_width / 2)
    peak_max = peak_center + (peak_width / 2)
    peak_indices = np.where((p_exposure >= peak_min) & (p_exposure <= peak_max))[0]

    if peak_indices.size == 0:
        logger.info(
            f"  No patients found in the range [{peak_min:.3f}, {peak_max:.3f}]."
        )
        return

    logger.info(f"  Found {peak_indices.size} patients in the peak.")

    # --- 2. Analyze Ages and Effect Components ---
    peak_ages = ages[peak_indices]
    age_series = pd.Series(peak_ages)
    logger.info("  Age distribution for patients in peak:")
    logger.info("\n" + age_series.describe().to_string())

    # (Effect component analysis from before...)

    # --- 3. Analyze Common Medical Codes and Their Penetration ---
    peak_vectors = patient_history_matrix[peak_indices, :]
    code_prevalence = peak_vectors.sum(axis=0)
    top_10_code_indices = np.argsort(code_prevalence)[-10:][::-1]

    # Calculate penetration percentage for top codes
    binary_peak_vectors = peak_vectors > 0
    penetration_counts = binary_peak_vectors[:, top_10_code_indices].sum(axis=0)
    penetration_percent = (penetration_counts / peak_indices.size) * 100

    top_codes_df = pd.DataFrame(
        {
            "Rank": range(1, 11),
            "Code": [vocabulary[i] for i in top_10_code_indices],
            "Penetration": [f"{p:.1f}%" for p in penetration_percent],
            "Prevalence Score": code_prevalence[top_10_code_indices],
        }
    )

    logger.info("\n  Top 10 most common/impactful codes for patients in peak:")
    logger.info("\n" + top_codes_df.to_string(index=False))

    # --- ✅ NEW: 4. Display Code Combinations for a Sample of Patients ---
    logger.info("\n  Code profiles for a sample of patients in the peak:")
    num_samples = min(10, peak_indices.size)  # Sample up to 10 patients
    # Use np.random.choice for a random sample, or slice for the first few
    sample_patient_indices = np.random.choice(
        peak_indices, size=num_samples, replace=False
    )

    for i, patient_idx in enumerate(sample_patient_indices):
        # Get the patient's row from the main history matrix
        patient_vector = patient_history_matrix[patient_idx, :]
        # Find the indices of all codes present for this patient
        non_zero_code_indices = np.where(patient_vector > 0)[0]

        # Get the corresponding code names from the vocabulary
        patient_codes = [vocabulary[code_idx] for code_idx in non_zero_code_indices]

        # Log the patient's original index and their list of codes
        log_message = (
            f"    - Sample Patient #{i + 1} (Original Index: {patient_idx}):\n"
            f"      Codes ({len(patient_codes)}): {patient_codes}"
        )
        logger.info(log_message)
    # --- End of New Section ---

    logger.info("--- End of peak analysis ---")
