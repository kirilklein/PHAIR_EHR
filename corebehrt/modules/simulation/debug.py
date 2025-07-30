import numpy as np
import json
import pandas as pd
from os.path import join
from corebehrt.constants.data import PID_COL, CONCEPT_COL, TIMESTAMP_COL


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
