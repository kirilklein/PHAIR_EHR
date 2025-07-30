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
    history_df: pd.DataFrame,
    pids: np.ndarray,
    patient_history_matrix: np.ndarray,
    weights,
    vocabulary,
    logger,
):
    """
    Identifies patients with no calculated history effect and prints their
    history for debugging purposes.
    """
    logger.info("--- Debugging patient history effects ---")
    n_patients = len(pids)
    weights = weights["exposure"]

    linear_effects = patient_history_matrix @ weights["linear"]

    interaction_effects = np.zeros(n_patients)
    joint_weights = weights["interaction_joint"]
    interaction_history = patient_history_matrix
    interaction_effects = (
        np.einsum(
            "ij,jk,ik->i", interaction_history, joint_weights, interaction_history
        )
        / 2
    )

    total_history_effect = linear_effects + interaction_effects
    problematic_patient_indices = np.where(
        np.isclose(total_history_effect, 0, atol=0.02)
    )[0]

    if len(problematic_patient_indices) > 0:
        logger.info(
            f"Found {len(problematic_patient_indices)} patients with zero history effect."
        )

        for i, pid_idx in enumerate(problematic_patient_indices[:5]):
            pid = pids[pid_idx]
            logger.info(
                f"--- History for patient {pid} (index {pid_idx}) with zero effect ---"
            )

            patient_vec = patient_history_matrix[pid_idx, :]
            present_code_indices = np.where(patient_vec > 0)[0]

            if len(present_code_indices) == 0:
                logger.info(
                    "  Patient has no codes from the vocabulary in their history."
                )
            else:
                present_codes = [vocabulary[i] for i in present_code_indices]
                logger.info(f"  Patient has codes: {present_codes}")

                active_linear_indices = np.where(
                    weights["linear"][present_code_indices] != 0
                )[0]
                if len(active_linear_indices) > 0:
                    logger.info(
                        f"  Of which, these have linear weights: {[present_codes[i] for i in active_linear_indices]}"
                    )
                else:
                    logger.info("  None of these codes have linear weights.")

                # Simplified check for interactions for brevity in logging
                interaction_slice = joint_weights[
                    np.ix_(present_code_indices, present_code_indices)
                ]
                if np.any(interaction_slice):
                    logger.info("  Patient has codes with active interaction weights.")
                else:
                    logger.info(
                        "  None of the patient's codes have interaction weights with each other."
                    )

            patient_history_df = history_df[history_df[PID_COL] == pid]
            logger.info(
                f"  Raw history for patient {pid} ({len(patient_history_df)} events) with total history effect {total_history_effect[pid_idx]:.4f}:\n"
                f"{patient_history_df[[CONCEPT_COL, TIMESTAMP_COL]].to_string()}"
            )
    else:
        logger.info("All patients have a non-zero history effect.")
    logger.info("--- End of debugging ---")
