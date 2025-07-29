import json
import logging
from os.path import join
from typing import Dict, Tuple, Union, Set

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from corebehrt.constants.causal.data import (
    CONTROL_PID_COL,
    EXPOSED_PID_COL,
    EXPOSURE_COL,
    OUTCOME_COL,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
)
from corebehrt.constants.causal.paths import (
    COUNTERFACTUALS_FILE,
    INDEX_DATE_MATCHING_FILE,
)
from corebehrt.constants.data import ABSPOS_COL, CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.simulation.config_all import (
    ExposureConfig,
    OutcomeConfig,
    SimulationConfig,
)

logger = logging.getLogger("simulate")


class CausalSimulator:
    """
    Simulates exposure and outcome events for patient cohorts using vectorized
    operations and builds its vocabulary and weights dynamically as it sees new data shards.
    """

    def __init__(self, config: SimulationConfig, seed: int = 42):
        self.config = config
        self.index_date = config.index_date
        self.rng = np.random.default_rng(seed)
        self._initialize_state()
        self.sampling_func = self.rng.normal

    def _initialize_state(self):
        """Initializes empty, stateful attributes for the simulator."""
        self.code_to_idx = {}
        self.vocabulary = []
        self.weights = {
            "exposure": {"linear": np.array([])},
            "_outcomes_shared": {"linear": np.array([])},
        }
        self.linear_code_indices = None
        self.interaction_code_indices = None
        self.interaction_idx_map = None

    def _update_vocabulary_and_weights(self, codes_in_shard: Set[str]):
        """
        Updates vocabulary and weights for new codes. Finalizes linear and
        interaction subsets once the vocabulary is sufficiently large.
        """
        new_codes = list(codes_in_shard - set(self.vocabulary))
        if not new_codes:
            return

        logger.info(f"Discovered {len(new_codes)} new codes. Updating weights...")

        n_new = len(new_codes)
        linear_config = self.config.simulation_model.linear

        # Determine if we should sample non-zero weights or just append zeros
        should_sample_non_zero = self.linear_code_indices is None

        if should_sample_non_zero:
            new_exposure_linear = self.rng.normal(
                linear_config.mean, linear_config.scale, n_new
            )
            new_outcomes_linear = self.rng.normal(
                linear_config.mean, linear_config.scale, n_new
            )
        else:
            new_exposure_linear = np.zeros(n_new)
            new_outcomes_linear = np.zeros(n_new)

        # Append the new weights (either sampled or zero)
        self.weights["exposure"]["linear"] = np.concatenate(
            [self.weights["exposure"]["linear"], new_exposure_linear]
        )
        self.weights["_outcomes_shared"]["linear"] = np.concatenate(
            [self.weights["_outcomes_shared"]["linear"], new_outcomes_linear]
        )

        # Update vocabulary AFTER preparing weights
        for code in new_codes:
            self.code_to_idx[code] = len(self.vocabulary)
            self.vocabulary.append(code)

        linear_subset_size = self.config.simulation_model.linear_subset_size
        if (
            self.linear_code_indices is None
            and len(self.vocabulary) >= linear_subset_size
        ):
            logger.info(
                f"Vocabulary size ({len(self.vocabulary)}) reached linear subset threshold ({linear_subset_size}). Finalizing linear weights."
            )

            # Choose the codes that will have non-zero linear effects
            self.linear_code_indices = self.rng.choice(
                len(self.vocabulary), size=linear_subset_size, replace=False
            )

            # Create a mask to zero-out weights for non-selected codes
            mask = np.zeros(len(self.vocabulary), dtype=bool)
            mask[self.linear_code_indices] = True

            # Apply the mask to the existing weight vectors
            self.weights["exposure"]["linear"] *= mask
            self.weights["_outcomes_shared"]["linear"] *= mask

        # Finalize interaction weights (logic is unchanged)
        interaction_subset_size = self.config.simulation_model.interaction_subset_size
        if (
            self.interaction_code_indices is None
            and len(self.vocabulary) >= interaction_subset_size
        ):
            logger.info(
                f"Vocabulary size ({len(self.vocabulary)}) reached interaction subset threshold ({interaction_subset_size}). Finalizing interaction weights."
            )

            self.interaction_code_indices = self.rng.choice(
                len(self.vocabulary), size=interaction_subset_size, replace=False
            )
            self.interaction_idx_map = {
                original_idx: new_idx
                for new_idx, original_idx in enumerate(self.interaction_code_indices)
            }
            interaction_config = self.config.simulation_model.interaction

            joint_exp = self.sampling_func(
                interaction_config.mean,
                interaction_config.scale,
                (interaction_subset_size, interaction_subset_size),
            )
            self.weights["exposure"]["interaction_joint"] = joint_exp

            joint_out = self.sampling_func(
                interaction_config.mean,
                interaction_config.scale,
                (interaction_subset_size, interaction_subset_size),
            )
            self.weights["_outcomes_shared"]["interaction_joint"] = joint_out

            self.save_weights()

    def save_weights(self) -> None:
        """Saves the current weights and vocabulary state to a file."""

        # This function is now more important for saving state
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return np.round(obj, 4).tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        serializable_weights = to_serializable(self.weights)
        state_to_save = {"vocabulary": self.vocabulary, "weights": serializable_weights}
        if self.interaction_code_indices is not None:
            state_to_save["interaction_codes"] = [
                self.vocabulary[i] for i in self.interaction_code_indices
            ]

        with open(
            join(self.config.paths.outcomes, "simulation_weights.json"), "w"
        ) as f:
            json.dump(state_to_save, f, indent=4)

    def simulate_dataset(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Simulates exposures and outcomes for a cohort shard, updating vocabulary on the fly.
        """
        # Step 1: Update vocabulary and weights based on codes in the current shard
        codes_in_shard = set(df[CONCEPT_COL].unique())
        self._update_vocabulary_and_weights(codes_in_shard)

        # Step 2: Prepare data using the full known vocabulary
        logger.info("Preparing patient history matrix...")
        history_df = df[df[TIMESTAMP_COL] <= self.index_date]
        all_pids = df[PID_COL].unique()

        patient_history_matrix, pids = self._get_patient_history_matrix(
            history_df, all_pids
        )
        n_patients = len(pids)

        if n_patients == 0:
            return {}

        # The rest of the simulation logic remains the same as the previous vectorized version
        logger.info(f"Simulating for {n_patients} patients...")
        # ... (The rest of this method is identical to the previous answer's simulate_dataset) ...
        confounder_exposure_effect, confounder_outcome_effects = (
            self._simulate_unobserved_confounder_effects(n_patients)
        )

        p_exposure = self._calculate_probabilities_vectorized(
            "exposure",
            self.config.exposure,
            patient_history_matrix,
            additional_logit_effect=confounder_exposure_effect,
        )
        is_exposed = self.rng.binomial(1, p_exposure).astype(bool)

        ite_records = {PID_COL: pids}
        cf_records = {PID_COL: pids, EXPOSURE_COL: is_exposed.astype(int)}
        all_factual_events = []

        for outcome_name, outcome_cfg in self.config.outcomes.items():
            confounder_effect = confounder_outcome_effects.get(
                outcome_name, np.zeros(n_patients)
            )

            p_if_treated = self._calculate_probabilities_vectorized(
                outcome_name,
                outcome_cfg,
                patient_history_matrix,
                is_exposed=True,
                additional_logit_effect=confounder_effect,
            )
            p_if_control = self._calculate_probabilities_vectorized(
                outcome_name,
                outcome_cfg,
                patient_history_matrix,
                is_exposed=False,
                additional_logit_effect=confounder_effect,
            )

            ite_records[f"ite_{outcome_name}"] = p_if_treated - p_if_control
            outcome_exposed = self.rng.binomial(1, p_if_treated)
            outcome_control = self.rng.binomial(1, p_if_control)
            factual_outcome = np.where(is_exposed, outcome_exposed, outcome_control)

            cf_records[f"{OUTCOME_COL}_{outcome_name}"] = factual_outcome
            cf_records[f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"] = outcome_exposed
            cf_records[f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"] = outcome_control
            cf_records[f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"] = p_if_treated
            cf_records[f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"] = p_if_control

            outcome_time = self.index_date + pd.Timedelta(days=outcome_cfg.run_in_days)
            patients_with_outcome = pids[factual_outcome == 1]
            if len(patients_with_outcome) > 0:
                events = pd.DataFrame(
                    {
                        PID_COL: patients_with_outcome,
                        TIMESTAMP_COL: outcome_time,
                        CONCEPT_COL: outcome_name,
                    }
                )
                all_factual_events.append(events)

        patients_with_exposure = pids[is_exposed]
        if len(patients_with_exposure) > 0:
            events = pd.DataFrame(
                {
                    PID_COL: patients_with_exposure,
                    TIMESTAMP_COL: self.index_date,
                    CONCEPT_COL: EXPOSURE_COL,
                }
            )
            all_factual_events.append(events)

        output_dfs = {}
        if all_factual_events:
            events_df = pd.concat(all_factual_events, ignore_index=True)
            events_df[ABSPOS_COL] = get_hours_since_epoch(events_df[TIMESTAMP_COL])
            for code, group in events_df.groupby(CONCEPT_COL):
                output_dfs[str(code)] = group[
                    [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
                ].copy()

        output_dfs["ite"] = pd.DataFrame(ite_records)
        output_dfs[COUNTERFACTUALS_FILE.split(".")[0]] = pd.DataFrame(cf_records)

        if EXPOSURE_COL in output_dfs:
            output_dfs[INDEX_DATE_MATCHING_FILE.split(".")[0]] = (
                self._create_index_date_matching_df(output_dfs[EXPOSURE_COL], all_pids)
            )
        return output_dfs

    def _calculate_probabilities_vectorized(
        self,
        event_name: str,
        event_cfg: Union[OutcomeConfig, ExposureConfig],
        history_matrix: np.ndarray,
        is_exposed: bool = False,
        additional_logit_effect: np.ndarray = 0.0,
    ) -> np.ndarray:
        """Calculates event probabilities for the entire cohort in a vectorized manner."""
        n_patients = history_matrix.shape[0]

        if event_name == "exposure":
            weights = self.weights["exposure"]
        else:  # It's an outcome
            weights = self.weights["_outcomes_shared"]

        logit_p_array = np.full(n_patients, logit(event_cfg.p_base))

        # 1. Linear effects (always applied)
        logit_p_array += history_matrix @ weights["linear"]

        # 2. Interaction effects (applied ONLY if they have been finalized)
        if "interaction_joint" in weights:
            interaction_history = history_matrix[:, self.interaction_code_indices]
            joint_weights = weights["interaction_joint"]

            n_interactions = len(self.interaction_code_indices)
            for i in range(n_interactions):
                for j in range(i + 1, n_interactions):
                    i_present = interaction_history[:, i]
                    j_present = interaction_history[:, j]

                    logit_p_array += (i_present & j_present) * joint_weights[i, j]
        # 3. Exposure effect (for outcomes)
        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p_array += event_cfg.exposure_effect

        # 4. Unobserved confounder effect
        logit_p_array += additional_logit_effect

        return expit(logit_p_array)

    # _get_patient_history_matrix, _simulate_unobserved_confounder_effects,
    # and _create_index_date_matching_df remain IDENTICAL to the previous answer.
    # They are included here for completeness.

    def _get_patient_history_matrix(
        self, history_df: pd.DataFrame, all_pids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a multi-hot encoded matrix of patient histories against the full known vocabulary."""
        history_crosstab = pd.crosstab(history_df[PID_COL], history_df[CONCEPT_COL])
        # Reindex against the full, dynamically grown vocabulary
        history_crosstab = history_crosstab.reindex(
            index=all_pids, columns=self.vocabulary, fill_value=0
        )
        return history_crosstab.values.astype(bool), history_crosstab.index.to_numpy()

    def _simulate_unobserved_confounder_effects(
        self, n_patients: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates unobserved confounder effects for a cohort."""
        confounder_cfg = self.config.unobserved_confounder
        if not confounder_cfg:
            return 0.0, {}

        has_confounder = self.rng.binomial(
            1, confounder_cfg.p_occurrence, n_patients
        ).astype(bool)
        exposure_effect = np.where(has_confounder, confounder_cfg.exposure_effect, 0.0)

        outcome_effects = {}
        if confounder_cfg.outcome_effects:
            for outcome_name, effect in confounder_cfg.outcome_effects.items():
                outcome_effects[outcome_name] = np.where(has_confounder, effect, 0.0)

        return exposure_effect, outcome_effects

    def _create_index_date_matching_df(
        self, exposure_df: pd.DataFrame, all_pids: list
    ) -> pd.DataFrame:
        """This function remains the same as the logic is already cohort-based."""
        exposed_pids = exposure_df[PID_COL].unique()

        if len(exposed_pids) == 0:
            return pd.DataFrame(
                columns=[CONTROL_PID_COL, EXPOSED_PID_COL, TIMESTAMP_COL, ABSPOS_COL]
            )

        control_pids = list(set(all_pids) - set(exposed_pids))
        if not control_pids:
            return pd.DataFrame(
                columns=[CONTROL_PID_COL, EXPOSED_PID_COL, TIMESTAMP_COL, ABSPOS_COL]
            )

        rng = np.random.default_rng(42)
        matched_exposed_pids = rng.choice(
            exposed_pids, size=len(control_pids), replace=True
        )

        match_df = pd.DataFrame(
            {CONTROL_PID_COL: control_pids, EXPOSED_PID_COL: matched_exposed_pids}
        )

        exposure_info = (
            exposure_df[[PID_COL, TIMESTAMP_COL, ABSPOS_COL]]
            .drop_duplicates(subset=[PID_COL])
            .set_index(PID_COL)
        )

        match_df = match_df.merge(
            exposure_info, left_on=EXPOSED_PID_COL, right_index=True
        )
        return match_df[[CONTROL_PID_COL, EXPOSED_PID_COL, TIMESTAMP_COL, ABSPOS_COL]]
