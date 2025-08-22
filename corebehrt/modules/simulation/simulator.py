import logging
from typing import Dict, Set, Tuple, Union

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
from corebehrt.constants.data import (
    ABSPOS_COL,
    BIRTH_CODE,
    CONCEPT_COL,
    DEATH_CODE,
    PID_COL,
    TIMESTAMP_COL,
)
from corebehrt.functional.utils.filter import safe_control_pids
from corebehrt.functional.utils.time import (
    get_hours_since_epoch,
)
from corebehrt.modules.simulation.config import (
    ExposureConfig,
    OutcomeConfig,
    SimulationConfig,
)
from corebehrt.modules.simulation.debug import (
    analyze_peak_patients,
    debug_patient_history,
    f_save_weights,
)
from corebehrt.modules.simulation.plot import plot_hist, plot_probability_distributions

logger = logging.getLogger("simulate")

WEIGHT_COL = "weight"


class CausalSimulator:
    """
    Simulates exposure and outcome events for patient cohorts using vectorized
    operations and builds its vocabulary and weights dynamically as it sees new data shards.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.index_date = config.index_date
        self.rng = np.random.default_rng(config.seed)
        self._initialize_state()
        self.sampling_func = self.rng.normal
        self.debug = config.debug

    def _initialize_state(self):
        """Initializes empty, stateful attributes for the simulator."""
        # The latent dimension for low-rank factorization. Make this configurable.
        self.latent_dim = 100

        self.code_to_idx = {}
        self.vocabulary = []
        self.weights = {
            "exposure": {
                "linear": np.array([]),
                "interaction_U": np.zeros((0, self.latent_dim)),
                "interaction_K": np.zeros((0, self.latent_dim)),
            },
            "_outcomes_shared": {
                "linear": np.array([]),
                "interaction_U": np.zeros((0, self.latent_dim)),
                "interaction_K": np.zeros((0, self.latent_dim)),
            },
        }
        self.linear_code_indices = None
        self.interaction_code_indices = None
        self.interaction_idx_map = None
        self.finalized = False

    def _update_vocabulary_and_weights(self, codes_in_shard: Set[str]):
        """
        Updates vocabulary and weights using low-rank factorization for interactions.
        Weights for new codes are sampled on the fly and appended, which is much
        faster than resizing a full V x V matrix.
        """
        logger.info("Checking for new codes to update vocabulary and weights...")
        new_codes = list(codes_in_shard - set(self.vocabulary))

        if not new_codes:
            logger.info("No new codes found. Vocabulary and weights are unchanged.")
            return

        logger.info(
            f"Discovered {len(new_codes)} new codes. Updating vocabulary and weights."
        )
        n_new = len(new_codes)

        # Grow vocabulary
        for code in new_codes:
            self.code_to_idx[code] = len(self.vocabulary)
            self.vocabulary.append(code)

        # Grow and sample linear weights (no change here)
        linear_config = self.config.simulation_model.linear
        for key in ["exposure", "_outcomes_shared"]:
            new_weights = self.sampling_func(
                linear_config.mean, linear_config.scale, n_new
            )
            is_zero = self.rng.random(n_new) < linear_config.sparsity_factor
            new_weights[is_zero] = 0
            if linear_config.max_weight is not None:
                np.clip(
                    new_weights,
                    -linear_config.max_weight,
                    linear_config.max_weight,
                    out=new_weights,
                )
            self.weights[key]["linear"] = np.append(
                self.weights[key]["linear"], new_weights
            )

        # Grow and sample interaction factor matrices (U and K)
        interaction_config = self.config.simulation_model.interaction
        for key in ["exposure", "_outcomes_shared"]:
            # Sample new rows for the U and K factor matrices
            new_U_rows = self.sampling_func(
                interaction_config.mean,
                interaction_config.scale,
                (n_new, self.latent_dim),
            )
            new_K_rows = self.sampling_func(
                interaction_config.mean,
                interaction_config.scale,
                (n_new, self.latent_dim),
            )

            # Apply sparsity
            is_zero_U = (
                self.rng.random((n_new, self.latent_dim))
                < interaction_config.sparsity_factor
            )
            is_zero_K = (
                self.rng.random((n_new, self.latent_dim))
                < interaction_config.sparsity_factor
            )
            new_U_rows[is_zero_U] = 0
            new_K_rows[is_zero_K] = 0

            # Apply clipping if configured
            if interaction_config.max_weight is not None:
                np.clip(
                    new_U_rows,
                    -interaction_config.max_weight,
                    interaction_config.max_weight,
                    out=new_U_rows,
                )
                np.clip(
                    new_K_rows,
                    -interaction_config.max_weight,
                    interaction_config.max_weight,
                    out=new_K_rows,
                )

            # Efficiently append new rows to the existing factor matrices
            self.weights[key]["interaction_U"] = np.vstack(
                [self.weights[key]["interaction_U"], new_U_rows]
            )
            self.weights[key]["interaction_K"] = np.vstack(
                [self.weights[key]["interaction_K"], new_K_rows]
            )

    def simulate_dataset(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Simulates exposures and outcomes for a cohort shard, updating vocabulary on the fly.
        """
        history_df, initial_pids = self._filter_initial_history(df)
        if history_df.empty:
            return {}

        ages = self._calculate_ages(history_df, initial_pids)
        history_df = self._filter_by_codes(history_df)
        if history_df.empty:
            logger.info(
                "No eligible patients remaining after code filtering. Skipping simulation."
            )
            return {}

        patient_history_matrix, pids, final_ages = self._prepare_simulation_inputs(
            history_df, ages
        )
        if len(pids) == 0:
            return {}

        n_patients = len(pids)
        logger.info(f"Simulating effects for {n_patients} patients...")

        confounder_exposure_effect, confounder_outcome_effects = (
            self._simulate_unobserved_confounder_effects(n_patients)
        )
        is_exposed, p_exposure = self._simulate_exposure(
            patient_history_matrix, final_ages, pids, confounder_exposure_effect
        )

        ite_records, cf_records, all_factual_events, all_probas_for_plotting = (
            self._simulate_outcomes(
                patient_history_matrix,
                final_ages,
                pids,
                is_exposed,
                confounder_outcome_effects,
            )
        )

        if np.any(is_exposed):
            all_factual_events.append(self._create_exposure_events(pids[is_exposed]))

        output_dfs = self._package_results(
            pids, ite_records, cf_records, all_factual_events, all_probas_for_plotting
        )
        return output_dfs

    def _filter_initial_history(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Filters the initial dataframe to get patients with history before the index date."""
        history_df = df[df[TIMESTAMP_COL] <= self.index_date].copy()

        all_pids_in_shard = df[PID_COL].unique()
        pids_with_history = history_df[PID_COL].unique()

        removed_no_history = len(all_pids_in_shard) - len(pids_with_history)
        if removed_no_history > 0:
            logger.info(f"Removed {removed_no_history} patients without history.")

        if len(pids_with_history) == 0:
            logger.info("No patients with history in this shard. Skipping simulation.")
            return pd.DataFrame(), np.array([])

        dead_pids = history_df[history_df[CONCEPT_COL] == DEATH_CODE][PID_COL].unique()
        if len(dead_pids) > 0:
            logger.info(f"Removing {len(dead_pids)} patients with DOD record.")
            history_df = history_df[~history_df[PID_COL].isin(dead_pids)]

        all_pids = history_df[PID_COL].unique()
        if len(all_pids) == 0:
            logger.info(
                "No eligible patients remaining in this shard. Skipping simulation."
            )
            return pd.DataFrame(), np.array([])

        return history_df, all_pids

    def _calculate_ages(
        self, history_df: pd.DataFrame, all_pids: np.ndarray
    ) -> pd.Series:
        """Calculates age for each patient."""
        dob_events = history_df[history_df[CONCEPT_COL] == BIRTH_CODE]
        patient_dobs = dob_events.groupby(PID_COL)[TIMESTAMP_COL].first()
        ages = pd.Series(np.nan, index=all_pids, dtype=float)
        if not patient_dobs.empty:
            calculated_ages = (self.index_date - patient_dobs).dt.days / 365.25
            ages.update(calculated_ages)

        mean_age = ages.mean()
        if np.isnan(mean_age):
            logger.warning(
                "No patients with DOB found. Age effect cannot be calculated accurately. Using default age 40 for all."
            )
            mean_age = 40
        else:
            logger.info(
                f"Mean age of cohort: {mean_age:.2f} years. Using this for patients without DOB."
            )

        return ages.fillna(mean_age)

    def _filter_by_codes(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Filters patients by code prefixes and minimum number of codes."""
        if self.config.include_code_prefixes:
            logger.info(
                f"Filtering codes based on prefixes: {self.config.include_code_prefixes}"
            )
            history_df = history_df[
                history_df[CONCEPT_COL].str.startswith(
                    tuple(self.config.include_code_prefixes)
                )
            ].copy()

        if self.config.min_num_codes > 1:
            min_codes = self.config.min_num_codes
            logger.info(
                f"Filtering out patients with fewer than {min_codes} unique codes."
            )

            code_counts = history_df.groupby(PID_COL)[CONCEPT_COL].nunique()
            pids_to_keep = code_counts[code_counts >= min_codes].index

            num_removed = len(history_df[PID_COL].unique()) - len(pids_to_keep)
            if num_removed > 0:
                logger.info(
                    f"Removed {num_removed} patients with fewer than {min_codes} unique codes."
                )
                history_df = history_df[history_df[PID_COL].isin(pids_to_keep)].copy()
            else:
                logger.info(
                    "No patients were removed by the minimum code count filter."
                )

        return history_df

    def _prepare_simulation_inputs(
        self, history_df: pd.DataFrame, ages: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepares the inputs required for the simulation, including the patient history matrix."""
        all_pids = history_df[PID_COL].unique()
        if len(all_pids) == 0:
            return np.array([]), np.array([]), np.array([])

        logger.info(f"Number of patients with history remaining: {len(all_pids)}")

        codes_in_shard = set(history_df[CONCEPT_COL].unique())
        self._update_vocabulary_and_weights(codes_in_shard)

        logger.info("Getting patient history matrix...")
        patient_history_matrix, pids = self._get_patient_history_matrix(
            history_df, all_pids
        )
        logger.info("Got patient history matrix.")

        final_ages = ages.loc[pids].values
        n_patients = len(pids)

        if n_patients > 0:
            logger.info(f"Number of patients with history: {n_patients}")
            logger.info(
                f"Size of patient history matrix: {patient_history_matrix.shape}"
            )
            logger.info(
                f"Num of non-zero entries in patient history matrix: {patient_history_matrix.sum()}"
            )
            logger.info(
                f"Average number of non-zero entries per patient: {patient_history_matrix.sum() / n_patients}"
            )

        return patient_history_matrix, pids, final_ages

    def _simulate_exposure(
        self,
        patient_history_matrix: np.ndarray,
        final_ages: np.ndarray,
        pids: np.ndarray,
        confounder_exposure_effect: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates exposure for the cohort."""
        logger.info("Simulating exposure probabilities...")
        p_exposure = self._calculate_probabilities_vectorized(
            "exposure",
            self.config.exposure,
            patient_history_matrix,
            ages=final_ages,
            additional_logit_effect=confounder_exposure_effect,
        )

        if self.debug:
            debug_patient_history(
                pids, patient_history_matrix, self.weights, self.vocabulary, logger
            )
            analyze_peak_patients(
                p_exposure,
                final_ages,
                patient_history_matrix,
                self.vocabulary,
                self.weights,
                logger,
            )

        logger.info("Plotting histogram of exposure probabilities...")
        plot_hist(p_exposure, self.config.paths.outcomes)
        is_exposed = self.rng.binomial(1, p_exposure).astype(bool)

        return is_exposed, p_exposure

    def _simulate_outcomes(
        self,
        patient_history_matrix: np.ndarray,
        final_ages: np.ndarray,
        pids: np.ndarray,
        is_exposed: np.ndarray,
        confounder_outcome_effects: Dict[str, np.ndarray],
    ) -> Tuple[Dict, Dict, list, Dict]:
        """Simulates outcomes for the cohort."""
        n_patients = len(pids)
        ite_records = {PID_COL: pids}
        cf_records = {PID_COL: pids, EXPOSURE_COL: is_exposed.astype(int)}
        all_factual_events = []
        all_probas_for_plotting = {}

        logger.info("Simulating outcome probabilities...")
        for outcome_name, outcome_cfg in self.config.outcomes.items():
            logger.info(f":Simulating outcome probabilities for {outcome_name}...")
            confounder_effect = confounder_outcome_effects.get(
                outcome_name, np.zeros(n_patients)
            )

            logger.info("Calculating probabilities under exposure...")
            p_if_treated = self._calculate_probabilities_vectorized(
                outcome_name,
                outcome_cfg,
                patient_history_matrix,
                ages=final_ages,
                is_exposed=True,
                additional_logit_effect=confounder_effect,
            )

            logger.info("Calculating probabilities under control...")
            p_if_control = self._calculate_probabilities_vectorized(
                outcome_name,
                outcome_cfg,
                patient_history_matrix,
                ages=final_ages,
                is_exposed=False,
                additional_logit_effect=confounder_effect,
            )

            all_probas_for_plotting[outcome_name] = {
                "P1": p_if_treated,
                "P0": p_if_control,
            }
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

        return ite_records, cf_records, all_factual_events, all_probas_for_plotting

    def _create_exposure_events(self, exposed_pids: np.ndarray) -> pd.DataFrame:
        """Creates a DataFrame for exposure events."""
        return pd.DataFrame(
            {
                PID_COL: exposed_pids,
                TIMESTAMP_COL: self.index_date,
                CONCEPT_COL: EXPOSURE_COL,
            }
        )

    def _package_results(
        self,
        pids: np.ndarray,
        ite_records: dict,
        cf_records: dict,
        all_factual_events: list,
        all_probas_for_plotting: dict,
    ) -> Dict[str, pd.DataFrame]:
        """Packages simulation results into a dictionary of DataFrames."""
        logger.info("Plotting probability distributions...")
        plot_probability_distributions(
            all_probas_for_plotting, self.config.paths.outcomes
        )

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
                self._create_index_date_matching_df(output_dfs[EXPOSURE_COL], pids)
            )

        return output_dfs

    def _calculate_probabilities_vectorized(
        self,
        event_name: str,
        event_cfg: Union[OutcomeConfig, ExposureConfig],
        history_matrix: np.ndarray,
        ages: np.ndarray,
        is_exposed: bool = False,
        additional_logit_effect: np.ndarray = 0.0,
    ) -> np.ndarray:
        """
        Calculates event probabilities for the entire cohort in a vectorized manner,
        using low-rank factorization for efficient interaction effect computation.
        """
        n_patients = history_matrix.shape[0]

        if event_name == "exposure":
            weights = self.weights["exposure"]
        else:  # It's an outcome
            weights = self.weights["_outcomes_shared"]

        logit_p_array = np.full(n_patients, logit(event_cfg.p_base), dtype=np.float32)

        # 1. Linear effects (always applied)
        # This operation is much faster if history_matrix is sparse
        logit_p_array += history_matrix @ weights["linear"]

        # 2. Interaction effects using Low-Rank Factorization (The key optimization) 🚀
        W_U = weights["interaction_U"]
        W_K = weights["interaction_K"]

        # Efficiently calculate (A @ U) * (A @ K) and sum over the latent dimension
        # where A is the history_matrix. This is much faster than the old einsum.
        left_term = history_matrix @ W_U  # Shape: (n_patients, latent_dim)
        right_term = history_matrix @ W_K  # Shape: (n_patients, latent_dim)

        # Element-wise product and sum is equivalent to the full quadratic form
        interaction_effect = np.sum(left_term * right_term, axis=1)
        logit_p_array += interaction_effect

        # 3. Exposure effect (for outcomes)
        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p_array += event_cfg.exposure_effect

        # 4. Age effect
        if hasattr(event_cfg, "age_effect") and event_cfg.age_effect is not None:
            logit_p_array += event_cfg.age_effect * ages

        # 5. Unobserved confounder effect
        logit_p_array += additional_logit_effect

        # Add random noise
        logit_p_array += self.rng.normal(0, 0.01, n_patients)

        return expit(logit_p_array)

    def save_weights(self):
        f_save_weights(self.vocabulary, self.weights, self.config.paths.outcomes)

    def _get_patient_history_matrix(
        self, history_df: pd.DataFrame, all_pids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a patient history matrix with weights based on temporal decay.

        For each patient, it finds the last occurrence of every unique code,
        calculates a weight using a half-life decay function based on the
        time difference to the index date, and populates the matrix with these weights.
        """
        # If there's no history, return an empty matrix of the correct shape.
        if history_df.empty:
            matrix_shape = (len(all_pids), len(self.vocabulary))
            return np.zeros(matrix_shape, dtype=np.float32), all_pids

        # 1. Find the latest occurrence for each unique patient-code pair
        latest_events_df = history_df.loc[
            history_df.groupby([PID_COL, CONCEPT_COL])[TIMESTAMP_COL].idxmax()
        ].copy()

        # 2. Calculate the difference in days from the index date
        # .clip(lower=0) ensures no negative days if data slips past the index_date
        latest_events_df["diff_days"] = (
            (self.index_date - latest_events_df[TIMESTAMP_COL]).dt.days
        ).clip(lower=0)

        # 3. Compute the weight using the half-life decay formula
        # Assumes `temporal_halflife_days` is in your config.
        halflife = self.config.simulation_model.time_decay_halflife_days
        if halflife and halflife > 0:
            latest_events_df[WEIGHT_COL] = 2 ** (
                -latest_events_df["diff_days"] / halflife
            )
        else:
            # If no halflife is defined, default to a weight of 1 (no decay)
            latest_events_df[WEIGHT_COL] = 1.0
        logger.info(f"Recency weights: {latest_events_df[WEIGHT_COL].describe()}")
        # 4. Build the final weighted matrix using the efficient NumPy approach
        pid_to_row = {pid: i for i, pid in enumerate(all_pids)}

        # Map dataframe columns to integer indices for the matrix
        rows = (
            latest_events_df[PID_COL]
            .map(pid_to_row.get, na_action="ignore")
            .to_numpy(na_value=-1, dtype=int)
        )
        # Use the class's code_to_idx map
        cols = (
            latest_events_df[CONCEPT_COL]
            .map(self.code_to_idx.get, na_action="ignore")
            .to_numpy(na_value=-1, dtype=int)
        )
        weights = latest_events_df[WEIGHT_COL].to_numpy(dtype=np.float32)

        # Filter out any entries that couldn't be mapped
        valid_indices = (rows != -1) & (cols != -1)
        rows, cols, weights = (
            rows[valid_indices],
            cols[valid_indices],
            weights[valid_indices],
        )

        # Initialize the matrix with float dtype
        matrix_shape = (len(all_pids), len(self.vocabulary))
        patient_history_matrix = np.zeros(matrix_shape, dtype=np.float32)

        # Populate the matrix with weights. Using `np.add.at` is robust,
        # though simple assignment would also work here since we've already taken the latest event.
        np.add.at(patient_history_matrix, (rows, cols), weights)

        return patient_history_matrix, all_pids

    def _simulate_unobserved_confounder_effects(
        self, n_patients: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates unobserved confounder effects for a cohort."""
        confounder_cfg = self.config.unobserved_confounder
        if not confounder_cfg:
            return np.zeros(n_patients), {}

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
        control_pids = safe_control_pids(all_pids, exposed_pids)

        if len(exposed_pids) == 0:
            return pd.DataFrame(
                columns=[CONTROL_PID_COL, EXPOSED_PID_COL, TIMESTAMP_COL, ABSPOS_COL]
            )
        if not control_pids:
            return pd.DataFrame(
                columns=[CONTROL_PID_COL, EXPOSED_PID_COL, TIMESTAMP_COL, ABSPOS_COL]
            )

        matched_exposed_pids = self.rng.choice(
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
