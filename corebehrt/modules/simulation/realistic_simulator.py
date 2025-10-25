import logging
import os
from os.path import join
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

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
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.simulation.config_realistic import (
    ExposureConfig,
    InfluenceScalesConfig,
    ModelWeightsConfig,
    OutcomeConfig,
    RealisticSimulationModelConfig,
    SimulationConfig,
)
from corebehrt.modules.simulation.plot import (
    plot_hist,
    plot_probability_distributions,
    plot_true_effects_vs_risk_differences,
)

logger = logging.getLogger("simulate")
WEIGHT_COL = "weight"


class RealisticCausalSimulator:
    """
    A realistic causal data generator for EHR data using decomposed latent health factors.

    This simulator implements a sophisticated Data Generating Process (DGP) that transforms
    patient longitudinal EHR data into realistic causal scenarios with known ground truth.

    **Data Generating Process (4 Steps):**

    **Step 1: Patient History Representation (x)**
        Converts longitudinal EHR data up to an index date into feature vectors x ∈ R^V.
        Features are weighted by exponential time decay: x_j = exp(-Δt_j / τ)
        where Δt_j is time since last occurrence and τ is the decay half-life.

    **Step 2: Latent Health Factors (z)**
        Maps high-dimensional history x to low-dimensional latent factors z ∈ R^D:
        z = tanh(x @ W_f), where W_f is a sparse weight matrix.

        The latent space is partitioned into three disjoint sets:
        - z_sh: Shared factors (confounders affecting both exposure and outcomes)
        - z_exp: Exposure-only factors (instrumental variables)
        - z_out: Outcome-only factors (independent outcome risks)

    **Step 3: Exposure Assignment (A)**
        Treatment probability (propensity score) depends on confounding and exposure factors:
        logit(P(A=1|z)) = α₀ + λ_sh(z_sh·w_sh→A) + λ_exp(z_exp·w_exp→A) + u
        where λ terms control factor influence and u is optional unobserved confounding.

    **Step 4: Potential Outcome Generation (Y(a))**
        Outcomes depend on confounders, outcome factors, and treatment:
        logit(P(Y_k(a)=1|z)) = γ₀ + ω_sh(z_sh·w_sh→Y_k) + ω_out(z_out·w_out→Y_k) + a·δ_k + u_k
        where δ_k is the true Average Treatment Effect for outcome k.

    **Key Features:**
    - Realistic confounding structure with identifiable components
    - Multiple correlated outcomes through shared latent factors
    - Ground truth causal effects for benchmarking
    - Configurable factor decomposition and influence strengths
    - Optional unobserved confounding simulation

    Args:
        config (SimulationConfig): Configuration object containing simulation parameters,
            factor decomposition settings, and outcome definitions.

    Example:
        ```python
        config = SimulationConfig(...)
        simulator = RealisticCausalSimulator(config)
        results = simulator.simulate_dataset(patient_df)
        ```
    """

    def __init__(
        self, config: SimulationConfig
    ):  # config should be an instance of SimulationConfig
        self.config = config
        self.index_date = config.index_date
        self.rng = np.random.default_rng(config.seed)
        self.sampling_func = self.rng.normal
        self.debug = config.debug
        self._initialize_state()

    def _initialize_state(self):
        """Initializes stateful attributes for the simulator."""
        self.code_to_idx: Dict[str, int] = {}
        self.vocabulary: List[str] = []

        model_cfg: RealisticSimulationModelConfig = self.config.simulation_model
        self.num_shared_factors = model_cfg.num_shared_factors
        self.num_exposure_only_factors = model_cfg.num_exposure_only_factors
        self.num_outcome_only_factors = model_cfg.num_outcome_only_factors
        self.total_latent_factors = (
            self.num_shared_factors
            + self.num_exposure_only_factors
            + self.num_outcome_only_factors
        )

        self.influence: InfluenceScalesConfig = model_cfg.influence_scales

        factor_config: ModelWeightsConfig = model_cfg.factor_mapping
        self.weights = {
            "factor_weights": np.zeros((0, self.total_latent_factors)),
            "exposure_factor_weights": self.rng.normal(
                factor_config.exposure_factor_mean,
                factor_config.exposure_factor_scale,
                self.num_shared_factors + self.num_exposure_only_factors,
            ),
            "outcomes_factor_weights": self._create_correlated_outcome_weights(),
        }

    def _create_correlated_outcome_weights(self) -> np.ndarray:
        """Generates weights from shared and outcome-only factors to outcomes."""
        num_outcomes = len(self.config.outcomes)
        num_relevant_factors = self.num_shared_factors + self.num_outcome_only_factors
        weights = np.zeros((num_relevant_factors, num_outcomes))

        factor_config = self.config.simulation_model.factor_mapping
        for i in range(num_relevant_factors):
            prob_influence = factor_config.outcome_influence_probability
            influenced_outcomes = self.rng.random(num_outcomes) < prob_influence
            factor_weights = self.rng.normal(
                factor_config.outcome_factor_mean,
                factor_config.outcome_factor_scale,
                np.sum(influenced_outcomes),
            )
            weights[i, influenced_outcomes] = factor_weights
        return weights

    def _update_vocabulary_and_weights(self, codes_in_shard: Set[str]):
        """Updates vocabulary and grows the code-to-factor weight matrix."""
        new_codes = list(codes_in_shard - set(self.vocabulary))
        if not new_codes:
            return

        n_new = len(new_codes)
        for code in new_codes:
            self.code_to_idx[code] = len(self.vocabulary)
            self.vocabulary.append(code)

        factor_config = self.config.simulation_model.factor_mapping
        new_factor_weights = self.sampling_func(
            factor_config.mean, factor_config.scale, (n_new, self.total_latent_factors)
        )
        is_zero = (
            self.rng.random((n_new, self.total_latent_factors))
            < factor_config.sparsity_factor
        )
        new_factor_weights[is_zero] = 0

        self.weights["factor_weights"] = np.vstack(
            [self.weights["factor_weights"], new_factor_weights]
        )

    def _calculate_latent_factors(
        self, history_matrix: np.ndarray, ages: np.ndarray
    ) -> np.ndarray:
        """Calculates the latent factor values for each patient."""
        latent_factors = history_matrix @ self.weights["factor_weights"]
        if self.config.simulation_model.treat_age_as_latent_factor:
            # Normalize age (e.g., standard scaling) to match the scale of other factors
            mean_age, std_age = np.mean(ages), np.std(ages)
            if std_age > 0:
                normalized_ages = (ages - mean_age) / std_age
            else:
                # If all ages are the same, their normalized value is 0 (since they are at the mean)
                normalized_ages = np.zeros_like(ages)
            # Inject age as the first latent factor (assuming it's a shared factor)
            latent_factors[:, 0] = normalized_ages

        return np.tanh(latent_factors)

    def _calculate_probabilities_decomposed(
        self,
        event_name: str,
        event_cfg: ExposureConfig | OutcomeConfig,
        latent_factors: np.ndarray,
        is_exposed: bool = False,
        additional_logit_effect: np.ndarray = 0.0,
    ) -> np.ndarray:
        """Calculates probabilities based on the decomposed latent factors."""
        n_patients = latent_factors.shape[0]
        logit_p_array = np.full(n_patients, logit(event_cfg.p_base), dtype=np.float32)

        s_end = self.num_shared_factors
        e_end = s_end + self.num_exposure_only_factors

        Z_shared = latent_factors[:, :s_end]
        Z_exposure = latent_factors[:, s_end:e_end]
        Z_outcome = latent_factors[:, e_end:]

        if event_name == "exposure":
            shared_effect = (
                Z_shared @ self.weights["exposure_factor_weights"][:s_end]
            ) * self.influence.shared_to_exposure
            exposure_only_effect = (
                Z_exposure @ self.weights["exposure_factor_weights"][s_end:]
            ) * self.influence.exposure_only_to_exposure
            logit_p_array += shared_effect + exposure_only_effect
        else:  # It's an outcome
            outcome_idx = list(self.config.outcomes.keys()).index(event_name)
            outcome_weights = self.weights["outcomes_factor_weights"][:, outcome_idx]
            shared_effect = (
                Z_shared @ outcome_weights[:s_end]
            ) * self.influence.shared_to_outcome
            outcome_only_effect = (
                Z_outcome @ outcome_weights[s_end:]
            ) * self.influence.outcome_only_to_outcome
            logit_p_array += shared_effect + outcome_only_effect

        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p_array += event_cfg.exposure_effect

        logit_p_array += additional_logit_effect

        noise_scale = self.config.simulation_model.noise.logit_noise_scale
        logit_p_array += self.rng.normal(0, noise_scale, n_patients)
        return expit(logit_p_array)

    def simulate_dataset(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Orchestrates the simulation for a single shard of patient data."""
        history_df, initial_pids = self._filter_initial_history(df)
        if history_df.empty:
            return {}
        logger.info(
            f"[Filter 1] Patients with history before index date: {len(initial_pids)}"
        )

        ages = self._calculate_ages(history_df, initial_pids)
        history_df = self._filter_by_codes(history_df)
        if history_df.empty:
            return {}
        logger.info(
            f"[Filter 2] After code filtering: {history_df[PID_COL].nunique()} patients"
        )

        patient_history_matrix, pids, final_ages = self._prepare_simulation_inputs(
            history_df, ages
        )
        if len(pids) == 0:
            return {}
        logger.info(
            f"[Filter 3] After history matrix preparation: {len(pids)} patients"
        )

        confounder_exposure_effect, confounder_outcome_effects = (
            self._simulate_unobserved_confounder_effects(len(pids))
        )

        is_exposed, p_exposure = self._simulate_exposure(
            patient_history_matrix, final_ages, confounder_exposure_effect
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

        return self._package_results(
            pids,
            ite_records,
            cf_records,
            all_factual_events,
            all_probas_for_plotting,
            p_exposure,
            is_exposed,
        )

    def _simulate_exposure(
        self,
        patient_history_matrix: np.ndarray,
        final_ages: np.ndarray,
        confounder_exposure_effect: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates exposure probabilities based on patient history and confounder effects."""
        latent_factors = self._calculate_latent_factors(
            patient_history_matrix, final_ages
        )
        p_exposure = self._calculate_probabilities_decomposed(
            "exposure",
            self.config.exposure,
            latent_factors,
            additional_logit_effect=confounder_exposure_effect,
        )
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
        latent_factors = self._calculate_latent_factors(
            patient_history_matrix, final_ages
        )
        n_patients = len(pids)
        ite_records = {PID_COL: pids}
        cf_records = {PID_COL: pids, EXPOSURE_COL: is_exposed.astype(int)}
        all_factual_events, all_probas_for_plotting = [], {}

        for outcome_name, outcome_cfg in self.config.outcomes.items():
            confounder_effect = confounder_outcome_effects.get(
                outcome_name, np.zeros(n_patients)
            )
            p_if_treated = self._calculate_probabilities_decomposed(
                outcome_name,
                outcome_cfg,
                latent_factors,
                is_exposed=True,
                additional_logit_effect=confounder_effect,
            )
            p_if_control = self._calculate_probabilities_decomposed(
                outcome_name,
                outcome_cfg,
                latent_factors,
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

    def _filter_initial_history(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Filters the dataframe to only include patients with history before the index date.
        """
        history_df = df[df[TIMESTAMP_COL] <= self.index_date].copy()
        pids_with_history = history_df[PID_COL].unique()
        logger.info(
            f"  Patients with any events before {self.index_date}: {len(pids_with_history)}"
        )
        if len(pids_with_history) == 0:
            return pd.DataFrame(), np.array([])
        dead_pids = history_df[history_df[CONCEPT_COL] == DEATH_CODE][PID_COL].unique()
        if len(dead_pids) > 0:
            logger.info(f"  Excluding {len(dead_pids)} deceased patients")
            history_df = history_df[~history_df[PID_COL].isin(dead_pids)]
        all_pids = history_df[PID_COL].unique()
        logger.info(f"  Patients after history filter: {len(all_pids)}")
        if len(all_pids) == 0:
            return pd.DataFrame(), np.array([])
        return history_df, all_pids

    def _calculate_ages(
        self, history_df: pd.DataFrame, all_pids: np.ndarray
    ) -> pd.Series:
        """
        Calculates the ages of patients on index date based on their birth dates.
        """
        dob_events = history_df[history_df[CONCEPT_COL] == BIRTH_CODE]
        patient_dobs = dob_events.groupby(PID_COL)[TIMESTAMP_COL].first()
        ages = pd.Series(np.nan, index=all_pids, dtype=float)
        if not patient_dobs.empty:
            days_per_year = self.config.simulation_model.age.days_per_year
            ages.update((self.index_date - patient_dobs).dt.days / days_per_year)
        mean_age = ages.mean()
        if np.isnan(mean_age):
            mean_age = self.config.simulation_model.age.default_age
        return ages.fillna(mean_age)

    def _filter_by_codes(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Only use certain codes for simulation."""
        if self.config.include_code_prefixes:
            before_count = history_df[PID_COL].nunique()
            history_df = history_df[
                history_df[CONCEPT_COL].str.startswith(
                    tuple(self.config.include_code_prefixes)
                )
            ].copy()
            after_count = history_df[PID_COL].nunique()
            if before_count > 0:
                retention_pct = (after_count / before_count) * 100
                logger.info(
                    f"  After code prefix filter {self.config.include_code_prefixes}: {after_count} patients ({retention_pct:.1f}% retained)"
                )

        if self.config.min_num_codes > 1:
            before_count = history_df[PID_COL].nunique()
            code_counts = history_df.groupby(PID_COL)[CONCEPT_COL].nunique()

            # Log distribution of code counts
            patients_by_code_count = code_counts.value_counts().sort_index()
            top_counts = dict(patients_by_code_count.head(10))
            logger.info(f"  Code count distribution (top 10): {top_counts}")

            pids_to_keep = code_counts[code_counts >= self.config.min_num_codes].index
            history_df = history_df[history_df[PID_COL].isin(pids_to_keep)].copy()
            after_count = history_df[PID_COL].nunique()
            excluded = before_count - after_count
            logger.info(
                f"  After min_codes filter (>={self.config.min_num_codes}): {after_count} patients"
            )
            if excluded > 0:
                logger.info(
                    f"  Excluded {excluded} patients with <{self.config.min_num_codes} codes"
                )
        return history_df

    def _prepare_simulation_inputs(
        self, history_df: pd.DataFrame, ages: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_pids = history_df[PID_COL].unique()
        if len(all_pids) == 0:
            return np.array([]), np.array([]), np.array([])
        codes_in_shard = set(history_df[CONCEPT_COL].unique())
        self._update_vocabulary_and_weights(codes_in_shard)
        patient_history_matrix, pids = self._get_patient_history_matrix(
            history_df, all_pids
        )
        final_ages = ages.loc[pids].values
        return patient_history_matrix, pids, final_ages

    def _get_patient_history_matrix(
        self, history_df: pd.DataFrame, all_pids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if history_df.empty:
            return np.zeros(
                (len(all_pids), len(self.vocabulary)), dtype=np.float32
            ), all_pids
        latest_events_df: pd.DataFrame = history_df.loc[
            history_df.groupby([PID_COL, CONCEPT_COL])[TIMESTAMP_COL].idxmax()
        ].copy()
        latest_events_df["diff_days"] = (
            (self.index_date - latest_events_df[TIMESTAMP_COL]).dt.days
        ).clip(lower=0)
        halflife = self.config.simulation_model.time_decay_halflife_days
        latest_events_df[WEIGHT_COL] = (
            2 ** (-latest_events_df["diff_days"] / halflife)
            if halflife and halflife > 0
            else 1.0
        )
        pid_to_row = {pid: i for i, pid in enumerate(all_pids)}
        rows = (
            latest_events_df[PID_COL].map(pid_to_row).to_numpy(na_value=-1, dtype=int)
        )
        cols = (
            latest_events_df[CONCEPT_COL]
            .map(self.code_to_idx)
            .to_numpy(na_value=-1, dtype=int)
        )
        weights = latest_events_df[WEIGHT_COL].to_numpy(dtype=np.float32)
        valid_indices = (rows != -1) & (cols != -1)
        patient_history_matrix = np.zeros(
            (len(all_pids), len(self.vocabulary)), dtype=np.float32
        )
        np.add.at(
            patient_history_matrix,
            (rows[valid_indices], cols[valid_indices]),
            weights[valid_indices],
        )
        return patient_history_matrix, all_pids

    def _simulate_unobserved_confounder_effects(
        self, n_patients: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        confounder_cfg = self.config.unobserved_confounder
        if not confounder_cfg:
            return np.zeros(n_patients), {}
        has_confounder = self.rng.binomial(
            1, confounder_cfg.p_occurrence, n_patients
        ).astype(bool)
        exposure_effect = np.where(has_confounder, confounder_cfg.exposure_effect, 0.0)
        outcome_effects = {
            name: np.where(has_confounder, effect, 0.0)
            for name, effect in confounder_cfg.outcome_effects.items()
        }
        return exposure_effect, outcome_effects

    def _create_exposure_events(self, exposed_pids: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                PID_COL: exposed_pids,
                TIMESTAMP_COL: self.index_date,
                CONCEPT_COL: EXPOSURE_COL,
            }
        )

    def _calculate_and_save_simulation_stats(
        self,
        pids: np.ndarray,
        is_exposed: np.ndarray,
        cf_records: Dict[str, np.ndarray],
        output_dir: str,
    ):
        """Calculate and save simple simulation statistics."""
        total_patients = len(pids)
        num_exposed = np.sum(is_exposed)
        num_control = total_patients - num_exposed

        # Calculate outcome statistics
        outcome_stats = {}
        for outcome_name in self.config.outcomes.keys():
            outcome_col = f"{OUTCOME_COL}_{outcome_name}"
            if outcome_col in cf_records:
                num_with_outcome = np.sum(cf_records[outcome_col])
                outcome_stats[outcome_name] = {
                    "total_with_outcome": int(num_with_outcome),
                    "percentage_with_outcome": float(
                        num_with_outcome / total_patients * 100
                    ),
                }

        # Save as CSV for easy reading
        stats_rows = [
            ["Statistic", "Value"],
            ["Total Patients", total_patients],
            ["Number Exposed", num_exposed],
            ["Number Control", num_control],
            ["Exposure Rate (%)", f"{num_exposed / total_patients * 100:.2f}"],
        ]

        # Add outcome statistics
        for outcome_name, outcome_data in outcome_stats.items():
            stats_rows.append(
                [
                    f"{outcome_name} - Total with Outcome",
                    outcome_data["total_with_outcome"],
                ]
            )
            stats_rows.append(
                [
                    f"{outcome_name} - Percentage with Outcome (%)",
                    f"{outcome_data['percentage_with_outcome']:.2f}",
                ]
            )

        stats_df = pd.DataFrame(stats_rows[1:], columns=stats_rows[0])

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        stats_path = join(output_dir, "simulation_stats.csv")
        stats_df.to_csv(stats_path, index=False)

        logger.info(f"Simulation statistics saved to {stats_path}")
        logger.info(
            f"Total patients: {total_patients}, Exposed: {num_exposed}, Control: {num_control}"
        )
        for outcome_name, outcome_data in outcome_stats.items():
            logger.info(
                f"{outcome_name}: {outcome_data['total_with_outcome']} patients ({outcome_data['percentage_with_outcome']:.2f}%)"
            )

    def _calculate_theoretical_roc_auc(
        self,
        cf_records: Dict[str, np.ndarray],
        is_exposed: np.ndarray,
        p_exposure: np.ndarray,
        output_dir: str,
    ) -> Dict[str, float]:
        """
        Calculate the theoretical maximum ROC AUC for exposure and outcomes.

        This represents the best possible ROC AUC that any model could achieve
        if it had perfect knowledge of the data generating process.

        Args:
            cf_records: Dictionary containing counterfactual records with true probabilities.
            is_exposed: Binary array indicating exposure status.
            p_exposure: Array of true probabilities for being exposed.
            output_dir: Directory to save results.

        Returns:
            Dictionary mapping exposure/outcome names to their theoretical max ROC AUC values.
        """

        theoretical_aucs = {}
        results_data = []

        logger.info(
            "Calculating theoretical maximum ROC AUC from true probabilities..."
        )

        # --- 1. Calculate for Exposure ---
        logger.info("Calculating theoretical max ROC AUC for Exposure prediction...")
        if len(np.unique(is_exposed)) > 1:
            auc_exposure = roc_auc_score(is_exposed, p_exposure)
            theoretical_aucs["exposure"] = auc_exposure

            results_data.append(
                {
                    "outcome": "exposure",  # Use 'outcome' column for consistency in the output table
                    "auc_factual_dgp": auc_exposure,
                    "auc_if_all_treated": np.nan,
                    "auc_if_all_control": np.nan,
                    "n_positive": int(np.sum(is_exposed)),
                    "n_total": len(is_exposed),
                    "prevalence": np.mean(is_exposed),
                }
            )
            logger.info(f"Exposure: Theoretical max ROC AUC = {auc_exposure:.4f}")
        else:
            logger.warning(
                "Cannot calculate ROC AUC for exposure: only one class present."
            )
            theoretical_aucs["exposure"] = np.nan

        # --- 2. Calculate for Outcomes ---
        logger.info("Calculating theoretical max ROC AUC for Outcome prediction...")
        for outcome_name in self.config.outcomes.keys():
            # Get the factual outcomes (ground truth labels)
            outcome_col = f"{OUTCOME_COL}_{outcome_name}"
            if outcome_col not in cf_records:
                logger.warning(f"Outcome {outcome_name} not found in cf_records.")
                continue

            y_true = cf_records[outcome_col]

            # Get the true probabilities for this outcome
            p_exposed_col = f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"
            p_control_col = f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"

            if p_exposed_col not in cf_records or p_control_col not in cf_records:
                logger.warning(f"Simulated probabilities not found for {outcome_name}.")
                continue

            p_if_treated = cf_records[p_exposed_col]
            p_if_control = cf_records[p_control_col]

            # The true probability of the factual outcome, given the patient's history and actual exposure
            y_prob_factual = np.where(is_exposed, p_if_treated, p_if_control)

            # Calculate ROC AUC using true DGP probabilities
            if len(np.unique(y_true)) > 1:  # Need both classes for ROC AUC
                auc_factual = roc_auc_score(y_true, y_prob_factual)
                theoretical_aucs[outcome_name] = auc_factual

                # Also calculate AUC using counterfactual probabilities for comparison
                auc_treated = roc_auc_score(y_true, p_if_treated)
                auc_control = roc_auc_score(y_true, p_if_control)

                results_data.append(
                    {
                        "outcome": outcome_name,
                        "auc_factual_dgp": auc_factual,
                        "auc_if_all_treated": auc_treated,
                        "auc_if_all_control": auc_control,
                        "n_positive": int(np.sum(y_true)),
                        "n_total": len(y_true),
                        "prevalence": np.mean(y_true),
                    }
                )

                logger.info(
                    f"{outcome_name}: Theoretical max ROC AUC = {auc_factual:.4f}"
                )
            else:
                logger.warning(
                    f"Cannot calculate ROC AUC for {outcome_name}: only one class present."
                )
                theoretical_aucs[outcome_name] = np.nan

        # --- 3. Save all results to CSV ---
        if results_data:
            results_df = pd.DataFrame(results_data)
            os.makedirs(output_dir, exist_ok=True)
            results_path = join(output_dir, "theoretical_max_roc_auc.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Theoretical ROC AUC results saved to {results_path}")

            # Print summary
            logger.info("=== THEORETICAL MAXIMUM ROC AUC SUMMARY ===")
            for _, row in results_df.iterrows():
                logger.info(
                    f"{row['outcome'].title()}: {row['auc_factual_dgp']:.4f} "
                    f"(prevalence: {row['prevalence']:.3f}, n={row['n_total']})"
                )

        return theoretical_aucs

    def _package_results(
        self,
        pids,
        ite_records,
        cf_records,
        all_factual_events,
        all_probas_for_plotting,
        p_exposure,
        is_exposed,
    ) -> Dict[str, pd.DataFrame]:
        # --- Calculate and save simulation statistics ---
        logger.info("Calculating and saving simulation statistics...")
        self._calculate_and_save_simulation_stats(
            pids, is_exposed, cf_records, self.config.paths.outcomes
        )

        # --- Calculate theoretical maximum ROC AUC ---
        logger.info("Calculating theoretical maximum ROC AUC...")
        theoretical_aucs = self._calculate_theoretical_roc_auc(
            cf_records, is_exposed, p_exposure, self.config.paths.outcomes
        )
        logger.info(f"Theoretical maximum ROC AUC: {theoretical_aucs}")
        # --- Plotting integrated here ---
        logger.info("Plotting ground truth probability distributions...")
        figs_dir = join(self.config.paths.outcomes, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        plot_hist(p_exposure, figs_dir, is_exposed)
        plot_probability_distributions(all_probas_for_plotting, figs_dir)

        # Create dataframes for effect comparison plotting
        ite_df = pd.DataFrame(ite_records)
        cf_df = pd.DataFrame(cf_records)

        # Extract true effects configuration from the config
        true_effects_config = {}
        for outcome_name, outcome_cfg in self.config.outcomes.items():
            true_effects_config[outcome_name] = {
                "exposure_effect": outcome_cfg.exposure_effect,
                "p_base": outcome_cfg.p_base,
            }

        # Plot true effects vs observed effects
        logger.info("Plotting true effects vs observed risk differences...")
        plot_true_effects_vs_risk_differences(
            ite_df=ite_df,
            cf_df=cf_df,
            true_effects_config=true_effects_config,
            output_dir=figs_dir,
        )

        output_dfs = {}
        if all_factual_events:
            events_df: pd.DataFrame = pd.concat(all_factual_events, ignore_index=True)
            events_df[ABSPOS_COL] = get_hours_since_epoch(events_df[TIMESTAMP_COL])
            for code, group in events_df.groupby(CONCEPT_COL):
                output_dfs[str(code)] = group[
                    [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
                ].copy()

        output_dfs["ite"] = ite_df
        output_dfs[COUNTERFACTUALS_FILE.split(".")[0]] = cf_df
        if EXPOSURE_COL in output_dfs:
            output_dfs[INDEX_DATE_MATCHING_FILE.split(".")[0]] = (
                self._create_index_date_matching_df(output_dfs[EXPOSURE_COL], pids)
            )

        return output_dfs

    def _create_index_date_matching_df(
        self, exposure_df: pd.DataFrame, all_pids: list
    ) -> pd.DataFrame:
        exposed_pids = exposure_df[PID_COL].unique()
        control_pids = safe_control_pids(all_pids, exposed_pids)
        if len(exposed_pids) == 0 or len(control_pids) == 0:
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
