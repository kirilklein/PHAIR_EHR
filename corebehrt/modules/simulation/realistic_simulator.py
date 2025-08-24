import logging
from os.path import join
from typing import Dict, List, Set, Tuple

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
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.simulation.plot import plot_hist, plot_probability_distributions
from corebehrt.modules.simulation.config_realistic import (
    SimulationConfig,
    RealisticSimulationModelConfig,
)

logger = logging.getLogger("simulate")
WEIGHT_COL = "weight"


class RealisticCausalSimulator:
    """
    A standalone simulator for a realistic causal scenario using decomposed latent health factors.

    The DGP is defined by:
    1. Patient History (X) -> Decomposed Latent Factors (Z_sh, Z_exp, Z_out)
    2. Latent Factors (Z) -> Exposure (A)
    3. Latent Factors (Z) + Exposure (A) -> Outcomes (Y)
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

        self.influence = model_cfg.influence_scales

        self.weights = {
            "factor_weights": np.zeros((0, self.total_latent_factors)),
            "exposure_factor_weights": self.rng.normal(
                0, 0.5, self.num_shared_factors + self.num_exposure_only_factors
            ),
            "outcomes_factor_weights": self._create_correlated_outcome_weights(),
        }

    def _create_correlated_outcome_weights(self) -> np.ndarray:
        """Generates weights from shared and outcome-only factors to outcomes."""
        num_outcomes = len(self.config.outcomes)
        num_relevant_factors = self.num_shared_factors + self.num_outcome_only_factors
        weights = np.zeros((num_relevant_factors, num_outcomes))

        for i in range(num_relevant_factors):
            prob_influence = 0.4
            influenced_outcomes = self.rng.random(num_outcomes) < prob_influence
            factor_weights = self.rng.normal(0, 0.75, np.sum(influenced_outcomes))
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

    def _calculate_latent_factors(self, history_matrix: np.ndarray) -> np.ndarray:
        """Calculates the latent factor values for each patient."""
        latent_factors = history_matrix @ self.weights["factor_weights"]
        return np.tanh(latent_factors)

    def _calculate_probabilities_decomposed(
        self,
        event_name: str,
        event_cfg,
        latent_factors: np.ndarray,
        ages: np.ndarray,
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
        if hasattr(event_cfg, "age_effect") and event_cfg.age_effect is not None:
            logit_p_array += event_cfg.age_effect * ages
        logit_p_array += additional_logit_effect
        logit_p_array += self.rng.normal(0, 0.01, n_patients)
        return expit(logit_p_array)

    def simulate_dataset(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Orchestrates the simulation for a single shard of patient data."""
        history_df, initial_pids = self._filter_initial_history(df)
        if history_df.empty:
            return {}

        ages = self._calculate_ages(history_df, initial_pids)
        history_df = self._filter_by_codes(history_df)
        if history_df.empty:
            return {}

        patient_history_matrix, pids, final_ages = self._prepare_simulation_inputs(
            history_df, ages
        )
        if len(pids) == 0:
            return {}

        confounder_exposure_effect, confounder_outcome_effects = (
            self._simulate_unobserved_confounder_effects(len(pids))
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
        self, patient_history_matrix, final_ages, pids, confounder_exposure_effect
    ) -> Tuple[np.ndarray, np.ndarray]:
        latent_factors = self._calculate_latent_factors(patient_history_matrix)
        p_exposure = self._calculate_probabilities_decomposed(
            "exposure",
            self.config.exposure,
            latent_factors,
            ages=final_ages,
            additional_logit_effect=confounder_exposure_effect,
        )
        is_exposed = self.rng.binomial(1, p_exposure).astype(bool)
        return is_exposed, p_exposure

    def _simulate_outcomes(
        self,
        patient_history_matrix,
        final_ages,
        pids,
        is_exposed,
        confounder_outcome_effects,
    ) -> Tuple[Dict, Dict, list, Dict]:
        latent_factors = self._calculate_latent_factors(patient_history_matrix)
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
                ages=final_ages,
                is_exposed=True,
                additional_logit_effect=confounder_effect,
            )
            p_if_control = self._calculate_probabilities_decomposed(
                outcome_name,
                outcome_cfg,
                latent_factors,
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

    def _filter_initial_history(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        history_df = df[df[TIMESTAMP_COL] <= self.index_date].copy()
        pids_with_history = history_df[PID_COL].unique()
        if len(pids_with_history) == 0:
            return pd.DataFrame(), np.array([])
        dead_pids = history_df[history_df[CONCEPT_COL] == DEATH_CODE][PID_COL].unique()
        if len(dead_pids) > 0:
            history_df = history_df[~history_df[PID_COL].isin(dead_pids)]
        all_pids = history_df[PID_COL].unique()
        if len(all_pids) == 0:
            return pd.DataFrame(), np.array([])
        return history_df, all_pids

    def _calculate_ages(
        self, history_df: pd.DataFrame, all_pids: np.ndarray
    ) -> pd.Series:
        dob_events = history_df[history_df[CONCEPT_COL] == BIRTH_CODE]
        patient_dobs = dob_events.groupby(PID_COL)[TIMESTAMP_COL].first()
        ages = pd.Series(np.nan, index=all_pids, dtype=float)
        if not patient_dobs.empty:
            ages.update((self.index_date - patient_dobs).dt.days / 365.25)
        mean_age = ages.mean()
        if np.isnan(mean_age):
            mean_age = 40
        return ages.fillna(mean_age)

    def _filter_by_codes(self, history_df: pd.DataFrame) -> pd.DataFrame:
        if self.config.include_code_prefixes:
            history_df = history_df[
                history_df[CONCEPT_COL].str.startswith(
                    tuple(self.config.include_code_prefixes)
                )
            ].copy()
        if self.config.min_num_codes > 1:
            code_counts = history_df.groupby(PID_COL)[CONCEPT_COL].nunique()
            pids_to_keep = code_counts[code_counts >= self.config.min_num_codes].index
            history_df = history_df[history_df[PID_COL].isin(pids_to_keep)].copy()
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
        # --- Plotting integrated here ---
        logger.info("Plotting ground truth probability distributions...")
        plot_hist(p_exposure, join(self.config.paths.outcomes, "figs"), is_exposed)
        plot_probability_distributions(
            all_probas_for_plotting, join(self.config.paths.outcomes, "figs")
        )

        output_dfs = {}
        if all_factual_events:
            events_df: pd.DataFrame = pd.concat(all_factual_events, ignore_index=True)
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
