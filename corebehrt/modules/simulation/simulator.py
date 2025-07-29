import json
import logging
from os.path import join
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm import tqdm

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
from corebehrt.modules.simulation.config import (
    ExposureConfig,
    OutcomeConfig,
    SimulationConfig,
)

logger = logging.getLogger("simulate")


class CausalSimulator:
    """
    Simulates exposure and outcome events based on a patient's history.
    """

    def __init__(self, config: SimulationConfig, seed: int = 42):
        self.config = config
        self.index_date = config.index_date
        self.rng = np.random.default_rng(seed)
        self._initialize_weights()

    def _initialize_weights(self):
        """Samples and stores weights for all simulation events."""

        def sample_laplace(cfg, size):
            """Sample from Laplace distribution using mean and std."""
            return self.rng.normal(cfg.mean, cfg.scale, size)

        self.weights = {}
        self.code_to_idx = {code: i for i, code in enumerate(self.config.trigger_codes)}
        n_codes = len(self.config.trigger_codes)

        linear_config = self.config.simulation_model.linear
        interaction_config = self.config.simulation_model.interaction

        # Sample weights for exposure
        self.weights["exposure"] = {
            "linear": sample_laplace(linear_config, n_codes),
            "interaction_joint": sample_laplace(interaction_config, (n_codes, n_codes)),
        }

        # Sample one set of weights for all outcomes
        outcome_weights = {
            "linear": sample_laplace(linear_config, n_codes),
            "interaction_joint": sample_laplace(interaction_config, (n_codes, n_codes)),
        }

        for outcome_name in self.config.outcomes:
            self.weights[outcome_name] = outcome_weights

        self.save_weights()

    def save_weights(self) -> None:
        """Saves the weights to a file."""

        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return np.round(obj, 4).tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        serializable_weights = to_serializable(self.weights)
        with open(
            join(self.config.paths.outcomes, "simulation_weights.json"), "w"
        ) as f:
            json.dump(serializable_weights, f)

    def simulate_dataset(
        self, df: pd.DataFrame, seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Simulates exposures and outcomes for a cohort, including counterfactuals.

        Args:
            df: DataFrame with patient data (subject_id, time, code).
            seed: Random seed for reproducibility.

        Returns:
            A dictionary of DataFrames, including factual events, ITEs,
            and counterfactual outcomes.
        """
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        factual_events, ite_records, cf_records = [], [], []
        subjects_as_dfs = [group for _, group in df.groupby(PID_COL)]

        for subj_df in tqdm(subjects_as_dfs, desc="Simulating exposures and outcomes"):
            subject_results = self._simulate_for_subject(subj_df)
            if not subject_results:
                continue

            factual_events.extend(subject_results["factual_events"])
            ite_records.append(subject_results["ite_record"])
            cf_records.append(subject_results["cf_record"])

        if not factual_events:
            return {}

        output_dfs = {}

        events_df = pd.DataFrame(factual_events)
        events_df[ABSPOS_COL] = get_hours_since_epoch(events_df[TIMESTAMP_COL])
        for code, group in events_df.groupby(CONCEPT_COL):
            output_dfs[str(code)] = group[[PID_COL, TIMESTAMP_COL, ABSPOS_COL]].copy()

        output_dfs["ite"] = pd.DataFrame(ite_records)
        output_dfs[COUNTERFACTUALS_FILE.split(".")[0]] = pd.DataFrame(cf_records)
        if EXPOSURE_COL in output_dfs:
            output_dfs[INDEX_DATE_MATCHING_FILE.split(".")[0]] = (
                self._create_index_date_matching_df(
                    output_dfs[EXPOSURE_COL], df[PID_COL].unique()
                )
            )
        return output_dfs

    def _simulate_for_subject(self, subj_df: pd.DataFrame) -> dict:
        """Simulates all events and counterfactuals for a single subject."""
        if subj_df.empty:
            return {}

        subject_id = subj_df[PID_COL].iloc[0]
        start_date = subj_df[TIMESTAMP_COL].min()
        end_date = subj_df[TIMESTAMP_COL].max()

        if (self.index_date >= end_date) or (self.index_date < start_date):
            return {}
        history_at_index = self._get_history_codes(subj_df, self.index_date)

        (
            unobserved_confounder_exposure_effect,
            unobserved_confounder_outcome_effects,
        ) = self._simulate_unobserved_confounder_effects()

        exposure_cfg = self.config.exposure
        p_exposure = self._calculate_probability(
            "exposure",
            exposure_cfg,
            history_at_index,
            additional_logit_effect=unobserved_confounder_exposure_effect,
        )
        is_exposed = self.rng.binomial(1, p_exposure) == 1

        factual_events = []
        if is_exposed:
            factual_events.append(
                {
                    PID_COL: subject_id,
                    TIMESTAMP_COL: self.index_date,
                    CONCEPT_COL: EXPOSURE_COL,
                }
            )

        history_for_outcomes = history_at_index.copy()

        ite_record = {PID_COL: subject_id}
        cf_record = {PID_COL: subject_id, EXPOSURE_COL: int(is_exposed)}

        for outcome_name, outcome_cfg in self.config.outcomes.items():
            assessment_time = self.index_date + pd.Timedelta(
                days=outcome_cfg.run_in_days
            )
            if assessment_time >= end_date:
                continue

            unobserved_confounder_outcome_effect = (
                unobserved_confounder_outcome_effects.get(outcome_name, 0.0)
            )

            p_if_treated = self._calculate_probability(
                outcome_name,
                outcome_cfg,
                history_for_outcomes,
                is_exposed=True,
                additional_logit_effect=unobserved_confounder_outcome_effect,
            )
            p_if_control = self._calculate_probability(
                outcome_name,
                outcome_cfg,
                history_for_outcomes,
                is_exposed=False,
                additional_logit_effect=unobserved_confounder_outcome_effect,
            )

            ite_record[f"ite_{outcome_name}"] = p_if_treated - p_if_control

            outcome_exposed = self.rng.binomial(1, p_if_treated)
            outcome_control = self.rng.binomial(1, p_if_control)
            factual_outcome = outcome_exposed if is_exposed else outcome_control

            if factual_outcome:
                factual_events.append(
                    {
                        PID_COL: subject_id,
                        TIMESTAMP_COL: assessment_time,
                        CONCEPT_COL: outcome_name,
                    }
                )

            cf_record[f"{OUTCOME_COL}_{outcome_name}"] = factual_outcome
            cf_record[f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"] = outcome_exposed
            cf_record[f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"] = outcome_control
            cf_record[f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"] = p_if_treated
            cf_record[f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"] = p_if_control

        return {
            "factual_events": factual_events,
            "ite_record": ite_record,
            "cf_record": cf_record,
        }

    def _simulate_unobserved_confounder_effects(self) -> Tuple[float, Dict[str, float]]:
        """Simulates the presence of an unobserved confounder and returns its effects."""
        confounder_cfg = self.config.unobserved_confounder
        if confounder_cfg and self.rng.binomial(1, confounder_cfg.p_occurrence) == 1:
            return (
                confounder_cfg.exposure_effect,
                confounder_cfg.outcome_effects or {},
            )

        return 0.0, {}

    def _get_history_codes(
        self, subj_df: pd.DataFrame, assessment_time: pd.Timestamp
    ) -> set:
        """Extracts codes from subject history up to assessment time."""
        history_mask = subj_df[TIMESTAMP_COL] <= assessment_time
        return set(subj_df.loc[history_mask, CONCEPT_COL])

    def _calculate_probability(
        self,
        event_name: str,
        event_cfg: Union[OutcomeConfig, ExposureConfig],
        history_codes: set,
        is_exposed: bool = False,
        additional_logit_effect: float = 0.0,
    ) -> float:
        """Calculates event probability based on history and, for outcomes, exposure status."""
        logit_p = logit(event_cfg.p_base)

        present_indices = {
            self.code_to_idx[c] for c in history_codes if c in self.code_to_idx
        }
        n_codes = len(self.config.trigger_codes)

        weights = self.weights[event_name]
        linear_weights = weights["linear"]
        joint_weights = weights["interaction_joint"]
        # Linear effects
        for i in present_indices:
            logit_p += linear_weights[i]

        # Interaction effects
        for i in range(n_codes):
            for j in range(i + 1, n_codes):
                i_present = i in present_indices
                j_present = j in present_indices

                # Joint presence
                if i_present and j_present:
                    logit_p += joint_weights[i, j]

        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p += event_cfg.exposure_effect

        logit_p += additional_logit_effect

        return expit(logit_p)

    def _create_index_date_matching_df(
        self, exposure_df: pd.DataFrame, all_pids: list
    ) -> pd.DataFrame:
        """
        Creates a DataFrame matching control subjects to exposed subjects for index date analysis.

        Args:
            exposure_df: DataFrame containing exposure events with PID, timestamp, and abspos
            all_pids: List of all patient IDs in the dataset

        Returns:
            DataFrame with columns: control_pid, exposed_pid, timestamp, abspos
        """

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

        # Set random seed for reproducible matching
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
