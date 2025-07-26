import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm import tqdm

from corebehrt.constants.causal.data import (
    OUTCOME_COL,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
    EXPOSURE_COL,
    CONTROL_PID_COL,
    EXPOSED_PID_COL,
)
from corebehrt.constants.causal.paths import (
    COUNTERFACTUALS_FILE,
    INDEX_DATE_MATCHING_FILE,
)

from corebehrt.constants.data import ABSPOS_COL, CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.simulation.config import (
    SimulationConfig,
    ExposureConfig,
    OutcomeConfig,
)

logger = logging.getLogger("simulate")


class CausalSimulator:
    """
    Simulates exposure and outcome events based on a patient's history.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.index_date = config.index_date

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

        unobserved_confounder_exposure_effect = 0
        unobserved_confounder_outcome_effects = {}
        if self.config.unobserved_confounder is not None:
            p_occurrence = self.config.unobserved_confounder.p_occurrence
            if np.random.binomial(1, p_occurrence) == 1:
                unobserved_confounder_exposure_effect = (
                    self.config.unobserved_confounder.exposure_effect
                )
                unobserved_confounder_outcome_effects = (
                    self.config.unobserved_confounder.outcome_effects
                )

        exposure_cfg = self.config.exposure
        p_exposure = self._calculate_probability(
            exposure_cfg,
            history_at_index,
            additional_logit_effect=unobserved_confounder_exposure_effect,
        )
        is_exposed = np.random.binomial(1, p_exposure) == 1

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
                outcome_cfg,
                history_for_outcomes,
                is_exposed=True,
                additional_logit_effect=unobserved_confounder_outcome_effect,
            )
            p_if_control = self._calculate_probability(
                outcome_cfg,
                history_for_outcomes,
                is_exposed=False,
                additional_logit_effect=unobserved_confounder_outcome_effect,
            )

            ite_record[f"ite_{outcome_name}"] = p_if_treated - p_if_control

            outcome_exposed = np.random.binomial(1, p_if_treated)
            outcome_control = np.random.binomial(1, p_if_control)
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

    def _get_history_codes(
        self, subj_df: pd.DataFrame, assessment_time: pd.Timestamp
    ) -> set:
        """Extracts codes from subject history up to assessment time."""
        history_mask = subj_df[TIMESTAMP_COL] <= assessment_time
        return set(subj_df.loc[history_mask, CONCEPT_COL])

    def _calculate_probability(
        self,
        event_cfg,
        history_codes: set,
        is_exposed: bool = False,
        additional_logit_effect: float = 0.0,
    ) -> float:
        """
        Calculates event probability based on history and, for outcomes, exposure status.
        Handles linear, quadratic, and interaction effects.
        """
        trigger_codes_array = np.array(list(event_cfg.trigger_codes))
        trigger_weights_array = np.array(list(event_cfg.trigger_weights))

        codes_present_mask = np.isin(trigger_codes_array, list(history_codes))
        trigger_effect_sum = np.sum(trigger_weights_array[codes_present_mask])

        logit_p = logit(event_cfg.p_base) + trigger_effect_sum

        logit_p = self._handle_quadratic_weights(
            logit_p, trigger_codes_array, codes_present_mask, event_cfg
        )
        logit_p = self._handle_combination(logit_p, history_codes, event_cfg)

        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p += event_cfg.exposure_effect

        logit_p += additional_logit_effect

        return expit(logit_p)

    @staticmethod
    def _handle_quadratic_weights(
        logit_p: float,
        trigger_codes_array: np.array,
        codes_present_mask: np.array,
        event_cfg: Union[OutcomeConfig, ExposureConfig],
    ) -> float:
        if event_cfg.quadratic_weights is not None:
            quad_weights = list(event_cfg.quadratic_weights)  # Create a copy
            if len(quad_weights) > len(trigger_codes_array):
                raise ValueError(
                    f"Quadratic weights length ({len(quad_weights)}) must be less than or equal to trigger codes length ({len(trigger_codes_array)})"
                )
            if len(quad_weights) < len(trigger_codes_array):
                quad_weights.extend(
                    [0] * (len(trigger_codes_array) - len(quad_weights))
                )
            quadratic_weights_array = np.array(quad_weights)
            quadratic_effect_sum = np.sum(quadratic_weights_array[codes_present_mask])
            logit_p += quadratic_effect_sum
        return logit_p

    @staticmethod
    def _handle_combination(
        logit_p: float,
        history_codes: set,
        event_cfg: Union[OutcomeConfig, ExposureConfig],
    ) -> float:
        if event_cfg.combinations is not None:
            for combination in event_cfg.combinations:
                if all(code in history_codes for code in combination["codes"]):
                    logit_p += combination["weight"]
        return logit_p

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
        np.random.seed(42)
        matched_exposed_pids = np.random.choice(
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
