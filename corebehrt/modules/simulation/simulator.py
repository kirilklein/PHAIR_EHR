import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm import tqdm

from corebehrt.constants.causal.data import (
    OUTCOME_COL,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
)
from corebehrt.constants.data import ABSPOS_COL, CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.simulation.config import SimulationConfig

logger = logging.getLogger("simulate")


class CausalSimulator:
    """
    Simulates exposure and outcome events based on a patient's history.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

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

        # The original DF might be empty, so we build it from the list of events
        if factual_events:
            events_df = pd.DataFrame(factual_events)
            events_df[ABSPOS_COL] = get_hours_since_epoch(events_df[TIMESTAMP_COL])
            for code, group in events_df.groupby(CONCEPT_COL):
                output_dfs[str(code)] = group[
                    [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
                ].copy()

        output_dfs["ite"] = pd.DataFrame(ite_records)
        output_dfs["counterfactuals"] = pd.DataFrame(cf_records)

        return output_dfs

    def _simulate_for_subject(self, subj_df: pd.DataFrame) -> dict:
        """Simulates all events and counterfactuals for a single subject."""
        if subj_df.empty:
            return {}

        subject_id = subj_df[PID_COL].iloc[0]
        end_date = subj_df[TIMESTAMP_COL].max()

        exposure_events = self._simulate_exposure_process(subj_df)
        is_exposed = bool(exposure_events)

        if is_exposed:
            index_date = min(e[TIMESTAMP_COL] for e in exposure_events)
        else:
            start_date = subj_df[TIMESTAMP_COL].min()
            index_date = start_date + pd.Timedelta(
                days=self.config.exposure.run_in_days
            )

        if index_date >= end_date:
            return {}

        history_at_index = self._get_history_codes(subj_df, index_date)

        history_for_outcomes = history_at_index.copy()
        if is_exposed:
            history_for_outcomes.add(self.config.exposure.code)

        ite_record = {PID_COL: subject_id}
        cf_record = {PID_COL: subject_id, "exposure": int(is_exposed)}
        factual_events = exposure_events

        for outcome_name, outcome_cfg in self.config.outcomes.items():
            assessment_time = index_date + pd.Timedelta(days=outcome_cfg.run_in_days)
            if assessment_time >= end_date:
                continue

            p_if_treated = self._calculate_probability(
                outcome_cfg, history_for_outcomes, is_exposed=True
            )
            p_if_control = self._calculate_probability(
                outcome_cfg, history_for_outcomes, is_exposed=False
            )

            ite_record[f"ite_{outcome_cfg.code}"] = p_if_treated - p_if_control

            outcome_exposed = np.random.binomial(1, p_if_treated)
            outcome_control = np.random.binomial(1, p_if_control)
            factual_outcome = outcome_exposed if is_exposed else outcome_control

            if factual_outcome:
                factual_events.append(
                    {
                        PID_COL: subject_id,
                        TIMESTAMP_COL: assessment_time,
                        CONCEPT_COL: outcome_cfg.code,
                    }
                )

            cf_record[f"{OUTCOME_COL}_{outcome_cfg.code}"] = factual_outcome
            cf_record[f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_cfg.code}"] = (
                outcome_exposed
            )
            cf_record[f"{SIMULATED_OUTCOME_CONTROL}_{outcome_cfg.code}"] = (
                outcome_control
            )

        return {
            "factual_events": factual_events,
            "ite_record": ite_record,
            "cf_record": cf_record,
        }

    def _simulate_exposure_process(self, subj_df: pd.DataFrame) -> List[Dict]:
        """Simulates the full exposure process for a subject, including compliance."""
        cfg = self.config.exposure

        first_exposure_event = self._simulate_time_to_first_event(subj_df)
        if not first_exposure_event:
            return []

        first_exposure_date = first_exposure_event[TIMESTAMP_COL]
        compliance_end_date = self._get_random_compliance_end_date(
            first_exposure_date, subj_df[TIMESTAMP_COL].max()
        )

        exposure_dates = self._generate_regular_exposures(
            first_exposure_date, compliance_end_date, cfg.compliance_interval_days
        )

        subject_id = subj_df[PID_COL].iloc[0]
        exposure_events = [
            {PID_COL: subject_id, TIMESTAMP_COL: date, CONCEPT_COL: cfg.code}
            for date in exposure_dates
        ]
        return exposure_events

    def _get_random_compliance_end_date(
        self, first_exposure_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Timestamp:
        """Selects a compliance end date using a stable linear weighting."""
        cfg = self.config.exposure
        earliest_end = first_exposure_date + pd.Timedelta(days=cfg.min_compliance_days)
        if earliest_end >= end_date:
            return end_date
        total_days = (end_date - earliest_end).days + 1
        weights = np.arange(1, total_days + 1, dtype=np.float64)
        probabilities = weights / np.sum(weights)
        chosen_offset = np.random.choice(np.arange(total_days), p=probabilities)
        return earliest_end + pd.Timedelta(days=int(chosen_offset))

    def _generate_regular_exposures(
        self,
        first_exposure_date: pd.Timestamp,
        compliance_end_date: pd.Timestamp,
        interval_days: int,
    ) -> pd.DatetimeIndex:
        """Generates exposure dates at regular intervals."""
        return pd.date_range(
            start=first_exposure_date, end=compliance_end_date, freq=f"{interval_days}D"
        )

    def _simulate_time_to_first_event(self, subj_df: pd.DataFrame) -> dict:
        """Simulates time to first event occurrence."""
        cfg = self.config.exposure
        timeline_info = self._setup_simulation_timeline(subj_df, cfg.run_in_days)
        if timeline_info is None:
            return {}

        daily_timeline, total_days = timeline_info
        p_daily_base = self._compute_daily_prob(cfg.p_base, total_days)

        feature_matrix = self._build_feature_matrix(
            subj_df, daily_timeline, cfg.trigger_codes
        )
        event_probabilities = self._compute_event_probabilities(
            feature_matrix, cfg.trigger_codes, cfg.trigger_weights, p_daily_base
        )

        event_draws = np.random.binomial(1, event_probabilities)
        if event_draws.any():
            event_idx = np.argmax(event_draws)
            event_time = daily_timeline[event_idx]
            return {
                PID_COL: subj_df[PID_COL].iloc[0],
                TIMESTAMP_COL: event_time,
                CONCEPT_COL: cfg.code,
            }
        return {}

    def _setup_simulation_timeline(
        self, subj_df: pd.DataFrame, run_in_days: int
    ) -> tuple[pd.DatetimeIndex, int] | None:
        """Sets up simulation timeline and validates window."""
        start_date = subj_df[TIMESTAMP_COL].min().normalize()
        end_date = subj_df[TIMESTAMP_COL].max().normalize()
        sim_window_start = start_date + pd.Timedelta(days=run_in_days)

        if sim_window_start >= end_date:
            return None

        total_days = (end_date - sim_window_start).days
        daily_timeline = pd.date_range(start=sim_window_start, end=end_date, freq="D")
        return daily_timeline, total_days

    def _build_feature_matrix(
        self,
        subj_df: pd.DataFrame,
        daily_timeline: pd.DatetimeIndex,
        trigger_codes: List[str],
    ) -> pd.DataFrame:
        """Creates feature matrix with cumulative triggers."""
        events_pivot = subj_df.pivot_table(
            index=TIMESTAMP_COL, columns=CONCEPT_COL, aggfunc="size", fill_value=0
        ).astype(bool)

        all_codes = set(trigger_codes) | set(events_pivot.columns)
        feature_matrix = events_pivot.reindex(
            index=daily_timeline, columns=list(all_codes), fill_value=False
        )

        return feature_matrix.cummax(axis=0)

    def _compute_event_probabilities(
        self,
        feature_matrix: pd.DataFrame,
        trigger_codes: List[str],
        trigger_weights: List[float],
        p_daily_base: float,
    ) -> np.ndarray:
        """Computes daily event probabilities using vectorized operations."""
        weights_array = np.array(trigger_weights)

        # Ensure feature_matrix has all trigger_codes as columns, filling missing with False
        present_trigger_codes = [
            code for code in trigger_codes if code in feature_matrix.columns
        ]
        trigger_matrix = feature_matrix[present_trigger_codes].values

        # Align weights with the present trigger codes
        present_weights_mask = np.isin(trigger_codes, present_trigger_codes)
        aligned_weights = weights_array[present_weights_mask]

        logit_p_days = logit(p_daily_base) + np.dot(trigger_matrix, aligned_weights)
        return expit(logit_p_days)

    def _compute_daily_prob(self, total_prob: float, num_days: int) -> float:
        """Converts total probability over a period into daily probability."""
        if num_days <= 0 or total_prob >= 1.0:
            return total_prob
        return 1 - (1 - total_prob) ** (1 / num_days)

    def _get_history_codes(
        self, subj_df: pd.DataFrame, assessment_time: pd.Timestamp
    ) -> set:
        """Extracts codes from subject history up to assessment time."""
        history_mask = subj_df[TIMESTAMP_COL] <= assessment_time
        return set(subj_df.loc[history_mask, CONCEPT_COL])

    def _calculate_probability(
        self, event_cfg, history_codes: set, is_exposed: bool = False
    ) -> float:
        """
        Calculates event probability based on history and, for outcomes, exposure status.
        """
        trigger_codes_array = np.array(list(event_cfg.trigger_codes))
        trigger_weights_array = np.array(list(event_cfg.trigger_weights))

        codes_present_mask = np.isin(trigger_codes_array, list(history_codes))
        trigger_effect_sum = np.sum(trigger_weights_array[codes_present_mask])

        logit_p = logit(event_cfg.p_base) + trigger_effect_sum

        if is_exposed and hasattr(event_cfg, "exposure_effect"):
            logit_p += event_cfg.exposure_effect

        return expit(logit_p)
