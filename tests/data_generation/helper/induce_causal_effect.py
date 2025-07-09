from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm import tqdm
from tests.data_generation.helper.config import SimulationConfig


class CausalSimulator:
    """
    Handles a multi-phase, time-to-event causal simulation for EHR data,
    driven by a hierarchical configuration object. Includes vectorized,
    age-based index date matching for controls.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.first_exposure_dates = {}

    def simulate_dataset(
        self, df: pd.DataFrame, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the full causal simulation with a three-pass approach."""
        np.random.seed(seed)

        if "date_of_birth" not in df.columns:
            dob_events = df[df["code"] == "DOB"][
                ["subject_id", "time"]
            ].drop_duplicates(subset="subject_id")
            dob_events = dob_events.rename(columns={"time": "date_of_birth"})
            df = df.merge(dob_events, on="subject_id")

        subjects_as_dfs = [group for _, group in df.groupby("subject_id")]
        subjects_with_exposure = self._simulate_all_exposures(subjects_as_dfs)
        all_subjects_with_index_dates = self._assign_matched_index_dates(
            subjects_with_exposure
        )
        return self._simulate_all_outcomes_dataset(all_subjects_with_index_dates)

    def _simulate_all_exposures(
        self, subject_dfs: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Simulates exposure processes for all subjects."""
        subjects_with_exposure = []
        self.first_exposure_dates = {}
        for subj_df in tqdm(subject_dfs, desc="Simulating exposures"):
            subj_df_sorted = subj_df.sort_values(
                "time", na_position="first"
            ).reset_index(drop=True)
            df_with_exposure = self._simulate_exposure_process(subj_df_sorted)
            subjects_with_exposure.append(df_with_exposure)
        return subjects_with_exposure

    def _simulate_exposure_process(self, subj_df: pd.DataFrame) -> pd.DataFrame:
        """Stores first exposure date in a dictionary, using the correct subject_id."""
        cfg = self.config.exposure
        subject_id = subj_df.iloc[0]["subject_id"]
        df_with_temp_exposure = self._simulate_time_to_first_event(
            subj_df,
            cfg.p_base,
            cfg.trigger_codes,
            cfg.trigger_weights,
            f"TEMP_{cfg.code}",
            cfg.run_in_days,
        )
        temp_exposure_events = df_with_temp_exposure[
            df_with_temp_exposure["code"] == f"TEMP_{cfg.code}"
        ]
        if temp_exposure_events.empty:
            return subj_df

        first_exposure_date = temp_exposure_events["time"].min()
        self.first_exposure_dates[subject_id] = first_exposure_date
        compliance_end_date = self._get_random_compliance_end_date(
            first_exposure_date, subj_df["time"].max()
        )
        exposure_dates = self._generate_regular_exposures(
            first_exposure_date, compliance_end_date, cfg.compliance_interval_days
        )
        return self._replace_temp_with_final_exposures(
            df_with_temp_exposure, exposure_dates, cfg.code
        )

    def _assign_matched_index_dates(
        self, subject_dfs: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Assigns index dates via an efficient, vectorized age-matching algorithm."""
        if not self.first_exposure_dates:
            return self._assign_fallback_index_dates(subject_dfs)

        exposed_subjects, control_subjects = [], []
        max_outcome_fu = max(cfg.run_in_days for cfg in self.config.outcomes.values())
        run_in_days = self.config.exposure.run_in_days

        for subj_df in subject_dfs:
            subject_id = subj_df.iloc[0]["subject_id"]
            dob = subj_df.iloc[0]["date_of_birth"]
            if subject_id in self.first_exposure_dates:
                index_date = self.first_exposure_dates[subject_id]
                age_at_index = (index_date - dob).days / 365.25
                exposed_subjects.append(
                    {
                        "exp_subject_id": subject_id,
                        "exp_index_date": index_date,
                        "exp_age_at_index": age_at_index,
                    }
                )
            else:
                start_date, end_date = subj_df["time"].min(), subj_df["time"].max()
                control_subjects.append(
                    {
                        "ctrl_subject_id": subject_id,
                        "ctrl_dob": dob,
                        "valid_start": start_date + pd.Timedelta(days=run_in_days),
                        "valid_end": end_date - pd.Timedelta(days=max_outcome_fu),
                    }
                )

        if not control_subjects or not exposed_subjects:
            return self._assign_fallback_index_dates(subject_dfs)

        exposed_df = pd.DataFrame(exposed_subjects)
        controls_df = pd.DataFrame(control_subjects)
        df_cross = pd.merge(controls_df, exposed_df, how="cross")
        ctrl_potential_age = (
            df_cross["exp_index_date"] - df_cross["ctrl_dob"]
        ).dt.days / 365.25
        df_cross["age_diff"] = abs(ctrl_potential_age - df_cross["exp_age_at_index"])
        best_match_indices = df_cross.groupby("ctrl_subject_id")["age_diff"].idxmin()
        best_matches = df_cross.loc[best_match_indices].reset_index(drop=True)
        best_matches["assigned_index_date"] = best_matches.apply(
            lambda row: max(
                row["valid_start"], min(row["exp_index_date"], row["valid_end"])
            ),
            axis=1,
        )
        matched_control_index_dates = pd.Series(
            best_matches.assigned_index_date.values, index=best_matches.ctrl_subject_id
        ).to_dict()

        final_subject_dfs = []
        for subj_df in subject_dfs:
            subject_id = subj_df.iloc[0]["subject_id"]
            new_df = subj_df.copy()
            if subject_id in self.first_exposure_dates:
                new_df["assigned_index_date"] = self.first_exposure_dates[subject_id]
            elif subject_id in matched_control_index_dates:
                new_df["assigned_index_date"] = matched_control_index_dates[subject_id]
            final_subject_dfs.append(new_df)
        return final_subject_dfs

    def _assign_fallback_index_dates(
        self, subject_dfs: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Assigns a simple index date if no exposures occurred or matching is not possible."""
        final_subject_dfs = []
        for subj_df in subject_dfs:
            new_df = subj_df.copy()
            start_date = new_df["time"].min()
            new_df["assigned_index_date"] = start_date + pd.Timedelta(
                days=self.config.exposure.run_in_days
            )
            final_subject_dfs.append(new_df)
        return final_subject_dfs

    def _simulate_all_outcomes_dataset(
        self, subjects_with_index_dates: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulates outcomes for all subjects using pre-assigned index dates."""
        all_subject_dfs, all_ite_records = [], []
        for df_with_index_date in tqdm(
            subjects_with_index_dates, desc="Simulating outcomes"
        ):
            if "assigned_index_date" not in df_with_index_date.columns:
                continue
            final_df, ite_record = self._simulate_subject_outcomes(df_with_index_date)
            all_subject_dfs.append(final_df)
            if ite_record:
                all_ite_records.append(ite_record)
        if not all_subject_dfs:
            return pd.DataFrame(), pd.DataFrame()
        simulated_df = pd.concat(all_subject_dfs, ignore_index=True)
        ite_df = pd.DataFrame(all_ite_records) if all_ite_records else pd.DataFrame()
        return simulated_df, ite_df

    def _simulate_subject_outcomes(
        self, subj_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """Uses the pre-assigned index date to simulate outcomes."""
        subject_info = self._extract_subject_info(subj_df)
        if not subject_info:
            return subj_df, {}
        _, end_date, has_exposure, _, subject_id = subject_info
        index_date = subj_df["assigned_index_date"].iloc[0]
        df_with_outcomes = subj_df.copy()
        ite_record = {"subject_id": subject_id, "has_exposure": int(has_exposure)}
        for outcome_cfg in self.config.outcomes.values():
            assessment_time = index_date + pd.Timedelta(days=outcome_cfg.run_in_days)
            if assessment_time > end_date:
                assessment_time = end_date - pd.Timedelta(days=1)
            df_with_outcomes, ite_value = self._simulate_single_outcome_complete(
                df_with_outcomes, outcome_cfg, assessment_time
            )
            ite_record[f"ite_{outcome_cfg.code}"] = ite_value
        return df_with_outcomes, ite_record

    def _extract_subject_info(self, subj_df: pd.DataFrame) -> Optional[Tuple]:
        """Extracts key information about a subject."""
        if (
            subj_df.empty
            or "time" not in subj_df.columns
            or subj_df["time"].dropna().empty
        ):
            return None
        start_date = subj_df["time"].dropna().min().normalize()
        end_date = subj_df["time"].dropna().max().normalize()
        exposure_events = subj_df[subj_df["code"] == self.config.exposure.code]
        has_exposure = not exposure_events.empty
        first_exposure_date = exposure_events["time"].min() if has_exposure else None
        subject_id = subj_df.iloc[0]["subject_id"]
        return start_date, end_date, has_exposure, first_exposure_date, subject_id

    def _simulate_single_outcome_complete(
        self, subj_df: pd.DataFrame, outcome_cfg, assessment_time: pd.Timestamp
    ) -> Tuple[pd.DataFrame, float]:
        """Simulates a single outcome: calculates ITE and simulates factual outcome."""
        history_codes = self._get_history_codes(subj_df, assessment_time)
        is_exposed = self.config.exposure.code in history_codes
        ite = self._calculate_ite(outcome_cfg, history_codes)
        p_factual = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed
        )
        if np.random.binomial(1, p_factual):
            subj_df = self._add_outcome_event(
                subj_df, outcome_cfg.code, assessment_time
            )
        return subj_df, ite

    def _get_history_codes(
        self, subj_df: pd.DataFrame, assessment_time: pd.Timestamp
    ) -> set:
        """Extracts codes from subject history up to assessment time."""
        history_mask = subj_df["time"] <= assessment_time
        return set(subj_df.loc[history_mask, "code"])

    def _calculate_ite(self, outcome_cfg, history_codes: set) -> float:
        """Calculates Individual Treatment Effect."""
        p_if_treated = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=True
        )
        p_if_control = self._calculate_outcome_probability(
            outcome_cfg, history_codes, is_exposed=False
        )
        return p_if_treated - p_if_control

    def _calculate_outcome_probability(
        self, outcome_cfg, history_codes: set, is_exposed: bool
    ) -> float:
        """Calculates outcome probability given history and exposure status."""
        trigger_codes_array = np.array(list(outcome_cfg.trigger_codes))
        trigger_weights_array = np.array(list(outcome_cfg.trigger_weights))
        codes_present_mask = np.isin(trigger_codes_array, list(history_codes))
        trigger_effect_sum = np.sum(trigger_weights_array[codes_present_mask])
        logit_p = logit(outcome_cfg.p_base) + trigger_effect_sum
        if is_exposed:
            logit_p += outcome_cfg.exposure_effect
        return expit(logit_p)

    def _add_outcome_event(
        self, subj_df: pd.DataFrame, outcome_code: str, assessment_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Adds an outcome event to the subject dataframe."""
        new_event = pd.DataFrame(
            {
                "subject_id": [subj_df.iloc[0]["subject_id"]],
                "time": [assessment_time],
                "code": [outcome_code],
            }
        )
        return (
            pd.concat([subj_df, new_event], ignore_index=True)
            .sort_values("time")
            .reset_index(drop=True)
        )

    def _replace_temp_with_final_exposures(
        self,
        df_with_temp: pd.DataFrame,
        exposure_dates: pd.DatetimeIndex,
        exposure_code: str,
    ) -> pd.DataFrame:
        """Replaces temporary exposure events with final ones."""
        df_clean = df_with_temp[df_with_temp["code"] != f"TEMP_{exposure_code}"]
        if exposure_dates.empty:
            return df_clean
        return self._add_events_to_dataframe(df_clean, exposure_dates, exposure_code)

    def _add_events_to_dataframe(
        self, df: pd.DataFrame, event_times: pd.DatetimeIndex, event_code: str
    ) -> pd.DataFrame:
        """Generic method to add multiple events to a dataframe."""
        if event_times.empty:
            return df
        new_events = pd.DataFrame(
            {
                "subject_id": df.iloc[0]["subject_id"],
                "time": event_times,
                "code": event_code,
            }
        )
        return (
            pd.concat([df, new_events], ignore_index=True)
            .sort_values("time", na_position="first")
            .reset_index(drop=True)
        )

    def _simulate_time_to_first_event(
        self,
        subj_df: pd.DataFrame,
        p_total_base: float,
        trigger_codes: List[str],
        trigger_weights: List[float],
        event_name: str,
        run_in_days: int,
    ) -> pd.DataFrame:
        """Simulates time to first event occurrence."""
        timeline_info = self._setup_simulation_timeline(subj_df, run_in_days)
        if timeline_info is None:
            return subj_df
        daily_timeline, total_days = timeline_info
        p_daily_base = self._compute_daily_prob(p_total_base, total_days)
        feature_matrix = self._build_feature_matrix(
            subj_df, daily_timeline, trigger_codes
        )
        event_probabilities = self._compute_event_probabilities(
            feature_matrix, trigger_codes, trigger_weights, p_daily_base
        )
        return self._simulate_and_add_event(
            subj_df, daily_timeline, event_probabilities, event_name
        )

    def _get_random_compliance_end_date(
        self, first_exposure_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.Timestamp:
        """Selects a compliance end date using a stable linear weighting."""
        earliest_end = first_exposure_date + pd.Timedelta(
            days=self.config.exposure.min_compliance_days
        )
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

    def _setup_simulation_timeline(
        self, subj_df: pd.DataFrame, run_in_days: int
    ) -> Optional[Tuple[pd.DatetimeIndex, int]]:
        """Sets up simulation timeline and validates window."""
        start_date = subj_df["time"].min().normalize()
        end_date = subj_df["time"].max().normalize()
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
            index="time", columns="code", aggfunc="size", fill_value=0
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
        trigger_matrix = feature_matrix[trigger_codes].values
        logit_p_days = logit(p_daily_base) + np.dot(trigger_matrix, weights_array)
        return expit(logit_p_days)

    def _simulate_and_add_event(
        self,
        subj_df: pd.DataFrame,
        daily_timeline: pd.DatetimeIndex,
        event_probabilities: np.ndarray,
        event_name: str,
    ) -> pd.DataFrame:
        """Simulates event occurrence and adds to dataframe if it occurs."""
        event_draws = np.random.binomial(1, event_probabilities)
        if event_draws.any():
            event_idx = np.argmax(event_draws)
            event_time = daily_timeline[event_idx]
            return self._add_events_to_dataframe(
                subj_df, pd.DatetimeIndex([event_time]), event_name
            )
        return subj_df

    def _compute_daily_prob(self, total_prob: float, num_days: int) -> float:
        """Converts total probability over a period into daily probability."""
        if num_days <= 0 or total_prob >= 1.0:
            return total_prob
        return 1 - (1 - total_prob) ** (1 / num_days)
