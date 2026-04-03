"""Semi-synthetic causal simulator using oracle features from real EHR data."""

import logging
import os
from os.path import join
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
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
    CONCEPT_COL,
    DEATH_CODE,
    PID_COL,
    TIMESTAMP_COL,
)
from corebehrt.functional.utils.filter import safe_control_pids
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.simulation.config_semisynthetic import (
    OutcomeModelConfig,
    SemiSyntheticSimulationConfig,
    TreatmentEffectConfig,
)
from corebehrt.modules.simulation.oracle_features import extract_oracle_features
from corebehrt.modules.simulation.plot import (
    plot_probability_distributions,
    plot_true_effects_vs_risk_differences,
)

logger = logging.getLogger("simulate")

ASSIGNED_INDEX_DATE_COL = "assigned_index_date"
EXCLUDED_PREFIXES = ("OUTCOME",)


class SemiSyntheticCausalSimulator:
    """Simulate causal outcomes from real EHR sequences using oracle features.

    Treatment assignment comes from the data (presence of exposure_code).
    Outcome probabilities are computed via a parametric model over
    hand-crafted oracle features extracted from pre-index history.
    """

    def __init__(self, config: SemiSyntheticSimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def simulate_dataset(self, shard_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Orchestrate the semi-synthetic simulation for a single data shard.

        Returns a dict of DataFrames matching the format produced by
        ``RealisticCausalSimulator.simulate_dataset``.
        """
        pids, is_exposed, index_dates = self._extract_treatment_and_index_dates(
            shard_df
        )
        if len(pids) == 0:
            return {}

        history_df = self._filter_to_pre_index(shard_df, index_dates)
        history_df, pids, is_exposed, index_dates = self._apply_min_num_codes(
            history_df, pids, is_exposed, index_dates
        )
        if len(pids) == 0:
            return {}

        features_df, _ = extract_oracle_features(
            history_df, pids, index_dates, self.config.features
        )

        ite_records, cf_records, all_factual_events, all_probas = (
            self._simulate_outcomes(features_df, pids, is_exposed, index_dates)
        )

        # Create exposure events for exposed patients
        if np.any(is_exposed):
            exposure_events = self._create_exposure_events(
                pids[is_exposed], index_dates
            )
            all_factual_events.append(exposure_events)

        return self._package_results(
            pids,
            ite_records,
            cf_records,
            all_factual_events,
            all_probas,
            is_exposed,
        )

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_treatment_and_index_dates(
        self, shard_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Identify exposed/control patients and their index dates."""
        # Drop patients without an assigned index date
        valid = shard_df.dropna(subset=[ASSIGNED_INDEX_DATE_COL])
        if valid.empty:
            return (
                np.array([]),
                np.array([], dtype=bool),
                pd.Series(dtype="datetime64[ns]"),
            )

        # Per-patient index dates
        index_dates = valid.groupby(PID_COL)[ASSIGNED_INDEX_DATE_COL].first()

        # Patients with at least one exposure event
        exposed_pids = set(
            valid.loc[valid[CONCEPT_COL] == self.config.exposure_code, PID_COL].unique()
        )

        pids = index_dates.index.values
        is_exposed = np.array([pid in exposed_pids for pid in pids])

        logger.info(
            f"Extracted {len(pids)} patients ({np.sum(is_exposed)} exposed, "
            f"{np.sum(~is_exposed)} control)"
        )
        return pids, is_exposed, index_dates

    def _filter_to_pre_index(
        self, df: pd.DataFrame, index_dates: pd.Series
    ) -> pd.DataFrame:
        """Keep only events before each patient's index date, excluding special codes."""
        df = df[df[PID_COL].isin(index_dates.index)].copy()
        patient_index = index_dates.reindex(df[PID_COL]).values
        df = df[df[TIMESTAMP_COL] <= patient_index]

        # Exclude special codes (keep BIRTH_CODE — needed for age computation)
        excluded_exact = {self.config.exposure_code, DEATH_CODE, "GENDER"}
        mask_exact = ~df[CONCEPT_COL].isin(excluded_exact)
        mask_prefix = ~df[CONCEPT_COL].str.startswith(EXCLUDED_PREFIXES)
        df = df[mask_exact & mask_prefix].copy()

        logger.info(
            f"Pre-index history: {df[PID_COL].nunique()} patients, {len(df)} events"
        )
        return df

    def _apply_min_num_codes(
        self,
        history_df: pd.DataFrame,
        pids: np.ndarray,
        is_exposed: np.ndarray,
        index_dates: pd.Series,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
        """Drop patients with fewer than min_num_codes unique codes."""
        if self.config.min_num_codes <= 1:
            return history_df, pids, is_exposed, index_dates

        code_counts = history_df.groupby(PID_COL)[CONCEPT_COL].nunique()
        keep_pids = set(code_counts[code_counts >= self.config.min_num_codes].index)
        before = len(pids)
        mask = np.array([pid in keep_pids for pid in pids])
        pids = pids[mask]
        is_exposed = is_exposed[mask]
        index_dates = index_dates.loc[pids]
        history_df = history_df[history_df[PID_COL].isin(keep_pids)].copy()

        logger.info(
            f"After min_num_codes filter (>={self.config.min_num_codes}): "
            f"{len(pids)} patients (dropped {before - len(pids)})"
        )
        return history_df, pids, is_exposed, index_dates

    # ------------------------------------------------------------------
    # Outcome simulation
    # ------------------------------------------------------------------

    def _simulate_outcomes(
        self,
        features_df: pd.DataFrame,
        pids: np.ndarray,
        is_exposed: np.ndarray,
        index_dates: pd.Series,
    ) -> Tuple[Dict, Dict, list, Dict]:
        n_patients = len(pids)
        ite_records = {PID_COL: pids}
        cf_records = {PID_COL: pids, EXPOSURE_COL: is_exposed.astype(int)}
        all_factual_events = []
        all_probas = {}

        for outcome_name, outcome_cfg in self.config.outcomes.items():
            eta_0 = self._compute_eta_0(features_df, outcome_cfg.outcome_model)
            tau = self._compute_tau(features_df, outcome_cfg.treatment_effect)
            noise = self.rng.normal(
                0, outcome_cfg.outcome_model.noise_scale, n_patients
            )

            p0 = expit(eta_0 + noise)
            p1 = expit(eta_0 + tau + noise)

            y1 = self.rng.binomial(1, p1)
            y0 = self.rng.binomial(1, p0)
            y_obs = np.where(is_exposed, y1, y0)

            all_probas[outcome_name] = {"P1": p1, "P0": p0}
            ite_records[f"ite_{outcome_name}"] = p1 - p0

            cf_records[f"{OUTCOME_COL}_{outcome_name}"] = y_obs
            cf_records[f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"] = y1
            cf_records[f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"] = y0
            cf_records[f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"] = p1
            cf_records[f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"] = p0

            # Create outcome events for patients with factual outcome == 1
            run_in_days = outcome_cfg.outcome_model.run_in_days
            patients_with_outcome = pids[y_obs == 1]
            if len(patients_with_outcome) > 0:
                outcome_timestamps = index_dates.loc[
                    patients_with_outcome
                ].values + pd.Timedelta(days=run_in_days)
                events = pd.DataFrame(
                    {
                        PID_COL: patients_with_outcome,
                        TIMESTAMP_COL: outcome_timestamps,
                        CONCEPT_COL: outcome_name,
                    }
                )
                all_factual_events.append(events)

        return ite_records, cf_records, all_factual_events, all_probas

    def _compute_eta_0(
        self,
        features_df: pd.DataFrame,
        outcome_model: OutcomeModelConfig,
    ) -> np.ndarray:
        """Compute the baseline log-odds: beta_0 + f(r_i) + interactions."""
        n = len(features_df)
        eta = np.full(n, outcome_model.beta_0)

        for name, coeff in outcome_model.coefficients.items():
            if name in features_df.columns:
                eta += coeff * features_df[name].values

        for interaction in outcome_model.interactions:
            feat_a = interaction.get("features", [None, None])[0]
            feat_b = interaction.get("features", [None, None])[1]
            coeff = interaction.get("coefficient", 0.0)
            if feat_a in features_df.columns and feat_b in features_df.columns:
                eta += coeff * features_df[feat_a].values * features_df[feat_b].values

        return eta

    def _compute_tau(
        self,
        features_df: pd.DataFrame,
        treatment_effect: TreatmentEffectConfig,
    ) -> np.ndarray:
        """Compute individual treatment effects on the logit scale."""
        n = len(features_df)
        if treatment_effect.mode == "constant":
            return np.full(n, treatment_effect.delta)

        # heterogeneous mode
        tau = np.full(n, treatment_effect.delta_0)
        for name, coeff in treatment_effect.heterogeneous_coefficients.items():
            if name in features_df.columns:
                tau += coeff * features_df[name].values
        return tau

    # ------------------------------------------------------------------
    # Result packaging (mirrors RealisticCausalSimulator._package_results)
    # ------------------------------------------------------------------

    def _package_results(
        self,
        pids,
        ite_records,
        cf_records,
        all_factual_events,
        all_probas,
        is_exposed,
    ) -> Dict[str, pd.DataFrame]:
        output_dir = self.config.paths.outcomes

        logger.info("Calculating and saving simulation statistics...")
        self._calculate_and_save_simulation_stats(
            pids, is_exposed, cf_records, output_dir
        )

        logger.info("Calculating theoretical maximum ROC AUC...")
        theoretical_aucs = self._calculate_theoretical_roc_auc(
            cf_records, is_exposed, output_dir
        )
        logger.info(f"Theoretical maximum ROC AUC: {theoretical_aucs}")

        # Plots
        figs_dir = join(output_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        logger.info("Plotting ground truth probability distributions...")
        plot_probability_distributions(all_probas, figs_dir)

        ite_df = pd.DataFrame(ite_records)
        cf_df = pd.DataFrame(cf_records)

        # Build true_effects_config for the comparison plot
        true_effects_config = {}
        for outcome_name, outcome_cfg in self.config.outcomes.items():
            te = outcome_cfg.treatment_effect
            om = outcome_cfg.outcome_model
            true_effects_config[outcome_name] = {
                "exposure_effect": te.delta if te.mode == "constant" else te.delta_0,
                "p_base": expit(om.beta_0),
            }

        logger.info("Plotting true effects vs observed risk differences...")
        plot_true_effects_vs_risk_differences(
            ite_df=ite_df,
            cf_df=cf_df,
            true_effects_config=true_effects_config,
            output_dir=figs_dir,
        )

        # Build output DataFrames
        output_dfs = {}
        if all_factual_events:
            events_df = pd.concat(all_factual_events, ignore_index=True)
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

    # ------------------------------------------------------------------
    # Helpers (same patterns as RealisticCausalSimulator)
    # ------------------------------------------------------------------

    def _create_exposure_events(
        self, exposed_pids: np.ndarray, index_dates: pd.Series
    ) -> pd.DataFrame:
        timestamps = index_dates.loc[exposed_pids].values
        df = pd.DataFrame(
            {
                PID_COL: exposed_pids,
                TIMESTAMP_COL: timestamps,
                CONCEPT_COL: EXPOSURE_COL,
            }
        )
        return df

    def _create_index_date_matching_df(
        self, exposure_df: pd.DataFrame, all_pids: np.ndarray
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
        return match_df

    def _calculate_and_save_simulation_stats(
        self,
        pids: np.ndarray,
        is_exposed: np.ndarray,
        cf_records: Dict[str, np.ndarray],
        output_dir: str,
    ):
        total_patients = len(pids)
        num_exposed = int(np.sum(is_exposed))
        num_control = total_patients - num_exposed

        outcome_stats = {}
        for outcome_name in self.config.outcomes:
            outcome_col = f"{OUTCOME_COL}_{outcome_name}"
            if outcome_col in cf_records:
                num_with = int(np.sum(cf_records[outcome_col]))
                outcome_stats[outcome_name] = {
                    "total_with_outcome": num_with,
                    "percentage_with_outcome": num_with / total_patients * 100,
                }

        stats_rows = [
            ["Statistic", "Value"],
            ["Total Patients", total_patients],
            ["Number Exposed", num_exposed],
            ["Number Control", num_control],
            ["Exposure Rate (%)", f"{num_exposed / total_patients * 100:.2f}"],
        ]
        for outcome_name, data in outcome_stats.items():
            stats_rows.append(
                [f"{outcome_name} - Total with Outcome", data["total_with_outcome"]]
            )
            stats_rows.append(
                [
                    f"{outcome_name} - Percentage with Outcome (%)",
                    f"{data['percentage_with_outcome']:.2f}",
                ]
            )

        stats_df = pd.DataFrame(stats_rows[1:], columns=stats_rows[0])
        os.makedirs(output_dir, exist_ok=True)
        stats_path = join(output_dir, "simulation_stats.csv")
        stats_df.to_csv(stats_path, index=False)

        logger.info(f"Simulation statistics saved to {stats_path}")
        logger.info(
            f"Total patients: {total_patients}, Exposed: {num_exposed}, Control: {num_control}"
        )
        for outcome_name, data in outcome_stats.items():
            logger.info(
                f"{outcome_name}: {data['total_with_outcome']} patients "
                f"({data['percentage_with_outcome']:.2f}%)"
            )

    def _calculate_theoretical_roc_auc(
        self,
        cf_records: Dict[str, np.ndarray],
        is_exposed: np.ndarray,
        output_dir: str,
    ) -> Dict[str, float]:
        theoretical_aucs = {}
        results_data = []

        for outcome_name in self.config.outcomes:
            outcome_col = f"{OUTCOME_COL}_{outcome_name}"
            p_exposed_col = f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"
            p_control_col = f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"

            if outcome_col not in cf_records:
                continue
            if p_exposed_col not in cf_records or p_control_col not in cf_records:
                continue

            y_true = cf_records[outcome_col]
            p_treated = cf_records[p_exposed_col]
            p_control = cf_records[p_control_col]
            y_prob_factual = np.where(is_exposed, p_treated, p_control)

            if len(np.unique(y_true)) > 1:
                auc_factual = roc_auc_score(y_true, y_prob_factual)
                auc_treated = roc_auc_score(y_true, p_treated)
                auc_control = roc_auc_score(y_true, p_control)
                theoretical_aucs[outcome_name] = auc_factual

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

        if results_data:
            results_df = pd.DataFrame(results_data)
            os.makedirs(output_dir, exist_ok=True)
            results_path = join(output_dir, "theoretical_max_roc_auc.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Theoretical ROC AUC results saved to {results_path}")

        return theoretical_aucs
