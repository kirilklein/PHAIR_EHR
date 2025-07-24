import pandas as pd

from corebehrt.constants.causal.data import (
    EXPOSURE_COL,
    OUTCOME,
    PS_COL,
    SIMULATED_OUTCOME_CONTROL,
    SIMULATED_OUTCOME_EXPOSED,
    SIMULATED_PROBAS_CONTROL,
    SIMULATED_PROBAS_EXPOSED,
    EffectColumns,
)
from corebehrt.constants.data import PID_COL
from corebehrt.functional.causal.effect import compute_effect_from_counterfactuals
from corebehrt.functional.causal.estimate import (
    calculate_risk_difference,
    calculate_risk_ratio,
)
from CausalEstimate.filter.propensity import filter_common_support


def append_unadjusted_effect(df: pd.DataFrame, effect_df: pd.DataFrame) -> pd.DataFrame:
    exposed = df[df[EXPOSURE_COL] == 1]
    unexposed = df[df[EXPOSURE_COL] == 0]

    # Calculate basic statistics
    risk_exposed = exposed[OUTCOME].mean()
    risk_unexposed = unexposed[OUTCOME].mean()
    n_exposed = len(exposed)
    n_unexposed = len(unexposed)

    # Calculate both measures
    rd_row = calculate_risk_difference(
        risk_exposed, risk_unexposed, n_exposed, n_unexposed
    )
    rr_row = calculate_risk_ratio(risk_exposed, risk_unexposed, n_exposed, n_unexposed)
    # Append both rows to existing results
    return pd.concat([effect_df, rd_row, rr_row], ignore_index=True)


def append_true_effect(
    analysis_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    outcome_name: str,
    effect_type: str,
    common_support_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Add ground truth effect estimates from simulated counterfactual outcomes.

    Uses the same analysis cohort as other estimators for consistency.
    Now supports both combined and individual counterfactual outcome files.

    Adds the true effect to the effect_df. (TRUE_EFFECT_COL)
    """
    combined = pd.merge(
        analysis_df[[PID_COL, PS_COL]], counterfactual_df, on=PID_COL, how="inner"
    )
    combined = filter_common_support(
        combined, PS_COL, EXPOSURE_COL, common_support_threshold
    )
    cf_outcomes = prepare_counterfactual_data_for_outcome(combined, outcome_name)
    true_effect = compute_effect_from_counterfactuals(cf_outcomes, effect_type)
    effect_df[EffectColumns.true_effect] = true_effect
    return effect_df


def prepare_counterfactual_data_for_outcome(
    counterfactual_df: pd.DataFrame, outcome_name: str
) -> pd.DataFrame:
    """
    Prepare counterfactual data for a specific outcome from the combined file.
    """
    cf_data = {
        PID_COL: counterfactual_df[PID_COL],
        SIMULATED_OUTCOME_CONTROL: counterfactual_df[
            f"{SIMULATED_OUTCOME_CONTROL}_{outcome_name}"
        ],
        SIMULATED_OUTCOME_EXPOSED: counterfactual_df[
            f"{SIMULATED_OUTCOME_EXPOSED}_{outcome_name}"
        ],
        SIMULATED_PROBAS_CONTROL: counterfactual_df[
            f"{SIMULATED_PROBAS_CONTROL}_{outcome_name}"
        ],
        SIMULATED_PROBAS_EXPOSED: counterfactual_df[
            f"{SIMULATED_PROBAS_EXPOSED}_{outcome_name}"
        ],
        EXPOSURE_COL: counterfactual_df[EXPOSURE_COL],
    }
    return pd.DataFrame(cf_data)
