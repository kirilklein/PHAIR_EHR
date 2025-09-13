# Basic outcome and probability columns
PROBAS = "probas"
TARGETS = "targets"
OUTCOMES = "outcomes"
EXPOSURES = "exposures"
COHORT = "cohort"
DATA = "data"


CONTROL_PID_COL = "control_subject_id"
EXPOSED_PID_COL = "exposed_subject_id"
BIRTH_YEAR_COL = "birth_year"

START_COL = "start"
END_COL = "end"
START_TIME_COL = "start_time"
END_TIME_COL = "end_time"

# Prepare causal finetune data
EXPOSURE = "exposure"
OUTCOME = "outcome"
CF_OUTCOME = "cf_outcome"

# Predicted outcome probabilities
PROB_KEY = "prob"  # Overall predicted outcome probability
PROB_T_KEY = "prob_t"  # Predicted outcome probability under treatment
PROB_C_KEY = "prob_c"  # Predicted outcome probability under control

# Treatment/exposure related
EXPOSURE_COL = "exposure"
PS_COL = "ps"  # Propensity score
OUTCOME_COL = "outcome"

# Counterfactual probabilities
PROBAS_EXPOSED = "probas_exposed"
PROBAS_CONTROL = "probas_control"
CF_PROBAS = "cf_probas"

# Simulation specific columns
SIMULATED_OUTCOME_EXPOSED = "Y1"  # Simulated outcome under treatment
SIMULATED_OUTCOME_CONTROL = "Y0"  # Simulated outcome under control
SIMULATED_PROBAS_EXPOSED = "P1"  # Simulated probability under treatment
SIMULATED_PROBAS_CONTROL = "P0"  # Simulated probability under control

# Treatment effect
INDEX_DATE = "index_date"


# New flow-related constants
FLOW = "flow"
FLOW_INITIAL = "initial"
FLOW_AFTER_AGE = "after_age"
FLOW_AFTER_STRICT = "after_strict"
FLOW_FINAL = "final"

CALIBRATION_COLLAPSE_THRESHOLD = 0.01
PROBAS_ROUND_DIGIT = 9  # For saving probas
EFFECT_ROUND_DIGIT = 9  # FOr saving estimates

# Prepare finetune
GROUP_COL = "group"
DEATH_COL = "death"
NON_COMPLIANCE_COL = "non_compliance"

# Training
EXPOSURE_TARGET = "exposure_target"
OUTCOME_TARGETS = "outcome_targets"

STATUS = "status"


class EffectColumns:
    method = "method"
    effect = "effect"
    true_effect = "true_effect"
    ps_bias = "ps_bias"
    y_bias = "y_bias"
    std_err = "std_err"
    CI95_lower = "CI95_lower"
    CI95_upper = "CI95_upper"
    effect_1 = "effect_1"
    effect_0 = "effect_0"
    outcome = OUTCOME

    @classmethod
    def get_columns(cls):
        """
        Returns a list of all custom attributes defined in the class,
        """
        return [
            name
            for name, value in cls.__dict__.items()
            if not name.startswith("_") and not callable(value)
        ]


class TMLEAnalysisColumns:
    initial_effect_1 = "initial_effect_1"
    initial_effect_0 = "initial_effect_0"
    adjustment_1 = "adjustment_1"
    adjustment_0 = "adjustment_0"
    method = "method"
    outcome = "outcome"
    effect = "effect"
    initial_effect = "initial_effect"
    adjustment = "adjustment"

    @classmethod
    def get_columns(cls):
        """
        Returns a list of all custom attributes defined in the class,
        """
        return [
            name
            for name, value in cls.__dict__.items()
            if not name.startswith("_")
            and not callable(value)
            and name != "get_columns"
        ]
