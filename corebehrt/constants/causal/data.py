# Basic outcome and probability columns
PROBAS = "probas"
TARGETS = "targets"
OUTCOMES = "outcomes"
EXPOSURES = "exposures"
COHORT = "cohort"
DATA = "data"


CONTROL_PID_COL = "control_subject_id"
EXPOSED_PID_COL = "exposed_subject_id"

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
TRUE_EFFECT_COL = "true_effect"

INDEX_DATE = "index_date"


# New flow-related constants
FLOW = "flow"
FLOW_INITIAL = "initial"
FLOW_AFTER_AGE = "after_age"
FLOW_AFTER_STRICT = "after_strict"
FLOW_FINAL = "final"

CALIBRATION_COLLAPSE_THRESHOLD = 0.01


# Prepare finetune
GROUP_COL = "group"
DEATH_COL = "death"
NON_COMPLIANCE_COL = "non_compliance"

# Training
EXPOSURE_TARGET = "exposure_target"
OUTCOME_TARGETS = "outcome_targets"
