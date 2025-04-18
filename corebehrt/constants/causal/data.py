# Basic outcome and probability columns
PROBAS = "probas"
TARGETS = "targets"
OUTCOMES = "outcomes"

# Predicted outcome probabilities
PROB_KEY = "prob"  # Overall predicted outcome probability
PROB_T_KEY = "prob_t"  # Predicted outcome probability under treatment
PROB_C_KEY = "prob_c"  # Predicted outcome probability under control

# Treatment/exposure related
EXPOSURE_COL = "exposure"
PS_COL = "ps"  # Propensity score

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
FLOW_AFTER_MINIMUM_ONE = "after_minimum_one"
FLOW_FINAL = "final"
FLOW_AFTER_UNIQUE_CODES = "after_unique_codes"

CALIBRATION_COLLAPSE_THRESHOLD = 0.01
