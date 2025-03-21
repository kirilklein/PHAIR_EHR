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
