# Entries/Dataframe Columns
PROBAS = "probas"
TARGETS = "targets"

OUTCOMES = "outcomes"

SIMULATED_OUTCOME_EXPOSED = "Y1"
SIMULATED_OUTCOME_CONTROL = "Y0"
SIMULATED_PROBAS_EXPOSED = "P1"
SIMULATED_PROBAS_CONTROL = "P0"

PROBAS_EXPOSED = "probas_exposed"
PROBAS_CONTROL = "probas_control"
CF_PROBAS = "cf_probas"

TRUE_EFFECT_COL = "true_effect"

EXPOSURE_COL = "exposure"
PS_COL = "ps"


# Files
PREDICTIONS_FILE = "predictions_and_targets.csv"
CALIBRATED_PREDICTIONS_FILE = "predictions_and_targets_calibrated.csv"

TIMESTAMP_OUTCOME_FILE = "outcomes_with_timestamps.csv"
SIMULATION_RESULTS_FILE = "probas_and_outcomes.csv"

EXPERIMENT_DATA_FILE = "experiment_data.parquet"
EXPERIMENT_STATS_FILE = "experiment_stats.csv"
ESTIMATE_RESULTS_FILE = "estimate_results.csv"

PROB_KEY = "prob"  # Overall predicted outcome probability
PROB_T_KEY = "prob_t"  # Predicted outcome probability under treatment
PROB_C_KEY = "prob_c"  # Predicted outcome probability under control
