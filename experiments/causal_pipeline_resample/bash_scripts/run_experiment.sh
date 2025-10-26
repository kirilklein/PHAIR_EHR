#!/bin/bash

# ========================================
# Causal Pipeline RESAMPLING Experiment Runner (Baseline + BERT)
# This version samples a different subset of the cohort for each run
# ========================================

# --- Script Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# Default timeouts in seconds
TIMEOUT_SIMULATE=1800   # 30 mins
TIMEOUT_COHORT=900      # 15 mins
TIMEOUT_PREPARE=1200    # 20 mins
TIMEOUT_TRAIN_BL=3600   # 1 hour
TIMEOUT_CAL_BL=1800     # 30 mins
TIMEOUT_EST_BL=1800     # 30 mins
TIMEOUT_FINETUNE_BERT=7200 # 2 hours
TIMEOUT_CAL_BERT=1800   # 30 mins
TIMEOUT_EST_BERT=1800   # 30 mins
TIMEOUT_FACTOR=1.0      # Default timeout scaling factor

# --- Argument Parsing ---
if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <experiment_name> [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --timeout-factor F    Multiply all timeouts by this factor (e.g., 0.1 for quick tests)"
    echo "========================================"
    echo "Full Causal Pipeline Experiment Runner"
    echo "========================================"
    echo ""
    echo "Usage: $0 <experiment_name> [OPTIONS]"
    echo ""
    echo "ARGUMENTS:"
    echo "  experiment_name       Name of the experiment to run"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message"
    echo "  --baseline-only         Run only baseline (CatBoost) pipeline"
    echo "  --bert-only             Run only BERT pipeline (requires baseline data)"
    echo "  --overwrite             Force re-run all steps (default: skip completed steps)"
    echo "  --experiment-dir|-e DIR Base directory for experiments (default: ./outputs/causal/sim_study_sampling/runs)"
    echo "  --base-configs-dir DIR  Custom base configs directory (default: ../base_configs)"
    echo "  --base-seed N           Base seed for sampling (default: 42)"
    echo "  --sample-fraction F     Fraction of patients to sample (0 < F <= 1, mutually exclusive with --sample-size)"
    echo "  --sample-size N         Absolute number of patients to sample (takes precedence, mutually exclusive with --sample-fraction)"
    echo "  (no options)            Run both baseline and BERT pipelines"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 ce1_cy1_y0_i0"
    echo "    > Runs both baseline and BERT pipelines for experiment ce1_cy1_y0_i0"
    echo ""
    echo "  $0 ce1_cy1_y0_i0 --baseline-only"
    echo "    > Runs only baseline pipeline for experiment ce1_cy1_y0_i0"
    echo ""
    echo "  $0 ce1_cy1_y0_i0 --bert-only"
    echo "    > Runs only BERT pipeline for experiment ce1_cy1_y0_i0 (baseline data must exist)"
    echo ""
    echo "NOTES:"
    echo "  - This is the RESAMPLING variant: samples from MEDS → simulates → selects cohort for each run"
    echo "  - No pre-built cohort needed; sampling happens from raw MEDS data"
    echo "  - Seed is calculated as: base_seed + run_number (extracted from run_id)"
    echo "  - Experiment configs are read from: ../experiment_configs/<experiment_name>.yaml"
    echo "  - Generated configs are saved to: ../generated_configs/<experiment_name>/"
    echo "  - Results are saved to: ../../../outputs/causal/sim_study_sampling/runs/<experiment_name>/"
    echo "  - Use Ctrl+C to stop the experiment at any time"
    echo ""
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

# Default values
RUN_BASELINE=true
RUN_BERT=true
RUN_ID="run_01"
OVERWRITE=false  # Safe default: don't overwrite existing results
EXPERIMENTS_DIR="./outputs/causal/sim_study_sampling/runs"
BASE_CONFIGS_DIR=""  # Empty means use default in generate_configs.py
BASE_SEED=42
SAMPLE_FRACTION=""   # Either fraction or size must be provided
SAMPLE_SIZE=""       # Takes precedence over fraction
MEDS_DATA="./example_data/synthea_meds_causal"
FEATURES_DATA="./outputs/causal/data/features"
TOKENIZED_DATA="./outputs/causal/data/tokenized"
PRETRAIN_MODEL="./outputs/causal/pretrain/model"


while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-only) RUN_BERT=false; shift ;;
        --bert-only) RUN_BASELINE=false; shift ;;
        --run_id) RUN_ID="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --experiment-dir|-e|--experiment_dir) EXPERIMENTS_DIR="$2"; shift 2 ;;
        --base-configs-dir|--base_configs_dir) BASE_CONFIGS_DIR="$2"; shift 2 ;;
        --timeout-factor|--timeout_factor) TIMEOUT_FACTOR="$2"; shift 2 ;;
        --base-seed|--base_seed) BASE_SEED="$2"; shift 2 ;;
        --sample-fraction|--sample_fraction) SAMPLE_FRACTION="$2"; shift 2 ;;
        --sample-size|--sample_size) SAMPLE_SIZE="$2"; shift 2 ;;
        --meds)
            MEDS_DATA="$2"
            shift 2
            ;;
        --features)
            FEATURES_DATA="$2"
            shift 2
            ;;
        --tokenized)
            TOKENIZED_DATA="$2"
            shift 2
            ;;
        --pretrain-model|--pretrain_model)
            PRETRAIN_MODEL="$2"
            shift 2
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Running Experiment: $RUN_ID/$EXPERIMENT_NAME"
echo "Timeout Factor: $TIMEOUT_FACTOR"
echo "========================================"

run_step() {
    local step_name=$1
    local python_module=$2
    local config_name=$3
    local check_file=$4
    local timeout_secs=$5

    # Scale timeout by the factor (using awk for floating point arithmetic)
    local effective_timeout
    effective_timeout=$(echo "$timeout_secs $TIMEOUT_FACTOR" | awk '{result = $1 * $2; printf "%d", (result < 1) ? 1 : result}')
    
    echo "DEBUG: Timeout calculation: $timeout_secs * $TIMEOUT_FACTOR = $effective_timeout seconds"

    echo "" # Add spacing
    if [ "$OVERWRITE" = "false" ] && [ -f "$check_file" ]; then
        echo "==== Skipping $step_name (output already exists) ===="
        return
    fi

    echo "==== Running $step_name... ===="
    local config_path="experiments/causal_pipeline_resample/generated_configs/$EXPERIMENT_NAME/$config_name.yaml"

    timeout "$effective_timeout" python -m "$python_module" --config_path "$config_path"
    local exit_code=$?

    if [ $exit_code -eq 124 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Step '$step_name' timed out after $effective_timeout seconds."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 124
    elif [ $exit_code -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Step '$step_name' failed with exit code $exit_code."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit $exit_code
    fi
    echo "==== Success: $step_name completed. ===="
}


# --- Main Execution ---

# 1. Generate Configs
echo "Step 1: Generating experiment configs..."
CONFIG_GEN_CMD="python ../python_scripts/generate_configs.py \"$EXPERIMENT_NAME\" --run_id \"$RUN_ID\" --experiments_dir \"$EXPERIMENTS_DIR\" --meds \"$MEDS_DATA\" --features \"$FEATURES_DATA\" --tokenized \"$TOKENIZED_DATA\" --pretrain-model \"$PRETRAIN_MODEL\" --base-seed \"$BASE_SEED\""
if [ -n "$SAMPLE_SIZE" ]; then
    CONFIG_GEN_CMD="$CONFIG_GEN_CMD --sample-size \"$SAMPLE_SIZE\""
elif [ -n "$SAMPLE_FRACTION" ]; then
    CONFIG_GEN_CMD="$CONFIG_GEN_CMD --sample-fraction \"$SAMPLE_FRACTION\""
fi
if [ -n "$BASE_CONFIGS_DIR" ]; then
    CONFIG_GEN_CMD="$CONFIG_GEN_CMD --base-configs-dir \"$BASE_CONFIGS_DIR\""
fi
eval $CONFIG_GEN_CMD
if [ $? -ne 0 ]; then echo "ERROR: Config generation failed."; exit 1; fi
echo "All configs generated successfully."
echo ""

# 2. Change to Project Root
cd ../../..
export PYTHONPATH="$PWD:$PYTHONPATH"

# 3. Run Data Preparation Steps (ALWAYS run for resampling experiments)
echo "Step 2: Running Data Preparation Pipeline with Resampling..."
echo "NOTE: Each run samples from MEDS data, then simulates and selects cohort"
TARGET_DIR="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
run_step "simulate_outcomes_with_sampling" "corebehrt.main_causal.simulate_with_sampling" "simulation" "$TARGET_DIR/simulated_outcomes/counterfactuals.csv" $TIMEOUT_SIMULATE
run_step "select_cohort" "corebehrt.main_causal.select_cohort_full" "select_cohort" "$TARGET_DIR/cohort/pids.pt" $TIMEOUT_COHORT
run_step "prepare_finetune_data" "corebehrt.main_causal.prepare_ft_exp_y" "prepare_finetune" "$TARGET_DIR/prepared_data/patients.pt" $TIMEOUT_PREPARE

# 4. Run Baseline Pipeline (if enabled)
if [ "$RUN_BASELINE" = "true" ]; then
    echo ""
    echo "========================================"
    echo "==== Running Baseline Pipeline ===="
    echo "========================================"
    TARGET_DIR="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
    run_step "train_baseline" "corebehrt.main_causal.train_baseline" "train_baseline" "$TARGET_DIR/models/baseline/combined_predictions.csv" $TIMEOUT_TRAIN_BL
    run_step "calibrate (Baseline)" "corebehrt.main_causal.calibrate_exp_y" "calibrate" "$TARGET_DIR/models/baseline/calibrated/combined_calibrated_predictions.csv" $TIMEOUT_CAL_BL
    run_step "estimate (Baseline)" "corebehrt.main_causal.estimate" "estimate" "$TARGET_DIR/estimate/baseline/estimate_results.csv" $TIMEOUT_EST_BL
fi

# 5. Run BERT Pipeline (if enabled)
if [ "$RUN_BERT" = "true" ]; then
    echo ""
    echo "========================================"
    echo "==== Running BERT Pipeline ===="
    echo "========================================"
    TARGET_DIR="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
    run_step "finetune (BERT)" "corebehrt.main_causal.finetune_exp_y" "finetune_bert" "$TARGET_DIR/models/bert/combined_predictions.csv" $TIMEOUT_FINETUNE_BERT
    run_step "calibrate (BERT)" "corebehrt.main_causal.calibrate_exp_y" "calibrate_bert" "$TARGET_DIR/models/bert/calibrated/combined_calibrated_predictions.csv" $TIMEOUT_CAL_BERT
    run_step "estimate (BERT)" "corebehrt.main_causal.estimate" "estimate_bert" "$TARGET_DIR/estimate/bert/estimate_results.csv" $TIMEOUT_EST_BERT
fi

echo ""
echo "========================================"
echo "Experiment $RUN_ID/$EXPERIMENT_NAME completed successfully!"
echo "========================================"
exit 0