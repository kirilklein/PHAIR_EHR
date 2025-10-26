#!/bin/bash

# ========================================
# Causal Pipeline Full Experiment Runner (Baseline + BERT)
# Refactored for clarity and maintainability
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
    echo "  -h, --help           Show this help message"
    echo "  --baseline-only      Run only baseline (CatBoost) pipeline"
    echo "  --bert-only          Run only BERT pipeline (requires baseline data)"
    echo "  --reuse-data|-r      Reuse prepared data from run_01 if available (default: true)"
    echo "  --no-reuse-data      Force regenerate all data even if run_01 exists"
    echo "  --dont-overwrite     Skip steps if output already exists (useful for resuming failed runs)"
    echo "  --experiment-dir|-e  Base directory for experiments (default: ./outputs/causal/sim_study/runs)"
    echo "  (no options)         Run both baseline and BERT pipelines"
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
    echo "  - Experiment configs are read from: ../experiment_configs/<experiment_name>.yaml"
    echo "  - Generated configs are saved to: ../generated_configs/<experiment_name>/"
    echo "  - Results are saved to: ../../../outputs/causal/sim_study/runs/<experiment_name>/"
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
REUSE_DATA=true
OVERWRITE=true
EXPERIMENTS_DIR="./outputs/causal/sim_study/runs"


while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-only) RUN_BERT=false; shift ;;
        --bert-only) RUN_BASELINE=false; shift ;;
        --run_id) RUN_ID="$2"; shift 2 ;;
        --reuse-data|-r) REUSE_DATA=true; shift ;;
        --no-reuse-data) REUSE_DATA=false; shift ;;
        --dont-overwrite) OVERWRITE=false; shift ;;
        --experiment-dir|-e) EXPERIMENTS_DIR="$2"; shift 2 ;;
        --timeout-factor) TIMEOUT_FACTOR="$2"; shift 2;;
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
        --pretrain-model)
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

    # Scale timeout by the factor
    local effective_timeout=$(echo "$timeout_secs * $TIMEOUT_FACTOR" | bc)
    
    echo "" # Add spacing
    if [ "$OVERWRITE" = "false" ] && [ -f "$check_file" ]; then
        echo "==== Skipping $step_name (output already exists) ===="
        return
    fi

    echo "==== Running $step_name... ===="
    local config_path="experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/$config_name.yaml"
    
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
python ../python_scripts/generate_configs.py "$EXPERIMENT_NAME" --run_id "$RUN_ID" --experiments_dir "$EXPERIMENTS_DIR" --meds "$MEDS_DATA" --features "$FEATURES_DATA" --tokenized "$TOKENIZED_DATA" --pretrain-model "$PRETRAIN_MODEL"
if [ $? -ne 0 ]; then echo "ERROR: Config generation failed."; exit 1; fi
echo "All configs generated successfully."
echo ""

# 2. Change to Project Root
cd ../../..
export PYTHONPATH="$PWD:$PYTHONPATH"

# 3. Handle Data Reuse
SHOULD_REUSE=false
if [ "$REUSE_DATA" = "true" ] && [ "$RUN_ID" != "run_01" ]; then
    RUN_01_PREPARED="$EXPERIMENTS_DIR/run_01/$EXPERIMENT_NAME/prepared_data"
    if [ -d "$RUN_01_PREPARED" ]; then
        SHOULD_REUSE=true
        echo "Reusing and copying prepared data from run_01..."
        TARGET_BASE="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
        mkdir -p "$TARGET_BASE"
        cp -r "$EXPERIMENTS_DIR/run_01/$EXPERIMENT_NAME/simulated_outcomes" "$TARGET_BASE/"
        cp -r "$EXPERIMENTS_DIR/run_01/$EXPERIMENT_NAME/cohort" "$TARGET_BASE/"
        cp -r "$EXPERIMENTS_DIR/run_01/$EXPERIMENT_NAME/prepared_data" "$TARGET_BASE/"
        echo "Data reuse complete."
    else
        echo "Note: --reuse-data enabled but run_01 data not found. Will generate new data for this run."
    fi
fi

# 4. Run Data Preparation Steps (if not reusing)
if [ "$SHOULD_REUSE" = "false" ]; then
    echo "Step 2: Running Data Preparation Pipeline..."
    TARGET_DIR="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
    run_step "simulate_outcomes" "corebehrt.main_causal.simulate_from_sequence" "simulation" "$TARGET_DIR/simulated_outcomes/counterfactuals.csv" $TIMEOUT_SIMULATE
    run_step "select_cohort" "corebehrt.main_causal.select_cohort_full" "select_cohort" "$TARGET_DIR/cohort/pids.pt" $TIMEOUT_COHORT
    run_step "prepare_finetune_data" "corebehrt.main_causal.prepare_ft_exp_y" "prepare_finetune" "$TARGET_DIR/prepared_data/patients.pt" $TIMEOUT_PREPARE
fi

# 5. Run Baseline Pipeline (if enabled)
if [ "$RUN_BASELINE" = "true" ]; then
    echo ""
    echo "========================================"
    echo "==== Running Baseline Pipeline ===="
    echo "========================================"
    TARGET_DIR="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
    run_step "train_baseline" "corebehrt.main_causal.train_baseline" "train_baseline" "$TARGET_DIR/models/baseline/combined_predictions.csv" $TIMEOUT_TRAIN_BL
    run_step "calibrate (Baseline)" "corebehrt.main_causal.calibrate_exp_y" "calibrate" "$TARGET_DIR/calibrated/baseline/combined_calibrated_predictions.csv" $TIMEOUT_CAL_BL
    run_step "estimate (Baseline)" "corebehrt.main_causal.estimate" "estimate" "$TARGET_DIR/estimate/baseline/estimate_results.csv" $TIMEOUT_EST_BL
fi

# 6. Run BERT Pipeline (if enabled)
if [ "$RUN_BERT" = "true" ]; then
    echo ""
    echo "========================================"
    echo "==== Running BERT Pipeline ===="
    echo "========================================"
    TARGET_DIR="$EXPERIMENTS_DIR/$RUN_ID/$EXPERIMENT_NAME"
    run_step "finetune (BERT)" "corebehrt.main_causal.finetune_exp_y" "finetune_bert" "$TARGET_DIR/models/bert/combined_predictions.csv" $TIMEOUT_FINETUNE_BERT
    run_step "calibrate (BERT)" "corebehrt.main_causal.calibrate_exp_y" "calibrate_bert" "$TARGET_DIR/calibrated/bert/combined_calibrated_predictions.csv" $TIMEOUT_CAL_BERT
    run_step "estimate (BERT)" "corebehrt.main_causal.estimate" "estimate_bert" "$TARGET_DIR/estimate/bert/estimate_results.csv" $TIMEOUT_EST_BERT
fi

echo ""
echo "========================================"
echo "Experiment $RUN_ID/$EXPERIMENT_NAME completed successfully!"
echo "========================================"
exit 0