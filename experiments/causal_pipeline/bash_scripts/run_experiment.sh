#!/bin/bash

# ========================================
# Causal Pipeline Full Experiment Runner (Baseline + BERT)
# ========================================

# Check for help or no arguments
if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
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
    if [ "$BATCH_MODE" != "true" ]; then
        read -p "Press Enter to continue..."
    fi
    exit 1
fi

echo "DEBUG: run_experiment.sh called with arguments: $*"
echo "DEBUG: First argument \$1 = \"$1\""
echo "DEBUG: Second argument \$2 = \"$2\""
echo "DEBUG: All arguments \$* = $*"

EXPERIMENT_NAME="$1"
echo "DEBUG: EXPERIMENT_NAME set to: $EXPERIMENT_NAME"
RUN_BASELINE=true
RUN_BERT=true
RUN_ID="run_01"  # Default run ID
echo "DEBUG: Initial flags - RUN_BASELINE=$RUN_BASELINE, RUN_BERT=$RUN_BERT, RUN_ID=$RUN_ID"

# Parse additional arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-only)
            RUN_BERT=false
            shift
            ;;
        --bert-only)
            RUN_BASELINE=false
            shift
            ;;
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <experiment_name> [--baseline-only|--bert-only] [--run_id run_XX]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Running Full Causal Pipeline Experiment: $RUN_ID/$EXPERIMENT_NAME"
echo "DEBUG: RUN_BASELINE=$RUN_BASELINE, RUN_BERT=$RUN_BERT, RUN_ID=$RUN_ID"
if [ "$RUN_BASELINE" = "true" ] && [ "$RUN_BERT" = "true" ]; then
    echo "Mode: BASELINE + BERT"
elif [ "$RUN_BASELINE" = "true" ]; then
    echo "Mode: BASELINE ONLY"
else
    echo "Mode: BERT ONLY"
fi
echo "========================================"

# Function to check for errors
check_error() {
    if [ $? -ne 0 ]; then
        echo ""
        echo "======================================"
        echo "An error occurred in the $EXPERIMENT_NAME experiment."
        echo "Check the output above for the Python traceback."
        echo "Terminating pipeline."
        echo "======================================"
        if [ "$BATCH_MODE" != "true" ]; then
            read -p "Press Enter to continue..."
        fi
        exit 1
    fi
}

# Generate experiment-specific configs
echo "Step 1: Generating experiment configs..."
python ../python_scripts/generate_configs.py "$EXPERIMENT_NAME" --run_id "$RUN_ID"
check_error

echo ""
echo "Step 2: Running pipeline steps..."
echo ""

# Change to project root for running pipeline commands
cd ../../..
echo "DEBUG: Current directory after cd: $(pwd)"
echo "DEBUG: Config path will be: experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/simulation.yaml"

# Verify the config file exists from this directory
if [ -f "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/simulation.yaml" ]; then
    echo "DEBUG: Config file exists at expected path from current directory"
else
    echo "ERROR: Config file NOT found at expected path from current directory"
    echo "DEBUG: Let's see what's in the generated_configs directory:"
    ls -la "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/" 2>/dev/null || echo "Directory does not exist"
    if [ "$BATCH_MODE" != "true" ]; then
        read -p "Press Enter to continue..."
    fi
    exit 1
fi

# Add the project root to Python path (if needed)
export PYTHONPATH="$PWD:$PYTHONPATH"

# --- Shared Data Preparation ---
echo "==== Running simulate_outcomes... ===="
python -m corebehrt.main_causal.simulate_from_sequence --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/simulation.yaml
check_error

echo "==== Running select_cohort... ===="
python -m corebehrt.main_causal.select_cohort_full --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/select_cohort.yaml
check_error

echo "==== Running prepare_finetune_data... ===="
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/prepare_finetune.yaml
check_error

# --- Baseline Pipeline ---
if [ "$RUN_BASELINE" = "true" ]; then
    echo ""
    echo "========================================"
    echo "==== Running Baseline Pipeline ===="
    echo "========================================"
    echo "==== Running train_baseline... ===="
    python -m corebehrt.main_causal.train_baseline --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/train_baseline.yaml
    check_error

    echo "==== Running calibrate (Baseline)... ===="
    python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/calibrate.yaml
    check_error

    echo "==== Running estimate (Baseline)... ===="
    python -m corebehrt.main_causal.estimate --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/estimate.yaml
    check_error
fi

# --- BERT Pipeline ---
if [ "$RUN_BERT" = "true" ]; then
    echo ""
    echo "========================================"
    echo "==== Running BERT Pipeline ===="
    echo "========================================"
    echo "==== Running finetune (BERT)... ===="
    python -m corebehrt.main_causal.finetune_exp_y --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/finetune_bert.yaml
    check_error

    echo "==== Running calibrate (BERT)... ===="
    python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/calibrate_bert.yaml
    check_error

    echo "==== Running estimate (BERT)... ===="
    python -m corebehrt.main_causal.estimate --config_path experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/estimate_bert.yaml
    check_error
fi

echo ""
echo "========================================"
echo "Experiment $RUN_ID/$EXPERIMENT_NAME completed successfully!"
echo "Results saved in: outputs/causal/sim_study/runs/$RUN_ID/$EXPERIMENT_NAME/"
echo "========================================"

if [ "$BATCH_MODE" != "true" ]; then
    read -p "Press Enter to continue..."
fi
exit 0
