#!/bin/bash

# ========================================
# Causal Pipeline Full Experiment Runner (Baseline + BERT)
# ========================================

# Check if experiment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [--baseline-only|--bert-only]"
    echo "Example: $0 my_experiment --baseline-only"
    exit 1
fi

EXPERIMENT_NAME="$1"
RUN_BASELINE=true
RUN_BERT=true

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <experiment_name> [--baseline-only|--bert-only]"
            exit 1
            ;;
    esac
done

echo "Running Full Causal Pipeline Experiment: $EXPERIMENT_NAME"
if [ "$RUN_BASELINE" = "true" ]; then
    echo "Including Baseline pipeline"
fi
if [ "$RUN_BERT" = "true" ]; then
    echo "Including BERT pipeline"
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
python ../python_scripts/generate_configs.py "$EXPERIMENT_NAME"
check_error

echo ""
echo "Step 2: Running pipeline steps..."
echo ""

# Change to project root for running pipeline commands
cd ../..

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
echo "Experiment $EXPERIMENT_NAME completed successfully!"
echo "Results saved in: outputs/causal/experiments/$EXPERIMENT_NAME/"
echo "========================================"

if [ "$BATCH_MODE" != "true" ]; then
    read -p "Press Enter to continue..."
fi
exit 0
