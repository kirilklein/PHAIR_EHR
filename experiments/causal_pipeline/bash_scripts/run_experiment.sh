#!/bin/bash

# ========================================
# Causal Pipeline Experiment Runner
# ========================================

if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiment.sh <experiment_name>"
    echo ""
    echo "Available experiments:"
    for file in experiment_configs/*.yaml; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .yaml)
            echo "  - $filename"
        fi
    done
    echo ""
    echo "Example: ./run_experiment.sh ce0_cy0_y0_i0"
    read -p "Press Enter to continue..."
    exit 1
fi

EXPERIMENT_NAME=$1
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
CONFIG_DIR="$SCRIPT_DIR/generated_configs/$EXPERIMENT_NAME"

echo "========================================"
echo "Running Causal Pipeline Experiment: $EXPERIMENT_NAME"
echo "========================================"

# Check if experiment config exists
if [ ! -f "experiment_configs/$EXPERIMENT_NAME.yaml" ]; then
    echo "ERROR: Experiment config not found: experiment_configs/$EXPERIMENT_NAME.yaml"
    echo ""
    echo "Available experiments:"
    for file in experiment_configs/*.yaml; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .yaml)
            echo "  - $filename"
        fi
    done
    read -p "Press Enter to continue..."
    exit 1
fi

# Generate experiment-specific configs
echo "Step 1: Generating experiment configs..."
python ../python_scripts/generate_configs.py "$EXPERIMENT_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate configs"
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "Step 2: Running pipeline steps..."
echo ""

# Change to project root for running pipeline commands
cd ../..

# Run pipeline steps with generated configs
echo "==== Running simulate_outcomes... ===="
python -m corebehrt.main_causal.simulate_from_sequence --config_path "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/simulation.yaml"
if [ $? -ne 0 ]; then
    echo ""
    echo "======================================"
    echo "An error occurred in the $EXPERIMENT_NAME experiment."
    echo "Check the output above for the Python traceback."
    echo "Terminating pipeline."
    echo "======================================"
    # Only pause if not running in batch mode
    if [ "$BATCH_MODE" != "true" ]; then
        read -p "Press Enter to continue..."
    fi
    exit 1
fi

echo "==== Running select_cohort... ===="
python -m corebehrt.main_causal.select_cohort_full --config_path "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/select_cohort.yaml"
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

echo "==== Running prepare_finetune_data... ===="
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/prepare_finetune.yaml"
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

echo "==== Running train_baseline... ===="
python -m corebehrt.main_causal.train_baseline --config_path "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/train_baseline.yaml"
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

echo "==== Running calibrate... ===="
python -m corebehrt.main_causal.calibrate_exp_y --config_path "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/calibrate.yaml"
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

echo "==== Running estimate... ===="
python -m corebehrt.main_causal.estimate --config_path "experiments/causal_pipeline/generated_configs/$EXPERIMENT_NAME/estimate.yaml"
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

echo "==== Running test_estimate_result... ===="
python -m tests.test_main_causal.test_estimate_result --ci_stretch_factor 2.0 --ipw_ci_stretch_factor 2.0 --dir "outputs/causal/sim_study/runs/$EXPERIMENT_NAME/estimate"
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

echo ""
echo "========================================"
echo "Experiment $EXPERIMENT_NAME completed successfully!"
echo "Results saved in: outputs/causal/sim_study/runs/$EXPERIMENT_NAME/"
echo "========================================"

# Only pause if not running in batch mode
if [ "$BATCH_MODE" != "true" ]; then
    read -p "Press Enter to continue..."
fi
exit 0
