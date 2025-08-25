#!/bin/bash

# -------------------------------
# Run the pipeline with inline error checking
# -------------------------------

# Function to handle errors
handle_error() {
    echo ""
    echo "======================================"
    echo "An error occurred in one of the modules."
    echo "Check the output above for the Python traceback."
    echo "Terminating pipeline."
    echo "======================================"
    exit 1
}

# Set error handling - exit on any command failure
set -e
trap 'handle_error' ERR

# Run the pipeline with inline error checking
# Run Preprocessing and Pretraining
echo "======================================"
echo "==== Running Preprocessing and Pretraining... ===="
echo "==== Deleting old features... ===="
rm -rf outputs/causal/data/features

echo "==== Running create_data... ===="
python -m corebehrt.main.create_data --config_path corebehrt/configs/causal/prepare_and_pretrain/create_data.yaml

echo "==== Running prepare_training_data... ===="
python -m corebehrt.main.prepare_training_data --config_path corebehrt/configs/causal/prepare_and_pretrain/prepare_pretrain.yaml

echo "==== Running pretrain... ===="
python -m corebehrt.main.pretrain --config_path corebehrt/configs/causal/prepare_and_pretrain/pretrain.yaml

# Run Outcomes and Cohort Selection
echo "==== Running simulate_outcomes... ===="
python -m corebehrt.main_causal.simulate_from_sequence --config_path corebehrt/configs/causal/simulate_realistic.yaml

echo "==== Running select_cohort... ===="
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt/configs/causal/select_cohort_full/extract_simulated.yaml

echo "======================================"
echo "==== Running prepare_finetune_data... ===="
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt/configs/causal/finetune/prepare/simulated.yaml

echo "==== Running finetune... ===="
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt/configs/causal/finetune/simulated.yaml

# Run Causal Steps
echo "==== Running calibrate... ===="
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt/configs/causal/finetune/calibrate_simulated.yaml

# Run Estimation
echo "==== Running estimate... ===="
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate_simulated.yaml

echo "==== Running estimate with bias... ===="
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate_simulated_with_bias.yaml

echo "==== Running test_estimate_result... ===="
python -m tests.test_main_causal.test_estimate_result --ci_stretch_factor 1.4 --ipw_ci_stretch_factor 1.8 --dir outputs/causal/estimate/simulated

echo "Pipeline completed successfully."
