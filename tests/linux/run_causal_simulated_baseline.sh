#!/bin/bash

# -------------------------------
# Run the pipeline with inline error checking
# -------------------------------

# Function to check exit status and exit if error
check_error() {
    if [ $? -ne 0 ]; then
        echo ""
        echo "======================================"
        echo "An error occurred in one of the modules."
        echo "Check the output above for the Python traceback."
        echo "Terminating pipeline."
        echo "======================================"
        exit 1
    fi
}

# Run the pipeline with inline error checking
# Run Preprocessing and Pretraining

# Run Outcomes and Cohort Selection
echo "==== Running simulate_outcomes... ===="
python -m corebehrt.main_causal.simulate_from_sequence --config_path corebehrt/configs/causal/simulate_realistic.yaml
check_error

echo "==== Running select_cohort... ===="
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt/configs/causal/select_cohort_full/extract_simulated.yaml
check_error

echo "======================================"
echo "==== Running prepare_finetune_data... ===="
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt/configs/causal/finetune/prepare/simulated.yaml
check_error

echo "==== Running finetune... ===="
python -m corebehrt.main_causal.train_baseline --config_path corebehrt/configs/causal/finetune/simulated_bl.yaml
check_error

# Run Causal Steps
echo "==== Running calibrate... ===="
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt/configs/causal/finetune/calibrate_simulated_bl.yaml
check_error

# Run Estimation
echo "==== Running estimate... ===="
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate_simulated_bl.yaml
check_error

echo "==== Running test_estimate_result... ===="
python -m tests.test_main_causal.test_estimate_result --ci_stretch_factor 1.4 --ipw_ci_stretch_factor 1.8 --dir outputs/causal/estimate/baseline/simulated_bl
check_error

echo "Pipeline completed successfully."
