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

echo "======================================"
echo "==== Running Preprocessing and Pretraining... ===="
echo "==== Deleting old features... ===="
rm -rf outputs/causal/data/features

echo "==== Running create_data... ===="
python -m corebehrt.main.create_data --config_path corebehrt/configs/causal/prepare_and_pretrain/create_data.yaml
check_error

echo "==== Running prepare_training_data... ===="
python -m corebehrt.main.prepare_training_data --config_path corebehrt/configs/causal/prepare_and_pretrain/prepare_pretrain.yaml
check_error

echo "==== Running pretrain... ===="
python -m corebehrt.main.pretrain --config_path corebehrt/configs/causal/prepare_and_pretrain/pretrain.yaml
check_error

# Run Outcomes and Cohort Selection
echo "==== Running create_outcomes... ===="
python -m corebehrt.main.create_outcomes --config_path corebehrt/configs/causal/outcomes.yaml
check_error

echo "==== Running select_cohort... ===="
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt/configs/causal/select_cohort_full/extract.yaml
check_error

echo "======================================"
echo "==== Running prepare_finetune_data... ===="
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt/configs/causal/finetune/prepare/simple.yaml
check_error

echo "==== Testing prepare_data_ft_exp_y... ===="
python -m tests.pipeline.prepare_data_ft_exp_y ./outputs/causal/finetune/prepared_data
check_error

echo "==== Running finetune... ===="
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt/configs/causal/finetune/simple.yaml
check_error

echo "==== Testing ft_exp_y... ===="
python -m tests.pipeline.ft_exp_y ./outputs/causal/finetune/models/simple
check_error

echo "==== Checking performance... ===="
python -m tests.pipeline.test_performance_multitarget ./outputs/causal/finetune/models/simple --target-bounds "exposure:min:0.58,max:0.8" --target-bounds "OUTCOME:min:0.7,max:0.95" --target-bounds "OUTCOME_2:min:0.7,max:0.95" --target-bounds "OUTCOME_3:min:0.58,max:0.95"
check_error

# Run Causal Steps
echo "==== Running calibrate... ===="
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt/configs/causal/finetune/calibrate.yaml
check_error

# Run Estimation
echo "==== Running estimate... ===="
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate.yaml
check_error

echo "==== Running extract_criteria... ===="
python -m corebehrt.main_causal.helper_scripts.extract_criteria --config_path corebehrt/configs/causal/helper/extract_criteria.yaml
check_error

echo "==== Running get_stats... ===="
python -m corebehrt.main_causal.helper_scripts.get_stats --config_path corebehrt/configs/causal/helper/get_stats.yaml
check_error

echo "Pipeline completed successfully."