#!/bin/bash

echo "\n ================================ Prepare and pretrain ================================ "
echo "Running create_data"
python -m corebehrt.main.create_data --config_path corebehrt/configs/causal/prepare_and_pretrain/create_data.yaml

echo "Running prepare_training_data for pretrain"
python -m corebehrt.main.prepare_training_data --config_path corebehrt/configs/causal/prepare_and_pretrain/prepare_pretrain.yaml

echo "Running pretrain"
python -m corebehrt.main.pretrain --config_path corebehrt/configs/causal/prepare_and_pretrain/pretrain.yaml



echo "\n ================================ Outcomes ================================ "
echo "Running create_outcomes"
python -m corebehrt.main.create_outcomes --config_path corebehrt/configs/causal/outcomes.yaml

echo "Running select_cohort"
python -m corebehrt.main.select_cohort --config_path corebehrt/configs/causal/select_cohort/select_cohort_exposure.yaml

echo "Running select_cohort_advanced"
python -m corebehrt.main_causal.select_cohort_advanced --config_path corebehrt/configs/causal/select_cohort/advanced_extraction.yaml

echo "Running prepare_finetune_data"
python -m corebehrt.main.prepare_training_data --config_path corebehrt/configs/causal/finetune/prepare_finetune_exposure.yaml

echo "Running finetune"
python -m corebehrt.main.finetune_cv --config_path corebehrt/configs/causal/finetune/finetune_exposure.yaml



echo "\n ================================ Causal pipeline ================================ "
echo "Running calibrate"
python -m corebehrt.main_causal.calibrate --config_path corebehrt/configs/causal/finetune/calibrate.yaml

echo "Running encode"
python -m corebehrt.main_causal.encode --config_path corebehrt/configs/causal/finetune/encode.yaml

echo "Running train_mlp"
python -m corebehrt.main_causal.train_mlp --config_path corebehrt/configs/causal/double_robust/train_mlp.yaml

echo "Running train_xgb"
python -m corebehrt.main_causal.train_xgb --config_path corebehrt/configs/causal/double_robust/train_xgb.yaml

echo "Running estimate"
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate/estimate.yaml

echo "Running simulate"
python -m corebehrt.main_causal.simulate --config_path corebehrt/configs/causal/simulate.yaml

echo "Running train_mlp_simulated"
python -m corebehrt.main_causal.train_mlp --config_path corebehrt/configs/causal/double_robust/train_mlp_simulated.yaml

echo "Running estimate with true data"
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate/estimate_with_true.yaml

echo "Checking estimation"
python -m tests.test_main_causal.test_estimate_result --margin 0.1 --dir ./outputs/causal/estimate_with_true



echo "\n ================================ Simulated data with weak treatment effect ================================ "
echo "Running simulate_weak"
python -m corebehrt.main_causal.simulate --config_path corebehrt/configs/causal/simulate_weak.yaml

echo "Running train_mlp_simulated_weak"
python -m corebehrt.main_causal.train_mlp --config_path corebehrt/configs/causal/double_robust/train_mlp_simulated_weak.yaml

echo "Running estimate with true weak data"
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate/estimate_with_true_weak.yaml

echo "Checking estimation weak"
python -m tests.test_main_causal.test_estimate_result --margin 0.1 --dir ./outputs/causal/estimate_with_true_weak



echo "\n ================================ Simulated data with more contribution from encodings ================================ "
echo "Running simulate_enc_strong"
python -m corebehrt.main_causal.simulate --config_path corebehrt/configs/causal/simulate_enc_strong.yaml

echo "Running train_mlp_simulated_enc_strong"
python -m corebehrt.main_causal.train_mlp --config_path corebehrt/configs/causal/double_robust/train_mlp_simulated_enc_strong.yaml

echo "Running estimate with true enc_strong"
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate/estimate_with_true_enc_strong.yaml

echo "Checking estimation enc_strong"
python -m tests.test_main_causal.test_estimate_result --margin 0.1 --dir ./outputs/causal/estimate_with_true_enc_strong



echo "\n ================================ Simulated data and predictions with xgboost ================================ "
echo "Running train_xgb_simulated (using xgboost)"
python -m corebehrt.main_causal.train_xgb --config_path corebehrt/configs/causal/double_robust/train_xgb_simulated.yaml

echo "Running estimate with xgboost"
python -m corebehrt.main_causal.estimate --config_path corebehrt/configs/causal/estimate/estimate_with_true_xgb.yaml

echo "Checking estimation xgboost"
python -m tests.test_main_causal.test_estimate_result --margin 0.1 --dir ./outputs/causal/estimate_with_true_xgb