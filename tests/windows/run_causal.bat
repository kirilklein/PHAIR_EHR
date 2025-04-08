@echo off

REM Create outcomes exposure
python -m corebehrt.main.create_outcomes --config_path ./corebehrt/configs/causal/outcomes.yaml

REM Select cohort exposure
python -m corebehrt.main.select_cohort --config_path ./corebehrt/configs/causal/select_cohort_exposure.yaml

REM Prepare finetune exposure
python -m corebehrt.main.prepare_training_data --config_path ./corebehrt/configs/causal/prepare_finetune_exposure.yaml

REM Finetune exposure
python -m corebehrt.main.finetune_cv --config_path ./corebehrt/configs/causal/finetune_exposure.yaml


REM Causal related steps

REM Calibrate
python -m corebehrt.main_causal.calibrate

REM Encode
python -m corebehrt.main_causal.encode

REM Train MLP
python -m corebehrt.main_causal.train_mlp

REM Train XGB (alternative to MLP)
python -m corebehrt.main_causal.train_xgb --config_path ./corebehrt/configs/causal/train_xgb.yaml

REM Estimate
python -m corebehrt.main_causal.estimate

REM With counterfactual outcomes

REM Simulate
python -m corebehrt.main_causal.simulate

REM Train MLP with simulated outcomes
python -m corebehrt.main_causal.train_mlp --config_path ./corebehrt/configs/causal/train_mlp_simulated.yaml

REM Estimate with true outcomes
python -m corebehrt.main_causal.estimate --config_path ./corebehrt/configs/causal/estimate/estimate_with_true.yaml

REM Pause to allow you to see the output before the window closes
pause
