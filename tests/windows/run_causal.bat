@echo off
setlocal EnableDelayedExpansion

REM Create data
python -m corebehrt.main.create_data --config_path ./corebehrt/configs/causal/prepare_and_pretrain/create_data.yaml
if !errorlevel! neq 0 (
    echo Error creating data
    pause
    exit /b !errorlevel!
)

REM Prepare pretraining data
python -m corebehrt.main.prepare_training_data --config_path ./corebehrt/configs/causal/prepare_and_pretrain/prepare_pretrain.yaml
if !errorlevel! neq 0 (
    echo Error preparing training data
    pause
    exit /b !errorlevel!
)

REM Pretrain
python -m corebehrt.main.pretrain --config_path ./corebehrt/configs/causal/prepare_and_pretrain/pretrain.yaml
if !errorlevel! neq 0 (
    echo Error during pretraining
    pause
    exit /b !errorlevel!
)

REM Create outcomes 
python -m corebehrt.main.create_outcomes --config_path ./corebehrt/configs/causal/outcomes.yaml
if !errorlevel! neq 0 (
    echo Error creating outcomes
    pause
    exit /b !errorlevel!
)

REM Select cohort exposure
python -m corebehrt.main.select_cohort --config_path ./corebehrt/configs/causal/select_cohort/select_cohort_exposure.yaml
if !errorlevel! neq 0 (
    echo Error selecting cohort exposure
    pause
    exit /b !errorlevel!
)

REM Prepare finetune exposure
python -m corebehrt.main.prepare_training_data --config_path ./corebehrt/configs/causal/finetune/prepare_finetune_exposure.yaml
if !errorlevel! neq 0 (
    echo Error preparing finetune exposure
    pause
    exit /b !errorlevel!
)

REM Finetune exposure
python -m corebehrt.main.finetune_cv --config_path ./corebehrt/configs/causal/finetune/finetune_exposure.yaml
if !errorlevel! neq 0 (
    echo Error during finetuning exposure
    pause
    exit /b !errorlevel!
)

REM Causal related steps

REM Calibrate
python -m corebehrt.main_causal.calibrate --config_path ./corebehrt/configs/causal/finetune/calibrate.yaml
if !errorlevel! neq 0 (
    echo Error during calibration
    pause
    exit /b !errorlevel!
)

REM Encode
python -m corebehrt.main_causal.encode --config_path ./corebehrt/configs/causal/finetune/encode.yaml
if !errorlevel! neq 0 (
    echo Error during encoding
    pause
    exit /b !errorlevel!
)

REM Train MLP
python -m corebehrt.main_causal.train_mlp
if !errorlevel! neq 0 (
    echo Error training MLP
    pause
    exit /b !errorlevel!
)

REM Train XGB (alternative to MLP)
python -m corebehrt.main_causal.train_xgb --config_path ./corebehrt/configs/causal/double_robust/train_xgb.yaml
if !errorlevel! neq 0 (
    echo Error training XGB
    pause
    exit /b !errorlevel!
)

REM Estimate
python -m corebehrt.main_causal.estimate
if !errorlevel! neq 0 (
    echo Error during estimation
    pause
    exit /b !errorlevel!
)

REM With counterfactual outcomes

REM Simulate
python -m corebehrt.main_causal.simulate
if !errorlevel! neq 0 (
    echo Error during simulation
    pause
    exit /b !errorlevel!
)

REM Train MLP with simulated outcomes
python -m corebehrt.main_causal.train_mlp --config_path ./corebehrt/configs/causal/double_robust/train_mlp_simulated.yaml
if !errorlevel! neq 0 (
    echo Error training MLP with simulated outcomes
    pause
    exit /b !errorlevel!
)

REM Estimate with true outcomes
python -m corebehrt.main_causal.estimate --config_path ./corebehrt/configs/causal/estimate/estimate_with_true.yaml
if !errorlevel! neq 0 (
    echo Error during estimation with true outcomes
    pause
    exit /b !errorlevel!
)

REM Pause to allow you to see the output before the window closes
pause
