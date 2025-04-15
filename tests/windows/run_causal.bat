@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
echo Running create_data...
python -m corebehrt.main.create_data --config_path corebehrt\configs\causal\prepare_and_pretrain\create_data.yaml
if errorlevel 1 goto :error

echo Running prepare_training_data...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\prepare_and_pretrain\prepare_pretrain.yaml
if errorlevel 1 goto :error

echo Running pretrain...
python -m corebehrt.main.pretrain --config_path corebehrt\configs\causal\prepare_and_pretrain\pretrain.yaml
if errorlevel 1 goto :error

:: Run Outcomes and Cohort Selection
echo Running create_outcomes...
python -m corebehrt.main.create_outcomes --config_path corebehrt\configs\causal\outcomes.yaml
if errorlevel 1 goto :error

echo Running select_cohort...
python -m corebehrt.main.select_cohort --config_path corebehrt\configs\causal\select_cohort\select_cohort_exposure.yaml
if errorlevel 1 goto :error

echo Running select_cohort_advanced...
python -m corebehrt.main_causal.select_cohort_advanced --config_path corebehrt\configs\causal\select_cohort\advanced_extraction.yaml
if errorlevel 1 goto :error

:: Run Finetuning
echo Running prepare_finetune_data...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\finetune\prepare_finetune_exposure.yaml
if errorlevel 1 goto :error

echo Running finetune...
python -m corebehrt.main.finetune_cv --config_path corebehrt\configs\causal\finetune\finetune_exposure.yaml
if errorlevel 1 goto :error

:: Run Causal Steps
echo Running calibrate...
python -m corebehrt.main_causal.calibrate --config_path corebehrt\configs\causal\finetune\calibrate.yaml
if errorlevel 1 goto :error

echo Running encode...
python -m corebehrt.main_causal.encode --config_path corebehrt\configs\causal\finetune\encode.yaml
if errorlevel 1 goto :error

echo Running train_mlp...
python -m corebehrt.main_causal.train_mlp --config_path corebehrt\configs\causal\double_robust\train_mlp.yaml
if errorlevel 1 goto :error

echo Running train_xgb...
python -m corebehrt.main_causal.train_xgb --config_path corebehrt\configs\causal\double_robust\train_xgb.yaml
if errorlevel 1 goto :error

:: Run Estimation
echo Running estimate...
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate\estimate.yaml
if errorlevel 1 goto :error

:: Run estimation with true outcome
echo Running simulate...
python -m corebehrt.main_causal.simulate --config_path corebehrt\configs\causal\simulate.yaml
if errorlevel 1 goto :error

echo Running train_mlp_simulated...
python -m corebehrt.main_causal.train_mlp --config_path corebehrt\configs\causal\double_robust\train_mlp_simulated.yaml
if errorlevel 1 goto :error

echo Running estimate_with_true...
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate\estimate_with_true.yaml
if errorlevel 1 goto :error

echo Pipeline completed successfully.
pause
exit /b 0

:error
    echo.
    echo ======================================
    echo An error occurred in one of the modules.
    echo Check the output above for the Python traceback.
    echo Terminating pipeline.
    echo ======================================
pause
exit /b 1