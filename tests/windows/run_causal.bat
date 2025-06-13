@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
echo Delete old features
rmdir /s /q outputs\causal\data\features

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
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract.yaml
if errorlevel 1 goto :error

:: Prepare and run finetuning
echo Running prepare_finetune_data...
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt\configs\causal\finetune\prepare\ft_exp_y.yaml
if errorlevel 1 goto :error

echo Running finetune...
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt\configs\causal\finetune\ft_exp_y.yaml
if errorlevel 1 goto :error


:: Run Causal Steps
echo Running calibrate...
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt\configs\causal\finetune\calibrate_exp_y.yaml
if errorlevel 1 goto :error

:: Run Estimation
echo Running estimate...
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate\estimate.yaml
if errorlevel 1 goto :error

echo Checking estimate...
python -m tests.pipeline.test_estimate ./outputs/causal/estimate/simple example_data/MEDS_correlated_causal/tuning

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