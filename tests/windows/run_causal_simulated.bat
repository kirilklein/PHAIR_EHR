@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
@REM echo ======================================
@REM echo ==== Running Preprocessing and Pretraining... ====
@REM echo ==== Deleting old features... ====
@REM rmdir /s /q outputs\causal\data\features

@REM echo ==== Running create_data... ====
@REM python -m corebehrt.main.create_data --config_path corebehrt\configs\causal\prepare_and_pretrain\create_data.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Running prepare_training_data... ====
@REM python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\prepare_and_pretrain\prepare_pretrain.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Running pretrain... ====
@REM python -m corebehrt.main.pretrain --config_path corebehrt\configs\causal\prepare_and_pretrain\pretrain.yaml
@REM if errorlevel 1 goto :error

:: Run Outcomes and Cohort Selection
echo ==== Running simulate_outcomes... ====
python -m corebehrt.main_causal.simulate_from_sequence --config_path corebehrt\configs\causal\simulate.yaml
if errorlevel 1 goto :error

echo ==== Running select_cohort... ====
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract_simulated.yaml
if errorlevel 1 goto :error

echo ======================================
echo ==== Running prepare_finetune_data... ====
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt\configs\causal\finetune\prepare\simulated.yaml
if errorlevel 1 goto :error

echo ==== Running finetune... ====
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt\configs\causal\finetune\simulated.yaml
if errorlevel 1 goto :error

:: Run Causal Steps
echo ==== Running calibrate... ====
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt\configs\causal\finetune\calibrate_simulated.yaml
if errorlevel 1 goto :error

:: Run Estimation
echo ==== Running estimate... ====
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate_simulated.yaml
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