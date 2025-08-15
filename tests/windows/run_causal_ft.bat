@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining

echo ==== Running finetune... ====
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt\configs\causal\finetune\simple.yaml
if errorlevel 1 goto :error

echo ==== Testing ft_exp_y... ====
python tests\pipeline\ft_exp_y.py .\outputs\causal\finetune\models\simple
if errorlevel 1 goto :error

echo ==== Checking performance... ====
python -m tests.pipeline.test_performance_multitarget .\outputs\causal\finetune\models\simple --target-bounds "exposure:min:0.58,max:0.8" --target-bounds "OUTCOME:min:0.7,max:0.95" --target-bounds "OUTCOME_2:min:0.7,max:0.95" --target-bounds "OUTCOME_3:min:0.58,max:0.95"
if errorlevel 1 goto :error

:: Run Causal Steps
echo ==== Running calibrate... ====
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt\configs\causal\finetune\calibrate.yaml
if errorlevel 1 goto :error

@REM echo ==== Checking cf magnitude... ====
@REM python -m tests.pipeline.test_cf_magnitude ./outputs/causal/finetune/models/simple/calibrated example_data/synthea_meds_causal/tuning --top_n_percent 10 --ate_tolerance 0.2
@REM if errorlevel 1 goto :error

:: Run Estimation
echo ==== Running estimate... ====
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate.yaml
if errorlevel 1 goto :error

@REM echo ==== Checking estimate... ====
@REM python -m tests.pipeline.test_estimate ./outputs/causal/estimate/simple example_data/synthea_meds_causal/tuning
@REM if errorlevel 1 goto :error

echo ==== Running extract_criteria... ====
python -m corebehrt.main_causal.helper_scripts.extract_criteria --config_path corebehrt\configs\causal\helper\extract_criteria.yaml
if errorlevel 1 goto :error

echo ==== Running get_stats... ====
python -m corebehrt.main_causal.helper_scripts.get_stats --config_path corebehrt\configs\causal\helper\get_stats.yaml
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