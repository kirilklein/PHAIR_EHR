@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Causal Pipeline Experiment Runner
REM ========================================

if "%1"=="" (
    echo Usage: run_experiment.bat ^<experiment_name^>
    echo.
    echo Available experiments:
    for %%f in (..\experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    echo.
    echo Example: run_experiment.bat no_influence
    pause
    exit /b 1
)

set EXPERIMENT_NAME=%1
set SCRIPT_DIR=%~dp0
set CONFIG_DIR=%SCRIPT_DIR%generated_configs\%EXPERIMENT_NAME%

echo ========================================
echo Running Causal Pipeline Experiment: %EXPERIMENT_NAME%
echo ========================================

REM Check if experiment config exists
if not exist "..\experiment_configs\%EXPERIMENT_NAME%.yaml" (
    echo ERROR: Experiment config not found: ..\experiment_configs\%EXPERIMENT_NAME%.yaml
    echo.
    echo Available experiments:
    for %%f in (..\experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    pause
    exit /b 1
)

REM Generate experiment-specific configs
echo Step 1: Generating experiment configs...
python ..\python_scripts\generate_configs.py %EXPERIMENT_NAME%
if errorlevel 1 (
    echo ERROR: Failed to generate configs
    pause
    exit /b 1
)

echo.
echo Step 2: Running pipeline steps...
echo.

REM Change to project root for running pipeline commands
cd ..\..

REM Run pipeline steps with generated configs
echo ==== Running simulate_outcomes... ====
python -m corebehrt.main_causal.simulate_from_sequence --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\simulation.yaml
if errorlevel 1 goto :error

echo ==== Running select_cohort... ====
python -m corebehrt.main_causal.select_cohort_full --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\select_cohort.yaml
if errorlevel 1 goto :error

echo ==== Running prepare_finetune_data... ====
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\prepare_finetune.yaml
if errorlevel 1 goto :error

echo ==== Running train_baseline... ====
python -m corebehrt.main_causal.train_baseline --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\train_baseline.yaml
if errorlevel 1 goto :error

echo ==== Running calibrate... ====
python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\calibrate.yaml
if errorlevel 1 goto :error

echo ==== Running estimate... ====
python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\estimate.yaml
if errorlevel 1 goto :error

echo ==== Running test_estimate_result... ====
python -m tests.test_main_causal.test_estimate_result --ci_stretch_factor 2.0 --ipw_ci_stretch_factor 2.0 --dir outputs\causal\sim_study\runs\%EXPERIMENT_NAME%\estimate
if errorlevel 1 goto :error

echo.
echo ========================================
echo Experiment %EXPERIMENT_NAME% completed successfully!
echo Results saved in: outputs\causal\sim_study\runs\%EXPERIMENT_NAME%\
echo ========================================

REM Only pause if not running in batch mode
if not "%BATCH_MODE%"=="true" pause
exit /b 0

:error
    echo.
    echo ======================================
    echo An error occurred in the %EXPERIMENT_NAME% experiment.
    echo Check the output above for the Python traceback.
    echo Terminating pipeline.
    echo ======================================

REM Only pause if not running in batch mode
if not "%BATCH_MODE%"=="true" pause
exit /b 1

