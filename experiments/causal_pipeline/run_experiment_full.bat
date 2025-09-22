@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Full Causal Pipeline Experiment Runner (Baseline + BERT)
REM ========================================

if "%1"=="" (
    echo Usage: run_experiment_full.bat ^<experiment_name^> [--baseline-only^|--bert-only]
    echo.
    echo Options:
    echo   --baseline-only    Run only baseline ^(CatBoost^) pipeline
    echo   --bert-only        Run only BERT pipeline ^(requires baseline data^)
    echo   ^(no flag^)          Run both baseline and BERT pipelines
    echo.
    echo Available experiments:
    for %%f in (experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    echo.
    echo Example: run_experiment_full.bat ce1_cy1_y0_i0
    pause
    exit /b 1
)

set EXPERIMENT_NAME=%1
set RUN_BASELINE=true
set RUN_BERT=true

REM Parse additional arguments
:parse_args
shift
if "%1"=="" goto :start_experiment
if "%1"=="--baseline-only" (
    set RUN_BERT=false
    goto :parse_args
)
if "%1"=="--bert-only" (
    set RUN_BASELINE=false
    goto :parse_args
)
shift
goto :parse_args

:start_experiment
set SCRIPT_DIR=%~dp0
set CONFIG_DIR=%SCRIPT_DIR%generated_configs\%EXPERIMENT_NAME%

echo ========================================
echo Running Full Causal Pipeline Experiment: %EXPERIMENT_NAME%
if "%RUN_BASELINE%"=="true" if "%RUN_BERT%"=="true" (
    echo Mode: BASELINE + BERT
) else if "%RUN_BASELINE%"=="true" (
    echo Mode: BASELINE ONLY
) else (
    echo Mode: BERT ONLY
)
echo ========================================

REM Check if experiment config exists
if not exist "experiment_configs\%EXPERIMENT_NAME%.yaml" (
    echo ERROR: Experiment config not found: experiment_configs\%EXPERIMENT_NAME%.yaml
    echo.
    echo Available experiments:
    for %%f in (experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    pause
    exit /b 1
)

REM Generate experiment-specific configs
echo Step 1: Generating experiment configs...
python scripts\generate_configs.py %EXPERIMENT_NAME%
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

if "%RUN_BASELINE%"=="true" (
    echo ======================================
    echo PHASE 1: SHARED DATA PREPARATION
    echo ======================================
    
    echo ==== Running simulate_outcomes... ====
    python -m corebehrt.main_causal.simulate_from_sequence --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\simulation.yaml
    if errorlevel 1 goto :error

    echo ==== Running select_cohort... ====
    python -m corebehrt.main_causal.select_cohort_full --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\select_cohort.yaml
    if errorlevel 1 goto :error

    echo ==== Running prepare_finetune_data... ====
    python -m corebehrt.main_causal.prepare_ft_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\prepare_finetune.yaml
    if errorlevel 1 goto :error

    echo.
    echo ======================================
    echo PHASE 2: BASELINE MODELS ^(CATBOOST^)
    echo ======================================

    echo ==== Running train_baseline... ====
    python -m corebehrt.main_causal.train_baseline --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\train_baseline.yaml
    if errorlevel 1 goto :error

    echo ==== Running calibrate_baseline... ====
    python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\calibrate.yaml
    if errorlevel 1 goto :error

    echo ==== Running estimate_baseline... ====
    python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\estimate.yaml
    if errorlevel 1 goto :error
)

if "%RUN_BERT%"=="true" (
    echo.
    echo ======================================
    echo PHASE 3: BERT MODELS
    echo ======================================

    echo ==== Running finetune_bert... ====
    python -m corebehrt.main_causal.finetune_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\finetune_bert.yaml
    if errorlevel 1 goto :error

    echo ==== Running calibrate_bert... ====
    python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\calibrate_bert.yaml
    if errorlevel 1 goto :error

    echo ==== Running estimate_bert... ====
    python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\estimate_bert.yaml
    if errorlevel 1 goto :error
)

echo.
echo ========================================
echo Experiment %EXPERIMENT_NAME% completed successfully!
echo Results saved in: outputs\causal\experiments\%EXPERIMENT_NAME%\
if "%RUN_BASELINE%"=="true" (
    echo   - Baseline results: outputs\causal\experiments\%EXPERIMENT_NAME%\estimate\baseline\
)
if "%RUN_BERT%"=="true" (
    echo   - BERT results: outputs\causal\experiments\%EXPERIMENT_NAME%\estimate\bert\
)
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
