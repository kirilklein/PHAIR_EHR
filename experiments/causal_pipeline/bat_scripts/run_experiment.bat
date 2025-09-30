@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Full Causal Pipeline Experiment Runner (Baseline + BERT)
REM ========================================

echo DEBUG: run_experiment.bat called with arguments: %*
echo DEBUG: First argument %%1 = "%1"
echo DEBUG: Second argument %%2 = "%2"
echo DEBUG: All arguments %%* = %*

if "%1"=="" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help

REM Arguments provided, proceed to main execution
goto :main_execution

:show_help
echo ========================================
echo Full Causal Pipeline Experiment Runner
echo ========================================
echo.
echo Usage: run_experiment.bat ^<experiment_name^> [OPTIONS]
echo.
echo ARGUMENTS:
echo   experiment_name       Name of the experiment to run
echo.
echo OPTIONS:
echo   -h, --help           Show this help message
echo   --baseline-only      Run only baseline ^(CatBoost^) pipeline
echo   --bert-only          Run only BERT pipeline ^(requires baseline data^)
echo   --reuse-data         Reuse prepared data from run_01 if available ^(default: true^)
echo   --no-reuse-data      Force regenerate all data even if run_01 exists
echo   ^(no options^)         Run both baseline and BERT pipelines
echo.
echo AVAILABLE EXPERIMENTS:
for %%f in (..\experiment_configs\*.yaml) do (
    set filename=%%~nf
    echo   - !filename!
)
echo.
echo EXAMPLES:
echo   run_experiment.bat ce1_cy1_y0_i0
echo     ^> Runs both baseline and BERT pipelines for experiment ce1_cy1_y0_i0
echo.
echo   run_experiment.bat ce1_cy1_y0_i0 --baseline-only
echo     ^> Runs only baseline pipeline for experiment ce1_cy1_y0_i0
echo.
echo   run_experiment.bat ce1_cy1_y0_i0 --bert-only
echo     ^> Runs only BERT pipeline for experiment ce1_cy1_y0_i0 ^(baseline data must exist^)
echo.
echo PIPELINE PHASES:
echo   PHASE 1: SHARED DATA PREPARATION
echo     - simulate_outcomes: Generate synthetic data
echo     - select_cohort: Select patient cohort
echo     - prepare_finetune_data: Prepare data for model training
echo.
echo   PHASE 2: BASELINE MODELS ^(CATBOOST^)
echo     - train_baseline: Train CatBoost models
echo     - calibrate_baseline: Calibrate baseline models
echo     - estimate_baseline: Run causal estimation with baseline models
echo.
echo   PHASE 3: BERT MODELS
echo     - finetune_bert: Fine-tune BERT models
echo     - calibrate_bert: Calibrate BERT models
echo     - estimate_bert: Run causal estimation with BERT models
echo.
echo NOTES:
echo   - Requires conda environment 'phair_ehr' to be available
echo   - Experiment configs are read from: ..\experiment_configs\^<experiment_name^>.yaml
echo   - Generated configs are saved to: generated_configs\^<experiment_name^>\
echo   - Results are saved to: ..\..\outputs\causal\sim_study\runs\^<experiment_name^>\
echo   - Use Ctrl+C to stop the experiment at any time
echo.
if not "%BATCH_MODE%"=="true" pause
exit /b 1

:main_execution

set EXPERIMENT_NAME=%1
echo DEBUG: EXPERIMENT_NAME set to: %EXPERIMENT_NAME%
set RUN_BASELINE=true
set RUN_BERT=true
set RUN_ID=run_01
set REUSE_DATA=true
echo DEBUG: Initial flags - RUN_BASELINE=%RUN_BASELINE%, RUN_BERT=%RUN_BERT%, RUN_ID=%RUN_ID%, REUSE_DATA=%REUSE_DATA%

REM Parse additional arguments
:parse_args
shift
if "%1"=="" goto :start_experiment
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="--baseline-only" (
    set RUN_BERT=false
    shift
    goto :parse_args
)
if "%1"=="--bert-only" (
    set RUN_BASELINE=false
    shift
    goto :parse_args
)
if "%1"=="--run_id" (
    if "%2"=="" (
        echo ERROR: --run_id requires a run ID ^(e.g., run_01^)
        exit /b 1
    )
    set RUN_ID=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--reuse-data" (
    set REUSE_DATA=true
    shift
    goto :parse_args
)
if "%1"=="--no-reuse-data" (
    set REUSE_DATA=false
    shift
    goto :parse_args
)
REM Unknown argument, skip it
shift
goto :parse_args

:start_experiment
set SCRIPT_DIR=%~dp0
set CONFIG_DIR=%SCRIPT_DIR%generated_configs\%EXPERIMENT_NAME%

echo ========================================
echo Running Full Causal Pipeline Experiment: %RUN_ID%/%EXPERIMENT_NAME%
echo DEBUG: RUN_BASELINE=%RUN_BASELINE%, RUN_BERT=%RUN_BERT%, RUN_ID=%RUN_ID%
if "%RUN_BASELINE%"=="true" if "%RUN_BERT%"=="true" (
    echo Mode: BASELINE + BERT
) else if "%RUN_BASELINE%"=="true" (
    echo Mode: BASELINE ONLY
) else (
    echo Mode: BERT ONLY
)
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
echo DEBUG: About to run: python ..\python_scripts\generate_configs.py %EXPERIMENT_NAME% --run_id %RUN_ID%
python ..\python_scripts\generate_configs.py %EXPERIMENT_NAME% --run_id %RUN_ID%
set CONFIG_EXIT_CODE=!errorlevel!
echo DEBUG: Config generation exit code: !CONFIG_EXIT_CODE!
if !CONFIG_EXIT_CODE! neq 0 (
    echo ERROR: Failed to generate configs with exit code !CONFIG_EXIT_CODE!
    pause
    exit /b 1
)

REM Verify that config files were created
echo DEBUG: Checking if config files were generated...
echo DEBUG: Current working directory: %CD%
echo DEBUG: Looking for: ..\generated_configs\%EXPERIMENT_NAME%\simulation.yaml
if exist "..\generated_configs\%EXPERIMENT_NAME%\simulation.yaml" (
    echo DEBUG: simulation.yaml found in ..\generated_configs\%EXPERIMENT_NAME%\
) else (
    echo ERROR: simulation.yaml not found in ..\generated_configs\%EXPERIMENT_NAME%\
    echo DEBUG: Let's see what exists in the generated_configs directory:
    dir "..\generated_configs\%EXPERIMENT_NAME%\" 2>nul || echo "Directory does not exist"
    pause
    exit /b 1
)

echo.
echo Step 2: Running pipeline steps...
echo.

REM Change to project root for running pipeline commands
cd ..\..\..
echo DEBUG: Current directory after cd: %CD%
echo DEBUG: Config path will be: experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\simulation.yaml

REM Verify the config file exists from this directory
if exist "experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\simulation.yaml" (
    echo DEBUG: Config file exists at expected path from current directory
) else (
    echo ERROR: Config file NOT found at expected path from current directory
    echo DEBUG: Let's see what's in the generated_configs directory:
    dir "experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\" 2>nul || echo "Directory does not exist"
    pause
    exit /b 1
)

REM --- Shared Data Preparation ---
REM Check if we should reuse data from run_01
set SHOULD_REUSE=false
if "%REUSE_DATA%"=="true" if not "%RUN_ID%"=="run_01" (
    REM Check if run_01 data exists for this experiment
    set RUN_01_PREPARED=.\outputs\causal\sim_study\runs\run_01\%EXPERIMENT_NAME%\prepared_data
    if exist "!RUN_01_PREPARED!" (
        set SHOULD_REUSE=true
        echo.
        echo =======================================
        echo Reusing prepared data from run_01
        echo =======================================
        echo This ensures identical data across runs for proper variance measurement.
        echo.
        
        REM Create target directories
        set TARGET_BASE=.\outputs\causal\sim_study\runs\%RUN_ID%\%EXPERIMENT_NAME%
        if not exist "!TARGET_BASE!" mkdir "!TARGET_BASE!"
        
        REM Copy the prepared data directories from run_01
        echo Copying simulated_outcomes...
        xcopy /E /I /Y ".\outputs\causal\sim_study\runs\run_01\%EXPERIMENT_NAME%\simulated_outcomes" "!TARGET_BASE!\simulated_outcomes"
        
        echo Copying cohort...
        xcopy /E /I /Y ".\outputs\causal\sim_study\runs\run_01\%EXPERIMENT_NAME%\cohort" "!TARGET_BASE!\cohort"
        
        echo Copying prepared_data...
        xcopy /E /I /Y ".\outputs\causal\sim_study\runs\run_01\%EXPERIMENT_NAME%\prepared_data" "!TARGET_BASE!\prepared_data"
        
        echo Data reuse complete. Skipping simulation and preparation steps.
        echo.
    ) else (
        echo Note: --reuse-data enabled but run_01 data not found at !RUN_01_PREPARED!
        echo Will generate new data for this run.
        echo.
    )
)

REM Only run data preparation if we're not reusing
if "%SHOULD_REUSE%"=="false" if "%RUN_BASELINE%"=="true" (
    echo ======================================
    echo PHASE 1: SHARED DATA PREPARATION
    echo ======================================
    
    echo ==== Running simulate_outcomes... ====
    echo DEBUG: About to run: python -m corebehrt.main_causal.simulate_from_sequence --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\simulation.yaml
    python -m corebehrt.main_causal.simulate_from_sequence --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\simulation.yaml
    set PYTHON_EXIT_CODE=!errorlevel!
    echo DEBUG: Python exit code: !PYTHON_EXIT_CODE!
    if !PYTHON_EXIT_CODE! neq 0 (
        echo ERROR: simulate_outcomes failed with exit code !PYTHON_EXIT_CODE!
        goto :error
    )

    echo ==== Running select_cohort... ====
    python -m corebehrt.main_causal.select_cohort_full --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\select_cohort.yaml
    if errorlevel 1 goto :error

    echo ==== Running prepare_finetune_data... ====
    python -m corebehrt.main_causal.prepare_ft_exp_y --config_path experiments\causal_pipeline\generated_configs\%EXPERIMENT_NAME%\prepare_finetune.yaml
    if errorlevel 1 goto :error
)

if "%RUN_BASELINE%"=="true" (
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
echo Experiment %RUN_ID%/%EXPERIMENT_NAME% completed successfully!
echo Results saved in: outputs\causal\sim_study\runs\%RUN_ID%\%EXPERIMENT_NAME%\
if "%RUN_BASELINE%"=="true" (
    echo   - Baseline results: outputs\causal\sim_study\runs\%RUN_ID%\%EXPERIMENT_NAME%\estimate\baseline\
)
if "%RUN_BERT%"=="true" (
    echo   - BERT results: outputs\causal\sim_study\runs\%RUN_ID%\%EXPERIMENT_NAME%\estimate\bert\
)
echo ========================================

REM Only pause if not running in batch mode
if not "%BATCH_MODE%"=="true" pause
exit /b 0

:error
    echo.
    echo ======================================
    echo EXPERIMENT FAILED: %EXPERIMENT_NAME%
    echo ======================================
    echo An error occurred in the %EXPERIMENT_NAME% experiment.
    echo Last error code: !PYTHON_EXIT_CODE!
    echo Current directory: %CD%
    echo Check the output above for the Python traceback.
    echo Terminating pipeline.
    echo ======================================

REM Only pause if not running in batch mode
if not "%BATCH_MODE%"=="true" pause
exit /b 1
