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
echo DEBUG: Initial flags - RUN_BASELINE=%RUN_BASELINE%, RUN_BERT=%RUN_BERT%

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
REM Unknown argument, skip it
shift
goto :parse_args

:start_experiment
set SCRIPT_DIR=%~dp0
set CONFIG_DIR=%SCRIPT_DIR%generated_configs\%EXPERIMENT_NAME%

echo ========================================
echo Running Full Causal Pipeline Experiment: %EXPERIMENT_NAME%
echo DEBUG: RUN_BASELINE=%RUN_BASELINE%, RUN_BERT=%RUN_BERT%
if "%RUN_BASELINE%"=="true" if "%RUN_BERT%"=="true" (
    echo Mode: BASELINE + BERT
) else if "%RUN_BASELINE%"=="true" (
    echo Mode: BASELINE ONLY
) else (
    echo Mode: BERT ONLY
)
echo ========================================

REM Only activate conda if not already in batch mode (to avoid double activation)
if not "%BATCH_MODE%"=="true" (
    echo.
    echo Setting up conda environment...
    call C:/Users/fjn197/Miniconda3/Scripts/activate
    call conda activate phair_ehr
    if errorlevel 1 (
        echo ERROR: Failed to activate conda environment 'phair_ehr'
        echo Please ensure the environment exists and conda is properly installed.
        pause
        exit /b 1
    )
    echo Conda environment 'phair_ehr' activated successfully.
    echo.
)

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
echo DEBUG: About to run: python ..\python_scripts\generate_configs.py %EXPERIMENT_NAME%
python ..\python_scripts\generate_configs.py %EXPERIMENT_NAME%
set CONFIG_EXIT_CODE=!errorlevel!
echo DEBUG: Config generation exit code: !CONFIG_EXIT_CODE!
if !CONFIG_EXIT_CODE! neq 0 (
    echo ERROR: Failed to generate configs with exit code !CONFIG_EXIT_CODE!
    pause
    exit /b 1
)

REM Verify that config files were created
echo DEBUG: Checking if config files were generated...
if exist "generated_configs\%EXPERIMENT_NAME%\simulation.yaml" (
    echo DEBUG: simulation.yaml found in generated_configs\%EXPERIMENT_NAME%\
) else (
    echo ERROR: simulation.yaml not found in generated_configs\%EXPERIMENT_NAME%\
    pause
    exit /b 1
)

echo.
echo Step 2: Running pipeline steps...
echo.

REM Change to project root for running pipeline commands
cd ..\..
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

if "%RUN_BASELINE%"=="true" (
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
echo Results saved in: outputs\causal\sim_study\runs\%EXPERIMENT_NAME%\
if "%RUN_BASELINE%"=="true" (
    echo   - Baseline results: outputs\causal\sim_study\runs\%EXPERIMENT_NAME%\estimate\baseline\
)
if "%RUN_BERT%"=="true" (
    echo   - BERT results: outputs\causal\sim_study\runs\%EXPERIMENT_NAME%\estimate\bert\
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
