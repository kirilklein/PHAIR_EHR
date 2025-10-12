@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run Multiple RESAMPLING Causal Pipeline Experiments
REM Each run samples a different subset of the base cohort
REM ========================================

set RUN_MODE=both
set EXPERIMENT_LIST=
set N_RUNS=1
set RUN_ID_OVERRIDE=
set EXPERIMENTS_DIR=.\outputs\causal\sim_study_sampling\runs
set BASE_SEED=42
set SAMPLE_FRACTION=0.5
set MEDS_DATA=./example_data/synthea_meds_causal
set FEATURES_DATA=./outputs/causal/data/features
set TOKENIZED_DATA=./outputs/causal/data/tokenized
set PRETRAIN_MODEL=./outputs/causal/pretrain/model

REM Parse command line arguments
:parse_args
if "%1"=="" goto :check_experiments
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="--baseline-only" (
    set RUN_MODE=baseline
    shift
    goto :parse_args
)
if "%1"=="--bert-only" (
    set RUN_MODE=bert
    shift
    goto :parse_args
)
if "%1"=="--n_runs" (
    if "%2"=="" (
        echo ERROR: --n_runs requires a number
        exit /b 1
    )
    set N_RUNS=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-n" (
    if "%2"=="" (
        echo ERROR: -n requires a number
        exit /b 1
    )
    set N_RUNS=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--run_id" (
    if "%2"=="" (
        echo ERROR: --run_id requires a run ID ^(e.g., run_01^)
        exit /b 1
    )
    set RUN_ID_OVERRIDE=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-e" (
    if "%2"=="" (
        echo ERROR: -e requires a directory path
        exit /b 1
    )
    set EXPERIMENTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--experiment-dir" (
    if "%2"=="" (
        echo ERROR: --experiment-dir requires a directory path
        exit /b 1
    )
    set EXPERIMENTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--base-seed" (
    if "%2"=="" (
        echo ERROR: --base-seed requires a number
        exit /b 1
    )
    set BASE_SEED=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--sample-fraction" (
    if "%2"=="" (
        echo ERROR: --sample-fraction requires a number
        exit /b 1
    )
    set SAMPLE_FRACTION=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--base-cohort" (
    if "%2"=="" (
        echo ERROR: --base-cohort requires a path
        exit /b 1
    )
    set BASE_COHORT_PATH=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--meds" (
    if "%2"=="" (
        echo ERROR: --meds requires a path
        exit /b 1
    )
    set MEDS_DATA=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--features" (
    if "%2"=="" (
        echo ERROR: --features requires a path
        exit /b 1
    )
    set FEATURES_DATA=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--tokenized" (
    if "%2"=="" (
        echo ERROR: --tokenized requires a path
        exit /b 1
    )
    set TOKENIZED_DATA=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--pretrain-model" (
    if "%2"=="" (
        echo ERROR: --pretrain-model requires a path
        exit /b 1
    )
    set PRETRAIN_MODEL=%2
    shift
    shift
    goto :parse_args
)
REM Add experiment to list
if "!EXPERIMENT_LIST!"=="" (
    set EXPERIMENT_LIST=%1
) else (
    set EXPERIMENT_LIST=!EXPERIMENT_LIST! %1
)
shift
goto :parse_args

:show_help
echo ========================================
echo Run Multiple RESAMPLING Causal Pipeline Experiments
echo ========================================
echo.
echo Usage: run_multiple_experiments.bat [OPTIONS] ^<experiment1^> ^<experiment2^> ...
echo.
echo ARGUMENTS:
echo   experiment1 experiment2 ...    Names of experiments to run
echo.
echo OPTIONS:
echo   -h, --help                Show this help message
echo   --n_runs^|-n N             Number of runs to execute ^(default: 1, creates run_01, run_02, etc.^)
echo   --run_id run_XX           Specific run ID to use ^(overrides --n_runs^)
echo   -e, --experiment-dir DIR  Base directory for experiments ^(default: .\outputs\causal\sim_study_sampling\runs^)
echo   --base-seed N             Base seed for sampling ^(default: 42^). Actual seed = base_seed + run_number
echo   --sample-fraction F       Fraction of MEDS patients to sample per run ^(default: 0.5^)
echo   --meds PATH               Path to MEDS data directory ^(default: ./example_data/synthea_meds_causal^)
echo   --features PATH           Path to features directory ^(default: ./outputs/causal/data/features^)
echo   --tokenized PATH          Path to tokenized data directory ^(default: ./outputs/causal/data/tokenized^)
echo   --pretrain-model PATH     Path to pretrained BERT model ^(default: ./outputs/causal/pretrain/model^)
echo   --baseline-only           Run only baseline ^(CatBoost^) pipeline for all experiments
echo   --bert-only               Run only BERT pipeline for all experiments ^(requires baseline data^)
echo   ^(no options^)            Run both baseline and BERT pipelines for all experiments
echo.
echo AVAILABLE EXPERIMENTS:
for %%f in (..\experiment_configs\*.yaml) do (
    set filename=%%~nf
    echo   - !filename!
)
echo.
echo EXAMPLES:
echo   run_multiple_experiments.bat my_experiment
echo     ^> Runs both baseline and BERT pipelines for my_experiment
echo.
echo   run_multiple_experiments.bat --baseline-only my_experiment
echo     ^> Runs only baseline pipeline for my_experiment
echo.
echo   run_multiple_experiments.bat --n_runs 100 my_experiment
echo     ^> Runs my_experiment 100 times, creating run_01 through run_100
echo.
echo   run_multiple_experiments.bat --run_id run_05 my_experiment
echo     ^> Runs my_experiment in run_05 folder specifically
echo.
echo NOTES:
echo   - This is the RESAMPLING variant: samples from MEDS data -^> simulates -^> selects cohort for each run
echo   - No pre-built cohort needed; sampling happens from raw MEDS data
echo   - Seed varies per run: base_seed + run_number
echo   - Experiment configs are read from: ..\experiment_configs\^<experiment_name^>.yaml
echo   - Results are saved to: ..\..\..\outputs\causal\sim_study_sampling\runs\run_XX\^<experiment_name^>\
echo   - Use Ctrl+C to stop the batch run at any time
echo.
if not "%BATCH_MODE%"=="true" (
    pause
)
exit /b 0

:check_experiments
if "%EXPERIMENT_LIST%"=="" (
echo ERROR: No experiments specified
echo.
echo Use -h or --help for usage information
if not "%BATCH_MODE%"=="true" (
    pause
)
exit /b 1
)

echo ========================================
echo Running Multiple RESAMPLING Experiments

REM Determine run configuration
if not "%RUN_ID_OVERRIDE%"=="" (
    echo Run mode: Specific run ^(%RUN_ID_OVERRIDE%^)
    set N_RUNS=1
) else (
    echo Number of runs: %N_RUNS%
    if %N_RUNS% gtr 1 (
        if %N_RUNS% gtr 9 (
            echo Will create: run_01 through run_%N_RUNS%
        ) else (
            echo Will create: run_01 through run_0%N_RUNS%
        )
    ) else (
        echo Will create: run_01
    )
)

if "%RUN_MODE%"=="both" (
    echo Pipeline: BASELINE + BERT
) else if "%RUN_MODE%"=="baseline" (
    echo Pipeline: BASELINE ONLY
) else (
    echo Pipeline: BERT ONLY
)

echo Experiments directory: %EXPERIMENTS_DIR%
echo Base seed: %BASE_SEED%
echo Sample fraction: %SAMPLE_FRACTION%
echo ========================================

REM Set batch mode to prevent pausing
set BATCH_MODE=true

set FAILED_EXPERIMENTS=
set SUCCESS_COUNT=0
set TOTAL_COUNT=0

REM Count experiments first
set EXPERIMENT_COUNT=0
for %%e in (%EXPERIMENT_LIST%) do (
    set /a EXPERIMENT_COUNT+=1
)

set /a TOTAL_COUNT=%EXPERIMENT_COUNT%*%N_RUNS%
echo Found %EXPERIMENT_COUNT% experiments × %N_RUNS% runs = %TOTAL_COUNT% total experiments to run
echo Experiments: %EXPERIMENT_LIST%
echo.

set CURRENT_COUNT=0

REM Run each experiment for each run (using for /L for numeric loop)
for /L %%r in (1,1,%N_RUNS%) do (
    REM Determine run ID
    if not "%RUN_ID_OVERRIDE%"=="" (
        set RUN_ID=%RUN_ID_OVERRIDE%
    ) else (
        if %%r lss 10 (
            set RUN_ID=run_0%%r
        ) else (
            set RUN_ID=run_%%r
        )
    )
    
    echo.
    echo =========================================
    echo STARTING RUN %%r of %N_RUNS%: !RUN_ID!
    echo =========================================
    echo.
    
    for %%e in (%EXPERIMENT_LIST%) do (
        set EXPERIMENT_NAME=%%e
        set /a CURRENT_COUNT+=1

        echo.
        echo ----------------------------------------
        echo Running experiment !CURRENT_COUNT! of %TOTAL_COUNT%: !RUN_ID!/!EXPERIMENT_NAME!
        echo ----------------------------------------

        REM Call experiment with proper argument passing
        echo DEBUG: Calling run_experiment.bat with experiment: !EXPERIMENT_NAME!, run_id: !RUN_ID!, mode: %RUN_MODE%
        
        REM Build command arguments
        set EXPERIMENT_ARGS=!EXPERIMENT_NAME! --run_id !RUN_ID!
        
        if "%RUN_MODE%"=="baseline" (
            set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --baseline-only
        )
        if "%RUN_MODE%"=="bert" (
            set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --bert-only
        )
        
        REM Add experiment directory
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --experiment-dir %EXPERIMENTS_DIR%
        
        REM Add resampling-specific arguments
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --base-seed %BASE_SEED%
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --sample-fraction %SAMPLE_FRACTION%
        
        REM Add data path arguments
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --meds %MEDS_DATA%
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --features %FEATURES_DATA%
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --tokenized %TOKENIZED_DATA%
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --pretrain-model %PRETRAIN_MODEL%
        
        REM Run the experiment
        call run_experiment.bat !EXPERIMENT_ARGS!
        
        set EXPERIMENT_RESULT=!errorlevel!

        if !EXPERIMENT_RESULT! neq 0 (
            echo.
            echo ERROR: Experiment !RUN_ID!/!EXPERIMENT_NAME! failed with error code !EXPERIMENT_RESULT!
            echo Check the output above for detailed error information.
            echo.
            if "!FAILED_EXPERIMENTS!"=="" (
                set FAILED_EXPERIMENTS=!RUN_ID!/!EXPERIMENT_NAME!
            ) else (
                set FAILED_EXPERIMENTS=!FAILED_EXPERIMENTS!, !RUN_ID!/!EXPERIMENT_NAME!
            )
        ) else (
            echo SUCCESS: Experiment !RUN_ID!/!EXPERIMENT_NAME! completed!
            set /a SUCCESS_COUNT+=1
        )
    )
    
    echo.
    echo =========================================
    echo COMPLETED RUN %%r of %N_RUNS%: !RUN_ID!
    echo =========================================
    echo.
)

echo.
echo ========================================
echo SUMMARY: Multiple Experiments Completed
echo ========================================
echo Total experiments: %TOTAL_COUNT%
echo Successful: %SUCCESS_COUNT%
set /a FAILED_COUNT=%TOTAL_COUNT%-%SUCCESS_COUNT%
echo Failed: %FAILED_COUNT%
echo.

if %FAILED_COUNT% gtr 0 (
    echo Failed experiments:
    echo %FAILED_EXPERIMENTS%
    echo.
    echo Some experiments failed. Check the logs above for details.
    if not "%BATCH_MODE%"=="true" (
        pause
    )
    exit /b 1
) else (
    echo All experiments completed successfully!
    if not "%BATCH_MODE%"=="true" (
        pause
    )
    exit /b 0
)

