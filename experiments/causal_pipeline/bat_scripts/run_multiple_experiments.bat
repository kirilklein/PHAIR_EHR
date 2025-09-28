@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run Multiple Causal Pipeline Experiments
REM ========================================

set RUN_MODE=both
set EXPERIMENT_LIST=
set N_RUNS=1
set RUN_ID_OVERRIDE=

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
echo Run Multiple Causal Pipeline Experiments
echo ========================================
echo.
echo Usage: run_multiple_experiments.bat [OPTIONS] ^<experiment1^> ^<experiment2^> ...
echo.
echo ARGUMENTS:
echo   experiment1 experiment2 ...    Names of experiments to run
echo.
echo OPTIONS:
echo   -h, --help           Show this help message
echo   --n_runs N           Number of runs to execute ^(default: 1, creates run_01, run_02, etc.^)
echo   --run_id run_XX      Specific run ID to use ^(overrides --n_runs^)
echo   --baseline-only      Run only baseline ^(CatBoost^) pipeline for all experiments
echo   --bert-only          Run only BERT pipeline for all experiments ^(requires baseline data^)
echo   ^(no options^)         Run both baseline and BERT pipelines for all experiments
echo.
echo AVAILABLE EXPERIMENTS:
for %%f in (..\experiment_configs\*.yaml) do (
    set filename=%%~nf
    echo   - !filename!
)
echo.
echo EXAMPLES:
echo   run_multiple_experiments.bat ce1_cy1_y0_i0 ce0_cy0_y0_i0
echo     ^> Runs both baseline and BERT pipelines for the specified experiments
echo.
echo   run_multiple_experiments.bat --baseline-only ce1_cy1_y0_i0 ce0_cy0_y0_i0
echo     ^> Runs only baseline pipeline for the specified experiments
echo.
echo   run_multiple_experiments.bat --bert-only ce1_cy1_y0_i0
echo     ^> Runs only BERT pipeline for the specified experiment ^(baseline data must exist^)
echo.
echo   run_multiple_experiments.bat --n_runs 3 ce1_cy1_y0_i0 ce0_cy0_y0_i0
echo     ^> Runs specified experiments 3 times each, creating run_01, run_02, run_03
echo.
echo   run_multiple_experiments.bat --run_id run_05 ce1_cy1_y0_i0
echo     ^> Runs specified experiment in run_05 folder specifically
echo.
echo NOTES:
echo   - Requires conda environment 'phair_ehr' to be available
echo   - Experiment configs are read from: ..\experiment_configs\^<experiment_name^>.yaml
echo   - Results are saved to: ..\..\outputs\causal\sim_study\runs\run_XX\^<experiment_name^>\
echo   - Use Ctrl+C to stop the batch run at any time
echo.
pause
exit /b 0

:check_experiments
if "%EXPERIMENT_LIST%"=="" (
    echo ERROR: No experiments specified
    echo.
    echo Use -h or --help for usage information
    pause
    exit /b 1
)

echo ========================================
echo Running Multiple Causal Pipeline Experiments

REM Determine run configuration
if not "%RUN_ID_OVERRIDE%"=="" (
    echo Run mode: Specific run ^(%RUN_ID_OVERRIDE%^)
    set N_RUNS=1
) else (
    echo Number of runs: %N_RUNS%
    if %N_RUNS% gtr 1 (
        echo Will create: run_01 through run_%N_RUNS:~-2%
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
        if "%RUN_MODE%"=="baseline" (
            call run_experiment.bat !EXPERIMENT_NAME! --run_id !RUN_ID! --baseline-only
        ) else if "%RUN_MODE%"=="bert" (
            call run_experiment.bat !EXPERIMENT_NAME! --run_id !RUN_ID! --bert-only
        ) else (
            call run_experiment.bat !EXPERIMENT_NAME! --run_id !RUN_ID!
        )
        set experiment_result=!errorlevel!

        if !experiment_result! neq 0 (
            echo.
            echo ERROR: Experiment !RUN_ID!/!EXPERIMENT_NAME! failed with error code !experiment_result!
            echo Check the output above for detailed error information.
            echo.
            set FAILED_EXPERIMENTS=!FAILED_EXPERIMENTS! !RUN_ID!/!EXPERIMENT_NAME!
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

if not "%FAILED_EXPERIMENTS%"=="" (
    echo.
    echo Failed experiments:%FAILED_EXPERIMENTS%
    echo.
    echo Some experiments failed. Check the output above for details.
    pause
    exit /b 1
) else (
    echo.
    echo All experiments completed successfully!
    pause
    exit /b 0
)

