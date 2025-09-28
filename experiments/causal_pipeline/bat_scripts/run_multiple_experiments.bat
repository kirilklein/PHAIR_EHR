@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run Multiple Causal Pipeline Experiments
REM ========================================

set RUN_MODE=both
set EXPERIMENT_LIST=

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
echo NOTES:
echo   - Requires conda environment 'phair_ehr' to be available
echo   - Experiment configs are read from: ..\experiment_configs\^<experiment_name^>.yaml
echo   - Results are saved to: ..\..\outputs\causal\sim_study\runs\^<experiment_name^>\
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

if "%RUN_MODE%"=="both" (
    echo Mode: BASELINE + BERT
) else if "%RUN_MODE%"=="baseline" (
    echo Mode: BASELINE ONLY
) else (
    echo Mode: BERT ONLY
)
echo ========================================

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

REM Set batch mode to prevent pausing
set BATCH_MODE=true

set FAILED_EXPERIMENTS=
set SUCCESS_COUNT=0
set TOTAL_COUNT=0

REM Count experiments first
for %%e in (%EXPERIMENT_LIST%) do (
    set /a TOTAL_COUNT+=1
)

echo Found %TOTAL_COUNT% experiments to run: %EXPERIMENT_LIST%
echo.

set CURRENT_COUNT=0

REM Run each experiment
for %%e in (%EXPERIMENT_LIST%) do (
    set EXPERIMENT_NAME=%%e
    set /a CURRENT_COUNT+=1

    echo.
    echo ----------------------------------------
    echo Running experiment !CURRENT_COUNT! of %TOTAL_COUNT%: !EXPERIMENT_NAME!
    echo ----------------------------------------

    REM Call experiment with proper argument passing
    echo DEBUG: Calling run_experiment.bat with experiment: !EXPERIMENT_NAME!, mode: %RUN_MODE%
    if "%RUN_MODE%"=="baseline" (
        call run_experiment.bat !EXPERIMENT_NAME! --baseline-only
    ) else if "%RUN_MODE%"=="bert" (
        call run_experiment.bat !EXPERIMENT_NAME! --bert-only
    ) else (
        call run_experiment.bat !EXPERIMENT_NAME!
    )
    set experiment_result=!errorlevel!

    if !experiment_result! neq 0 (
        echo.
        echo ERROR: Experiment !EXPERIMENT_NAME! failed with error code !experiment_result!
        echo Check the output above for detailed error information.
        echo.
        set FAILED_EXPERIMENTS=!FAILED_EXPERIMENTS! !EXPERIMENT_NAME!
    ) else (
        echo SUCCESS: Experiment !EXPERIMENT_NAME! completed!
        set /a SUCCESS_COUNT+=1
    )
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

