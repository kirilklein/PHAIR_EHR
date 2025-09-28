@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run Multiple Causal Pipeline Experiments
REM ========================================

if "%1"=="" (
    echo Usage: run_multiple_experiments.bat experiment1 experiment2 experiment3 ...
    echo.
    echo Available experiments:
    for %%f in (..\experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    echo.
    echo Example: run_multiple_experiments.bat no_influence strong_confounding minimal_confounding
    pause
    exit /b 1
)

echo ========================================
echo Running Multiple Causal Pipeline Experiments
echo ========================================

REM Set batch mode to prevent pausing
set BATCH_MODE=true

set FAILED_EXPERIMENTS=
set SUCCESS_COUNT=0
set TOTAL_COUNT=0

:loop
if "%1"=="" goto :summary

set EXPERIMENT_NAME=%1
set /a TOTAL_COUNT+=1

echo.
echo ----------------------------------------
echo Running experiment %TOTAL_COUNT%: %EXPERIMENT_NAME%
echo ----------------------------------------

call run_experiment.bat %EXPERIMENT_NAME%

if errorlevel 1 (
    echo ERROR: Experiment %EXPERIMENT_NAME% failed!
    set FAILED_EXPERIMENTS=!FAILED_EXPERIMENTS! %EXPERIMENT_NAME%
) else (
    echo SUCCESS: Experiment %EXPERIMENT_NAME% completed!
    set /a SUCCESS_COUNT+=1
)

shift
goto :loop

:summary
echo.
echo ========================================
echo SUMMARY: Multiple Experiments Completed
echo ========================================
echo Total experiments: %TOTAL_COUNT%
echo Successful: %SUCCESS_COUNT%
echo Failed: %FAILED_EXPERIMENTS%

if not "%FAILED_EXPERIMENTS%"=="" (
    echo.
    echo Failed experiments:%FAILED_EXPERIMENTS%
    pause
    exit /b 1
) else (
    echo.
    echo All experiments completed successfully!
    pause
    exit /b 0
)

