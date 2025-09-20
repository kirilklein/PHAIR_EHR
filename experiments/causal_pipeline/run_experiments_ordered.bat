@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run Experiments in Custom Order
REM ========================================

if "%1"=="" (
    echo Usage: run_experiments_ordered.bat experiment1 experiment2 experiment3 ...
    echo.
    echo This script runs experiments in the exact order specified.
    echo Each experiment will complete ^(successfully or with failure^) before the next starts.
    echo.
    echo Available experiments:
    for %%f in (experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    echo.
    echo Examples:
    echo   run_experiments_ordered.bat ce0_cy0_y0_i0 ce0p5_cy0p5_y0_i0 ce1_cy1_y0_i0
    echo   run_experiments_ordered.bat ce0_cy0_y0_i5
    echo.
    echo To run ALL experiments: run_all_experiments.bat
    pause
    exit /b 1
)

echo ========================================
echo Running Experiments in Custom Order
echo ========================================
echo.

REM Count experiments to run
set TOTAL_EXPERIMENTS=0
set EXPERIMENT_LIST=
:count_loop
if "%1"=="" goto :start_experiments
set /a TOTAL_EXPERIMENTS+=1
if "!EXPERIMENT_LIST!"=="" (
    set EXPERIMENT_LIST=%1
) else (
    set EXPERIMENT_LIST=!EXPERIMENT_LIST!, %1
)
shift
goto :count_loop

:start_experiments
echo Will run %TOTAL_EXPERIMENTS% experiments in order: %EXPERIMENT_LIST%
echo.

REM Initialize tracking
set SUCCESSFUL_EXPERIMENTS=0
set FAILED_EXPERIMENTS=0
set FAILED_LIST=

REM Get timestamp for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "NN=%dt:~10,2%" & set "SS=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%NN%-%SS%"

set LOG_FILE=run_ordered_experiments_%timestamp%.log

echo Starting ordered run at %YYYY%-%MM%-%DD% %HH%:%NN%:%SS%
echo Logging to: %LOG_FILE%
echo.

REM Set batch mode to prevent pausing
set BATCH_MODE=true

REM Reset argument processing
set CURRENT_EXPERIMENT=0
call :process_experiments %EXPERIMENT_LIST%

goto :show_summary

:process_experiments
:experiment_loop
if "%1"=="" goto :eof

set experiment_name=%1
set experiment_name=!experiment_name:,=!
if "!experiment_name!"=="" goto :next_experiment

set /a CURRENT_EXPERIMENT+=1

echo ========================================
echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !experiment_name!
echo ========================================

REM Check if experiment config exists
if not exist "experiment_configs\!experiment_name!.yaml" (
    echo ERROR: Experiment config not found: experiment_configs\!experiment_name!.yaml
    echo [!time!] ERROR: Config not found for !experiment_name! >> %LOG_FILE%
    set /a FAILED_EXPERIMENTS+=1
    if "!FAILED_LIST!"=="" (
        set FAILED_LIST=!experiment_name! ^(config not found^)
    ) else (
        set FAILED_LIST=!FAILED_LIST!, !experiment_name! ^(config not found^)
    )
    goto :next_experiment
)

REM Log start
echo [!time!] Starting experiment: !experiment_name! >> %LOG_FILE%

REM Run experiment
call run_experiment.bat !experiment_name!
set experiment_result=!errorlevel!

REM Log result
if !experiment_result! equ 0 (
    echo [!time!] SUCCESS: !experiment_name! >> %LOG_FILE%
    echo SUCCESS: Experiment !experiment_name! completed successfully
    set /a SUCCESSFUL_EXPERIMENTS+=1
) else (
    echo [!time!] FAILED: !experiment_name! ^(error code !experiment_result!^) >> %LOG_FILE%
    echo FAILED: Experiment !experiment_name! failed with error code !experiment_result!
    set /a FAILED_EXPERIMENTS+=1
    if "!FAILED_LIST!"=="" (
        set FAILED_LIST=!experiment_name!
    ) else (
        set FAILED_LIST=!FAILED_LIST!, !experiment_name!
    )
)

echo.
if !CURRENT_EXPERIMENT! lss %TOTAL_EXPERIMENTS% (
    echo Continuing to next experiment...
    echo.
    timeout /t 2 /nobreak >nul
)

:next_experiment
shift
goto :experiment_loop

:show_summary
echo ========================================
echo ORDERED RUN SUMMARY
echo ========================================
echo Total experiments: %TOTAL_EXPERIMENTS%
echo Successful: %SUCCESSFUL_EXPERIMENTS%
echo Failed: %FAILED_EXPERIMENTS%
echo.

if %FAILED_EXPERIMENTS% gtr 0 (
    echo Failed experiments: !FAILED_LIST!
    echo.
)

echo Results logged to: %LOG_FILE%
echo.

REM Log summary
echo ======================================== >> %LOG_FILE%
echo ORDERED RUN SUMMARY >> %LOG_FILE%
echo ======================================== >> %LOG_FILE%
echo Experiment order: %EXPERIMENT_LIST% >> %LOG_FILE%
echo Total experiments: %TOTAL_EXPERIMENTS% >> %LOG_FILE%
echo Successful: %SUCCESSFUL_EXPERIMENTS% >> %LOG_FILE%
echo Failed: %FAILED_EXPERIMENTS% >> %LOG_FILE%
if %FAILED_EXPERIMENTS% gtr 0 (
    echo Failed experiments: !FAILED_LIST! >> %LOG_FILE%
)

if %FAILED_EXPERIMENTS% equ 0 (
    echo All experiments completed successfully!
    pause
    exit /b 0
) else (
    echo Some experiments failed. Check the log file for details.
    pause
    exit /b 1
)
