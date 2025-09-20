@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run All Causal Pipeline Experiments Sequentially
REM ========================================

echo ========================================
echo Running All Causal Pipeline Experiments
echo ========================================
echo.

REM Check if any experiment configs exist
if not exist "experiment_configs\*.yaml" (
    echo ERROR: No experiment configs found in experiment_configs\
    echo.
    echo Create experiments with: create_new_experiment.bat ^<name^>
    pause
    exit /b 1
)

REM Initialize tracking variables
set TOTAL_EXPERIMENTS=0
set SUCCESSFUL_EXPERIMENTS=0
set FAILED_EXPERIMENTS=0
set FAILED_LIST=

REM Count total experiments
for %%f in (experiment_configs\*.yaml) do (
    set /a TOTAL_EXPERIMENTS+=1
)

echo Found %TOTAL_EXPERIMENTS% experiments to run
echo.

REM Get current timestamp for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "NN=%dt:~10,2%" & set "SS=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%NN%-%SS%"

set LOG_FILE=run_all_experiments_%timestamp%.log

echo Starting batch run at %YYYY%-%MM%-%DD% %HH%:%NN%:%SS%
echo Logging to: %LOG_FILE%
echo.

REM Set batch mode to prevent pausing
set BATCH_MODE=true

REM Run each experiment
set CURRENT_EXPERIMENT=0
for %%f in (experiment_configs\*.yaml) do (
    set filename=%%~nf
    set /a CURRENT_EXPERIMENT+=1
    
    echo ========================================
    echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !filename!
    echo ========================================
    
    REM Log start time
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "start_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    echo [!start_time!] Starting experiment: !filename! >> %LOG_FILE%
    
    REM Run the experiment
    call run_experiment.bat !filename!
    set experiment_result=!errorlevel!
    
    REM Log end time and result
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "end_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    
    if !experiment_result! equ 0 (
        echo [!end_time!] SUCCESS: !filename! >> %LOG_FILE%
        echo SUCCESS: Experiment !filename! completed successfully
        set /a SUCCESSFUL_EXPERIMENTS+=1
    ) else (
        echo [!end_time!] FAILED: !filename! >> %LOG_FILE%
        echo FAILED: Experiment !filename! failed with error code !experiment_result!
        set /a FAILED_EXPERIMENTS+=1
        if "!FAILED_LIST!"=="" (
            set FAILED_LIST=!filename!
        ) else (
            set FAILED_LIST=!FAILED_LIST!, !filename!
        )
    )
    
    echo.
    echo Continuing to next experiment...
    echo.
    timeout /t 2 /nobreak >nul
)

REM Final summary
echo ========================================
echo BATCH RUN SUMMARY
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

REM Log final summary
echo ======================================== >> %LOG_FILE%
echo BATCH RUN SUMMARY >> %LOG_FILE%
echo ======================================== >> %LOG_FILE%
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
