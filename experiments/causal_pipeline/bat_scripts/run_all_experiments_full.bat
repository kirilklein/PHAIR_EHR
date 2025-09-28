@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run All Full Causal Pipeline Experiments (Baseline + BERT)
REM ========================================

set SKIP_EXISTING=false
set RUN_MODE=both

REM Parse command line arguments
:parse_args
if "%1"=="" goto :start_main
if "%1"=="--skip-existing" (
    set SKIP_EXISTING=true
    shift
    goto :parse_args
)
if "%1"=="-s" (
    set SKIP_EXISTING=true
    shift
    goto :parse_args
)
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
shift
goto :parse_args

:start_main
echo ========================================
echo Running All Full Causal Pipeline Experiments
if "%SKIP_EXISTING%"=="true" (
    echo Mode: SKIPPING existing experiments
) else (
    echo Mode: Re-running ALL experiments
)
if "%RUN_MODE%"=="both" (
    echo Pipeline: BASELINE + BERT
) else if "%RUN_MODE%"=="baseline" (
    echo Pipeline: BASELINE ONLY
) else (
    echo Pipeline: BERT ONLY
)
echo ========================================
echo.

REM Check if any experiment configs exist
if not exist "..\experiment_configs\*.yaml" (
    echo ERROR: No experiment configs found in ..\experiment_configs\
    echo.
    echo Create experiments with: create_new_experiment.bat ^<name^>
    pause
    exit /b 1
)

REM Initialize tracking variables
set TOTAL_EXPERIMENTS=0
set SUCCESSFUL_EXPERIMENTS=0
set FAILED_EXPERIMENTS=0
set SKIPPED_EXPERIMENTS=0
set FAILED_LIST=

REM Count total experiments
for %%f in (..\experiment_configs\*.yaml) do (
    set /a TOTAL_EXPERIMENTS+=1
)

echo Found %TOTAL_EXPERIMENTS% experiments to run
echo.

REM Get current timestamp for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "NN=%dt:~10,2%" & set "SS=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%NN%-%SS%"

set LOG_FILE=run_all_experiments_full_%timestamp%.log

echo Starting batch run at %YYYY%-%MM%-%DD% %HH%:%NN%:%SS%
echo Logging to: %LOG_FILE%
echo.

REM Set batch mode to prevent pausing
set BATCH_MODE=true

REM Run each experiment
set CURRENT_EXPERIMENT=0
for %%f in (..\experiment_configs\*.yaml) do (
    set filename=%%~nf
    set /a CURRENT_EXPERIMENT+=1
    
    REM Check if experiment already exists and should be skipped
    if "%SKIP_EXISTING%"=="true" (
        set SHOULD_SKIP=false
        
        REM Check what exists based on run mode
        if "%RUN_MODE%"=="baseline" (
            if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\baseline\estimate_results.csv" (
                set SHOULD_SKIP=true
            ) else if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\estimate_results.csv" (
                set SHOULD_SKIP=true
            )
        ) else if "%RUN_MODE%"=="bert" (
            if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\bert\estimate_results.csv" (
                set SHOULD_SKIP=true
            )
        ) else (
            REM both mode - skip only if both exist
            if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\baseline\estimate_results.csv" (
                if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\bert\estimate_results.csv" (
                    set SHOULD_SKIP=true
                )
            ) else if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\estimate_results.csv" (
                if exist "..\..\outputs\causal\sim_study\runs\!filename!\estimate\bert\estimate_results.csv" (
                    set SHOULD_SKIP=true
                )
            )
        )
        
        if "!SHOULD_SKIP!"=="true" (
            echo ========================================
            echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !filename! ^(SKIPPED - already exists^)
            echo ========================================
            set /a SKIPPED_EXPERIMENTS+=1
            echo [!time!] SKIPPED: !filename! ^(already exists^) >> %LOG_FILE%
            goto :skip_experiment
        )
    )
    
    echo ========================================
    echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !filename!
    echo ========================================
    
    REM Log start time
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "start_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    echo [!start_time!] Starting experiment: !filename! >> %LOG_FILE%
    
    REM Build command arguments
    set EXPERIMENT_ARGS=!filename!
    if "%RUN_MODE%"=="baseline" (
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --baseline-only
    ) else if "%RUN_MODE%"=="bert" (
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --bert-only
    )
    
    REM Run the experiment
    call run_experiment_full.bat !EXPERIMENT_ARGS!
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
    
    :skip_experiment
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
if "%SKIP_EXISTING%"=="true" (
    echo Skipped: %SKIPPED_EXPERIMENTS%
)
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
if "%SKIP_EXISTING%"=="true" (
    echo Skipped: %SKIPPED_EXPERIMENTS% >> %LOG_FILE%
)
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
