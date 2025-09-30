@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Run All Full Causal Pipeline Experiments (Baseline + BERT)
REM ========================================

set SKIP_EXISTING=false
set RUN_MODE=both
set N_RUNS=1
set REUSE_DATA=true

REM Parse command line arguments
:parse_args
if "%1"=="" goto :start_main
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
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
if "%1"=="-r" (
    set REUSE_DATA=true
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
shift
goto :parse_args

:show_help
echo ========================================
echo Run All Full Causal Pipeline Experiments
echo ========================================
echo.
echo Usage: run_all_experiments_full.bat [OPTIONS]
echo.
echo OPTIONS:
echo   -h, --help           Show this help message
echo   -s, --skip-existing  Skip experiments that already have results
echo   --n_runs N           Number of runs to execute ^(default: 1, creates run_01, run_02, etc.^)
echo   -r, --reuse-data     Reuse prepared data from run_01 for all subsequent runs ^(default: true^)
echo   --no-reuse-data      Force regenerate data for each run ^(not recommended for variance studies^)
echo   --baseline-only      Run only baseline ^(CatBoost^) pipeline for all experiments
echo   --bert-only          Run only BERT pipeline for all experiments ^(requires baseline data^)
echo   ^(no options^)         Run both baseline and BERT pipelines for all experiments
echo.
echo EXAMPLES:
echo   run_all_experiments_full.bat
echo     ^> Runs all experiments with both baseline and BERT pipelines
echo.
echo   run_all_experiments_full.bat --skip-existing
echo     ^> Runs all experiments, skipping those that already have complete results
echo.
echo   run_all_experiments_full.bat --baseline-only -s
echo     ^> Runs only baseline pipeline for all experiments, skipping existing ones
echo.
echo   run_all_experiments_full.bat --bert-only
echo     ^> Runs only BERT pipeline for all experiments ^(baseline data must exist^)
echo.
echo NOTES:
echo   - Experiment configs are read from: ..\experiment_configs\*.yaml
echo   - Results are saved to: ..\..\outputs\causal\sim_study\runs\^<experiment_name^>\
echo   - Log files are saved to: ..\logs\run_all_experiments_full_YYYY-MM-DD_HH-MM-SS.log
echo   - Use Ctrl+C to stop the batch run at any time
echo.
pause
exit /b 0

:start_main
echo ========================================
echo Running All Full Causal Pipeline Experiments
if "%SKIP_EXISTING%"=="true" (
    echo Mode: SKIPPING existing experiments
) else (
    echo Mode: Re-running ALL experiments
)
echo Number of runs: %N_RUNS%
if %N_RUNS% gtr 1 (
    REM Format run numbers with leading zeros
    set /a LAST_RUN=%N_RUNS%
    if !LAST_RUN! lss 10 (
        echo Will create: run_01 through run_0!LAST_RUN!
    ) else (
        echo Will create: run_01 through run_!LAST_RUN!
    )
) else (
    echo Will create: run_01
)

if "%RUN_MODE%"=="both" (
    echo Pipeline: BASELINE + BERT
) else if "%RUN_MODE%"=="baseline" (
    echo Pipeline: BASELINE ONLY
) else (
    echo Pipeline: BERT ONLY
)

if "%REUSE_DATA%"=="true" (
    echo Data reuse: ENABLED ^(run_02+ will reuse run_01 data^)
) else (
    echo Data reuse: DISABLED ^(each run generates new data^)
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
set EXPERIMENTS_COUNT=0
for %%f in (..\experiment_configs\*.yaml) do (
    set /a EXPERIMENTS_COUNT+=1
)

set /a TOTAL_EXPERIMENTS=%EXPERIMENTS_COUNT% * %N_RUNS%
echo Found %EXPERIMENTS_COUNT% unique experiments × %N_RUNS% runs = %TOTAL_EXPERIMENTS% total runs
echo.

REM Get current timestamp for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "NN=%dt:~10,2%" & set "SS=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%NN%-%SS%"

REM Create logs directory if it doesn't exist
if not exist "..\logs" mkdir "..\logs"

set LOG_FILE=..\logs\run_all_experiments_full_%timestamp%.log

echo Starting batch run at %YYYY%-%MM%-%DD% %HH%:%NN%:%SS%
echo Logging to: %LOG_FILE%
echo.

REM Set batch mode to prevent pausing
set BATCH_MODE=true

REM Run each experiment for each run
set CURRENT_EXPERIMENT=0
for /L %%r in (1,1,%N_RUNS%) do (
    REM Format run ID with leading zero
    if %%r lss 10 (
        set RUN_ID=run_0%%r
    ) else (
        set RUN_ID=run_%%r
    )
    
    echo.
    echo =========================================
    echo STARTING RUN %%r of %N_RUNS%: !RUN_ID!
    echo =========================================
    echo.
    
    for %%f in (..\experiment_configs\*.yaml) do (
        set filename=%%~nf
        set /a CURRENT_EXPERIMENT+=1
    
    REM Check if experiment already exists and should be skipped
    if "%SKIP_EXISTING%"=="true" (
        set SHOULD_SKIP=false
        
        REM Check what exists based on run mode
        if "%RUN_MODE%"=="baseline" (
            if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\baseline\estimate_results.csv" (
                set SHOULD_SKIP=true
            ) else if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\estimate_results.csv" (
                set SHOULD_SKIP=true
            )
        ) else if "%RUN_MODE%"=="bert" (
            if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\bert\estimate_results.csv" (
                set SHOULD_SKIP=true
            )
        ) else (
            REM both mode - skip only if both exist
            if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\baseline\estimate_results.csv" (
                if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\bert\estimate_results.csv" (
                    set SHOULD_SKIP=true
                )
            ) else if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\estimate_results.csv" (
                if exist "..\..\..\outputs\causal\sim_study\runs\!RUN_ID!\!filename!\estimate\bert\estimate_results.csv" (
                    set SHOULD_SKIP=true
                )
            )
        )
        
        if "!SHOULD_SKIP!"=="true" (
            echo ========================================
            echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !RUN_ID!/!filename! ^(SKIPPED - already exists^)
            echo ========================================
            set /a SKIPPED_EXPERIMENTS+=1
            echo [!time!] SKIPPED: !RUN_ID!/!filename! ^(already exists^) >> %LOG_FILE%
            goto :skip_experiment
        )
    )
    
    echo ========================================
    echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !RUN_ID!/!filename!
    echo ========================================
    
    REM Log start time
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "start_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    echo [!start_time!] Starting experiment: !RUN_ID!/!filename! >> %LOG_FILE%
    
    REM Build command with run_id and data reuse flag
    set EXPERIMENT_ARGS=!filename! --run_id !RUN_ID!
    if "%RUN_MODE%"=="baseline" (
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --baseline-only
    ) else if "%RUN_MODE%"=="bert" (
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --bert-only
    )
    
    if "%REUSE_DATA%"=="true" (
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --reuse-data
    ) else (
        set EXPERIMENT_ARGS=!EXPERIMENT_ARGS! --no-reuse-data
    )
    
    REM Run the experiment
    call run_experiment.bat !EXPERIMENT_ARGS!
    set experiment_result=!errorlevel!
    
    REM Log end time and result
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "end_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    
    if !experiment_result! equ 0 (
        echo [!end_time!] SUCCESS: !RUN_ID!/!filename! >> %LOG_FILE%
        echo SUCCESS: Experiment !RUN_ID!/!filename! completed successfully
        set /a SUCCESSFUL_EXPERIMENTS+=1
    ) else (
        echo [!end_time!] FAILED: !RUN_ID!/!filename! >> %LOG_FILE%
        echo FAILED: Experiment !RUN_ID!/!filename! failed with error code !experiment_result!
        set /a FAILED_EXPERIMENTS+=1
        if "!FAILED_LIST!"=="" (
            set FAILED_LIST=!RUN_ID!/!filename!
        ) else (
            set FAILED_LIST=!FAILED_LIST!, !RUN_ID!/!filename!
        )
    )
    
    :skip_experiment
    echo.
    echo Continuing to next experiment...
    echo.
    timeout /t 2 /nobreak >nul
    )
    
    echo.
    echo =========================================
    echo COMPLETED RUN %%r of %N_RUNS%: !RUN_ID!
    echo =========================================
    echo.
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
