@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Generate Configs for All Experiments
REM ========================================

set SKIP_EXISTING=false
set CONFIG_DIR=..\experiment_configs

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
if "%1"=="--config-dir" (
    shift
    if "%1"=="" (
        echo ERROR: --config-dir requires a directory path
        goto :show_help
    )
    set CONFIG_DIR=%1
    shift
    goto :parse_args
)
shift
goto :parse_args

:show_help
echo ========================================
echo Generate Configs for All Experiments
echo ========================================
echo.
echo Usage: generate_configs.bat [OPTIONS]
echo.
echo OPTIONS:
echo   -h, --help           Show this help message
echo   -s, --skip-existing  Skip experiments that already have generated configs
echo   --config-dir DIR     Use DIR instead of default ..\experiment_configs
echo.
echo DESCRIPTION:
echo   Generates pipeline configuration files for all experiments in the
echo   experiment configs directory. Uses generate_configs.py to create
echo   experiment-specific configs from base templates.
echo.
echo EXAMPLES:
echo   generate_configs.bat
echo     ^> Generates configs for all experiments, overwriting existing ones
echo.
echo   generate_configs.bat --skip-existing
echo     ^> Generates configs only for experiments without existing configs
echo.
echo   generate_configs.bat --config-dir custom_experiments
echo     ^> Uses custom_experiments directory instead of default
echo.
echo NOTES:
echo   - Reads experiment configs from: %CONFIG_DIR%\*.yaml
echo   - Generated configs are saved to: ..\generated_configs\^<experiment_name^>\
echo   - Log files are saved to: ..\logs\generate_configs_YYYY-MM-DD_HH-MM-SS.log
echo   - Use Ctrl+C to stop the generation at any time
echo.
pause
exit /b 0

:start_main
echo ========================================
echo Generating Configs for All Experiments
if "%SKIP_EXISTING%"=="true" (
    echo Mode: SKIPPING existing configs
) else (
    echo Mode: Re-generating ALL configs
)
echo Config directory: %CONFIG_DIR%
echo ========================================
echo.

REM Check if any experiment configs exist
if not exist "%CONFIG_DIR%\*.yaml" (
    echo ERROR: No experiment configs found in %CONFIG_DIR%\
    echo.
    echo Make sure you're in the correct directory and experiment configs exist.
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
for %%f in (%CONFIG_DIR%\*.yaml) do (
    set /a TOTAL_EXPERIMENTS+=1
)

echo Found %TOTAL_EXPERIMENTS% experiments to process
echo.

REM Get current timestamp for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "NN=%dt:~10,2%" & set "SS=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%NN%-%SS%"

REM Create logs directory if it doesn't exist
if not exist "..\logs" mkdir "..\logs"

set LOG_FILE=..\logs\generate_configs_%timestamp%.log

echo Starting config generation at %YYYY%-%MM%-%DD% %HH%:%NN%:%SS%
echo Logging to: %LOG_FILE%
echo.

REM Process each experiment
set CURRENT_EXPERIMENT=0
for %%f in (%CONFIG_DIR%\*.yaml) do (
    set filename=%%~nf
    set /a CURRENT_EXPERIMENT+=1
    
    REM Check if configs already exist and should be skipped
    if "%SKIP_EXISTING%"=="true" (
        if exist "..\generated_configs\!filename!\simulation.yaml" (
            echo ========================================
            echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !filename! ^(SKIPPED - already exists^)
            echo ========================================
            set /a SKIPPED_EXPERIMENTS+=1
            echo [!time!] SKIPPED: !filename! ^(configs already exist^) >> %LOG_FILE%
            goto :skip_experiment
        )
    )
    
    echo ========================================
    echo Experiment !CURRENT_EXPERIMENT! of %TOTAL_EXPERIMENTS%: !filename!
    echo ========================================
    
    REM Log start time
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "start_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    echo [!start_time!] Starting config generation: !filename! >> %LOG_FILE%
    
    REM Generate configs
    python ..\python_scripts\generate_configs.py !filename!
    set generation_result=!errorlevel!
    
    REM Log end time and result
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "end_time=%dt:~8,2%:%dt:~10,2%:%dt:~12,2%"
    
    if !generation_result! equ 0 (
        echo [!end_time!] SUCCESS: !filename! >> %LOG_FILE%
        echo SUCCESS: Config generation for !filename! completed successfully
        set /a SUCCESSFUL_EXPERIMENTS+=1
    ) else (
        echo [!end_time!] FAILED: !filename! >> %LOG_FILE%
        echo FAILED: Config generation for !filename! failed with error code !generation_result!
        set /a FAILED_EXPERIMENTS+=1
        if "!FAILED_LIST!"=="" (
            set FAILED_LIST=!filename!
        ) else (
            set FAILED_LIST=!FAILED_LIST!, !filename!
        )
    )
    
    :skip_experiment
    echo.
)

REM Final summary
echo ========================================
echo CONFIG GENERATION SUMMARY
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
echo CONFIG GENERATION SUMMARY >> %LOG_FILE%
echo ======================================== >> %LOG_FILE%
echo Total experiments: %TOTAL_EXPERIMENTS% >> %LOG_FILE%
if "%SKIP_EXISTING%"=="true" (
    echo Skipped: %SKIPPED_EXPERIMENTS% >> %LOG_FILE%
)
echo Successful: %SUCCESSFUL_EXPERIMENTS% >> %LOG_FILE%
echo Failed: %FAILED_EXPERIMENTS% >> %LOG_FILE%
if not "%FAILED_EXPERIMENTS%"=="0" (
    echo Failed experiments: !FAILED_LIST! >> %LOG_FILE%
)

if "%FAILED_EXPERIMENTS%"=="0" (
    echo All configs generated successfully!
    pause
    exit /b 0
) else (
    echo Some config generations failed. Check the log file for details.
    pause
    exit /b 1
)
