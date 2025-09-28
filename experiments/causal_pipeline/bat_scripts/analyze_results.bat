@echo off

REM ========================================
REM Analyze Causal Pipeline Experiment Results
REM ========================================

if "%1"=="" (
    echo Usage: analyze_results.bat [all^|experiment1 experiment2 ...] 
    echo.
    echo Examples:
    echo   analyze_results.bat all                          ^(analyze all experiments^)
    echo   analyze_results.bat ce0_cy0_y0_i0 ce1_cy1_y0_i0  ^(analyze specific experiments^)
    echo.
    echo Running analysis of ALL experiments since no arguments provided...
    set RUN_ALL=true
) else (
    set RUN_ALL=false
)

set RESULTS_DIR=..\..\..\outputs\causal\sim_study\runs
set OUTPUT_DIR=..\..\..\outputs\causal\sim_study\analysis

echo ========================================
echo Analyzing Experiment Results
echo ========================================
echo.

if "%RUN_ALL%"=="true" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR%
) else if "%1"=="all" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR%
) else (
    echo Analyzing specific experiments: %*
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR% --experiments %*
)

if errorlevel 1 (
    echo ERROR: Analysis failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Analysis completed successfully!
echo Results saved in: %OUTPUT_DIR%
echo ========================================
pause
