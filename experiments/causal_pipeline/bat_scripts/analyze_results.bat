@echo off

REM ========================================
REM Analyze Causal Pipeline Experiment Results
REM ========================================

if "%1"=="" (
    echo Usage: analyze_results.bat [OPTIONS] [all^|experiment1 experiment2 ...]
    echo.
    echo OPTIONS:
    echo   --run_id run_XX    Analyze results from specific run only
    echo   --help, -h         Show this help message
    echo.
    echo EXAMPLES:
    echo   analyze_results.bat all                          ^(analyze all experiments, aggregate across runs^)
    echo   analyze_results.bat ce0_cy0_y0_i0 ce1_cy1_y0_i0  ^(analyze specific experiments, aggregate across runs^)
    echo   analyze_results.bat --run_id run_01 all          ^(analyze all experiments from run_01 only^)
    echo   analyze_results.bat --run_id run_02 ce0_cy0_y0_i0 ^(analyze specific experiment from run_02 only^)
    echo.
    echo NOTES:
    echo   - By default, results are aggregated across all runs ^(run_01, run_02, etc.^)
    echo   - Use --run_id to analyze results from a specific run only
    echo   - Results are saved to: ..\..\..\outputs\causal\sim_study\analysis\
    echo.
    echo Running analysis of ALL experiments since no arguments provided...
    set RUN_ALL=true
    set RUN_ID=
) else (
    set RUN_ALL=false
    set RUN_ID=
)

REM Parse arguments for --run_id
set EXPERIMENTS=
:parse_args
if "%1"=="" goto :start_analysis
if "%1"=="--run_id" (
    set RUN_ID=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-h" goto :show_full_help
if "%1"=="--help" goto :show_full_help
REM Add to experiments list
if "%EXPERIMENTS%"=="" (
    set EXPERIMENTS=%1
) else (
    set EXPERIMENTS=%EXPERIMENTS% %1
)
shift
goto :parse_args

:show_full_help
REM Show help and exit
pause
exit /b 0

:start_analysis
set RESULTS_DIR=..\..\..\outputs\causal\sim_study\runs
set OUTPUT_DIR=..\..\..\outputs\causal\sim_study\analysis

REM If specific run ID specified, point to that run
if not "%RUN_ID%"=="" (
    set RESULTS_DIR=%RESULTS_DIR%\%RUN_ID%
    set OUTPUT_DIR=%OUTPUT_DIR%\%RUN_ID%
)

echo ========================================
echo Analyzing Experiment Results
echo ========================================
echo.

if not "%RUN_ID%"=="" (
    echo Analyzing results from specific run: %RUN_ID%
) else (
    echo Analyzing results aggregated across all runs
)

if "%RUN_ALL%"=="true" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR%
) else if "%EXPERIMENTS%"=="all" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR%
) else if "%EXPERIMENTS%"=="" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR%
) else (
    echo Analyzing specific experiments: %EXPERIMENTS%
    python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR% --experiments %EXPERIMENTS%
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
