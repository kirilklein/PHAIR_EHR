@echo off

REM ========================================
REM Analyze Causal Pipeline Experiment Results
REM ========================================

if "%1"=="" (
    echo Running analysis of ALL experiments since no arguments provided...
    echo Use 'analyze_results.bat --help' for more options.
    echo.
    set RUN_ALL=true
    set RUN_ID=
    set RESULTS_DIR=..\..\..\outputs\causal\sim_study\runs
    set OUTPUT_DIR=..\..\..\outputs\causal\sim_study\analysis
    set OUTCOMES=
    set MAX_SUBPLOTS=
) else (
    set RUN_ALL=false
    set RUN_ID=
    set RESULTS_DIR=..\..\..\outputs\causal\sim_study\runs
    set OUTPUT_DIR=..\..\..\outputs\causal\sim_study\analysis
    set OUTCOMES=
    set MAX_SUBPLOTS=
)

REM Parse arguments
set EXPERIMENTS=
:parse_args
if "%1"=="" goto :start_analysis
if "%1"=="--run_id" (
    set RUN_ID=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--results_dir" (
    set RESULTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--results-dir" (
    set RESULTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--output_dir" (
    set OUTPUT_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--output-dir" (
    set OUTPUT_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--outcomes" (
    set OUTCOMES=%~2
    shift
    shift
    goto :parse_args
)
if "%1"=="--max-subplots" (
    set MAX_SUBPLOTS=%2
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
echo CAUSAL PIPELINE RESULTS ANALYZER
echo =================================
echo.
echo DESCRIPTION:
echo   Analyzes results from causal pipeline experiments. Can process all experiments
echo   or specific ones, with options to filter by run and customize input/output directories.
echo.
echo USAGE:
echo   analyze_results.bat [OPTIONS] [EXPERIMENTS...]
echo.
echo OPTIONS:
echo   --run_id RUN_ID           Analyze results from specific run only ^(e.g., run_01, run_02^)
echo   --results_dir DIR         Directory containing experiment results
echo   --results-dir DIR         ^(alternative: use hyphen instead of underscore^)
echo                             Default: ..\..\..\outputs\causal\sim_study\runs
echo   --output_dir DIR          Directory to save analysis outputs
echo   --output-dir DIR          ^(alternative: use hyphen instead of underscore^)
echo                             Default: ..\..\..\outputs\causal\sim_study\analysis
echo   --outcomes "OUTCOME1 OUTCOME2"  Filter analysis to specific outcomes only
echo                             ^(e.g., --outcomes "OUTCOME_1 OUTCOME_2"^)
echo   --max-subplots N          Maximum subplots per figure ^(creates multiple figures if exceeded^)
echo                             ^(e.g., --max-subplots 4 creates 2x2 grids^)
echo   --help, -h                Show this help message
echo.
echo EXPERIMENTS:
echo   all                       Analyze all available experiments ^(default if none specified^)
echo   experiment_name           Specific experiment to analyze ^(e.g., ce0_cy0_y0_i0^)
echo   exp1 exp2 exp3           Multiple specific experiments
echo.
echo EXAMPLES:
echo   analyze_results.bat
echo     -^> Analyze all experiments, aggregate across all runs
echo.
echo   analyze_results.bat all
echo     -^> Same as above ^(explicit 'all'^)
echo.
echo   analyze_results.bat ce0_cy0_y0_i0 ce1_cy1_y0_i0
echo     -^> Analyze specific experiments, aggregate across all runs
echo.
echo   analyze_results.bat --run_id run_01 all
echo     -^> Analyze all experiments from run_01 only
echo.
echo   analyze_results.bat --run_id run_02 ce0_cy0_y0_i0
echo     -^> Analyze specific experiment from run_02 only
echo.
echo   analyze_results.bat --results_dir C:\custom\path --output_dir C:\custom\output
echo     -^> Use custom input and output directories
echo.
echo   analyze_results.bat --run_id run_01 --results_dir C:\custom\path ce0_cy0_y0_i0
echo     -^> Combine multiple options
echo.
echo   analyze_results.bat --outcomes "OUTCOME_1 OUTCOME_2"
echo     -^> Analyze all experiments but only for specific outcomes
echo.
echo   analyze_results.bat --run_id run_01 --outcomes "OUTCOME_1"
echo     -^> Combine run filtering with outcome filtering
echo.
echo   analyze_results.bat --max-subplots 4
echo     -^> Create 2x2 grids, multiple figures if needed
echo.
echo   analyze_results.bat --max-subplots 6 --outcomes "OUTCOME_1"
echo     -^> Combine max-subplots with outcome filtering
echo.
echo NOTES:
echo   - By default, results are aggregated across all runs ^(run_01, run_02, etc.^)
echo   - Use --run_id to analyze results from a specific run only
echo   - When --run_id is used, it's appended to the results and output directories
echo   - The script will create the output directory if it doesn't exist
echo.
pause
exit /b 0

:start_analysis
REM Directories are now set as defaults above and can be overridden by arguments

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

REM Build the Python command with optional outcomes and max-subplots arguments
set PYTHON_CMD=python ..\python_scripts\analyze_experiment_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR%
if not "%OUTCOMES%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --outcomes %OUTCOMES%
    echo Filtering analysis for outcomes: %OUTCOMES%
)
if not "%MAX_SUBPLOTS%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --max-subplots %MAX_SUBPLOTS%
    echo Using maximum %MAX_SUBPLOTS% subplots per figure
)

if "%RUN_ALL%"=="true" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    %PYTHON_CMD%
) else if "%EXPERIMENTS%"=="all" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    %PYTHON_CMD%
) else if "%EXPERIMENTS%"=="" (
    echo Analyzing ALL experiments in %RESULTS_DIR%
    %PYTHON_CMD%
) else (
    echo WARNING: Specific experiment selection not supported by Python script. Analyzing ALL experiments in %RESULTS_DIR%
    %PYTHON_CMD%
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
