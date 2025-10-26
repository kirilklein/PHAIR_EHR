@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo Available Causal Pipeline Experiments
echo ========================================
echo.

if not exist ..\experiment_configs\*.yaml (
    echo No experiments found in ../experiment_configs/
    echo.
    echo Create a new experiment with: create_new_experiment.bat ^<name^>
    pause
    exit /b 0
)

echo Experiment Name         Description
echo ----------------       -----------

for %%f in (..\experiment_configs\*.yaml) do (
    set filename=%%~nf
    
    REM Try to extract description from YAML file
    set description=
    for /f "tokens=2*" %%a in ('findstr /c:"description:" "%%f" 2^>nul') do (
        set description=%%b
    )
    
    REM Remove quotes if present
    set description=!description:"=!
    
    if "!description!"=="" set description=No description
    
    echo !filename!                   !description!
)

echo.
echo Usage Examples:
echo   run_experiment.bat ^<experiment_name^>
echo   run_multiple_experiments.bat exp1 exp2 exp3
echo   create_new_experiment.bat ^<new_experiment_name^>
echo.
pause
