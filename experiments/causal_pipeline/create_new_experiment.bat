@echo off

if "%1"=="" (
    echo Usage: create_new_experiment.bat ^<experiment_name^>
    echo.
    echo Example: create_new_experiment.bat my_new_experiment
    pause
    exit /b 1
)

python scripts\create_new_experiment.py %1

