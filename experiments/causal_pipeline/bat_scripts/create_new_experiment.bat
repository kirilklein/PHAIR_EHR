@echo off

if "%1"=="" (
    echo Usage: create_new_experiment.bat ^<experiment_name^>
    echo.
    echo Creates a new experiment configuration template in experiment_configs/
    echo.
    echo Experiment Name Guidelines:
    echo   - Use underscores instead of spaces ^(e.g., strong_confounding^)
    echo   - Use descriptive names that indicate the experiment purpose
    echo   - Common patterns: influence levels, confounding scenarios, effect sizes
    echo.
    echo Examples:
    echo   create_new_experiment.bat no_influence        ^(baseline with no effects^)
    echo   create_new_experiment.bat strong_confounding  ^(high confounding scenario^)
    echo   create_new_experiment.bat mild_treatment      ^(small treatment effect^)
    echo   create_new_experiment.bat ce1_cy1_y0_i0       ^(systematic naming^)
    echo.
    echo After creation, edit the generated YAML file to customize:
    echo   - Simulation parameters ^(influence scales, factors^)
    echo   - Outcome definitions ^(effect sizes, base rates^)
    echo   - Exposure model settings
    echo.
    pause
    exit /b 1
)

python ..\python_scripts\create_new_experiment.py %1

