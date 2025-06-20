@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run outcomes
echo Create multiple exposures/targets
python -m corebehrt.main.create_outcomes --config_path corebehrt\configs\causal_multitarget\outcomes.yaml
if errorlevel 1 goto :error

pause
exit /b 0

:error
    echo.
    echo ======================================
    echo An error occurred in one of the modules.
    echo Check the output above for the Python traceback.
    echo Terminating pipeline.
    echo ======================================
pause
exit /b 1