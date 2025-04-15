@echo off
setlocal EnableDelayedExpansion

:: Check if minimum required parameters are provided
if "%~4"=="" (
    echo Usage: %~nx0 exposure_noise outcome_noise exposure_effect margin [weight] [intercept]
    echo Example: %~nx0 0.02 0.05 2.00 0.02 
    echo Example with optional params: %~nx0 0.02 0.05 2.00 0.02 0.1 1.0
    echo Default weight: 0.1
    echo Default intercept: 1.0
    exit /b 1
)

:: Set default values for optional parameters
set "weight=0.1"
set "intercept=1.0"

:: Override defaults if parameters are provided
if not "%~5"=="" set "weight=%~5"
if not "%~6"=="" set "intercept=%~6"

:: Run single test with provided parameters
call :run_estimate_test %~1 %~2 %~3 %~4 %weight% %intercept%
goto :eof

:: Function definition
:run_estimate_test
set "exposure_noise=%~1"
set "outcome_noise=%~2"
set "exposure_effect=%~3"
set "margin=%~4"
set "weight=%~5"
set "intercept=%~6"

echo.
echo Parameters:
echo   exposure_noise=%exposure_noise%
echo   outcome_noise=%outcome_noise%
echo   exposure_effect=%exposure_effect%
echo   margin=%margin%
echo   weight=%weight%
echo   intercept=%intercept%
echo.

:: Generate data
python -m tests.data_generation.generate_estimate_data --exposure-noise %exposure_noise% --outcome-noise %outcome_noise% --exposure-effect %exposure_effect% --weight %weight% --intercept %intercept%
if errorlevel 1 goto :error

:: Run estimation
python -m corebehrt.main_causal.estimate --config_path ./corebehrt/configs/causal/estimate/estimate_generated_data.yaml
if errorlevel 1 goto :error

:: Test results
python -m tests.test_main_causal.test_estimate_result --margin %margin%
if errorlevel 1 goto :error

goto :eof

:error
echo Error occurred in test
exit /b 1