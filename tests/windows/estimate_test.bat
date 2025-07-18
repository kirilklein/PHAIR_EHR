@echo off
setlocal EnableDelayedExpansion

echo Running estimate tests with multiple configurations...

:: Run tests with different configurations (matching causal_estimate.yml)
call :run_estimate_test 0 0 2 0.02 "No noise, strong effect" 0.1 1.0
if errorlevel 1 goto :error

call :run_estimate_test 0.02 0.05 2.0 0.02 "Default test" 0.1 1.0
if errorlevel 1 goto :error

call :run_estimate_test 0.05 0.05 1.0 0.02 "High noise test" 0.2 1.5
if errorlevel 1 goto :error

call :run_estimate_test 0.01 0.02 0.5 0.02 "Low noise test" 0.05 0.5
if errorlevel 1 goto :error

call :run_estimate_test 0.02 0.05 2.0 0.02 "Default test" 0.2 -1.0
if errorlevel 1 goto :error

echo All estimate tests completed successfully!
goto :eof

:: Function definition
:run_estimate_test
set "exposure_noise=%~1"
set "outcome_noise=%~2"
set "exposure_effect=%~3"
set "margin=%~4"
set "test_name=%~5"
set "weight=%~6"
set "intercept=%~7"

echo.
echo Running estimate test: %test_name%
echo Parameters:
echo   exposure_noise=%exposure_noise%
echo   outcome_noise=%outcome_noise%
echo   exposure_effect=%exposure_effect%
echo   margin=%margin%
echo   weight=%weight%
echo   intercept=%intercept%
echo.

:: Generate data
echo Generating data...
python -m tests.data_generation.generate_estimate_data --outcome-noise %outcome_noise% --exposure-noise %exposure_noise% --exposure-effect %exposure_effect% --weight %weight% --intercept %intercept%
if errorlevel 1 exit /b 1

:: Run estimation
echo Running estimation...
python -m corebehrt.main_causal.estimate --config_path ./corebehrt/configs/causal/estimate_generated_data.yaml
if errorlevel 1 exit /b 1

:: Test results
echo Testing results...
python -m tests.test_main_causal.test_estimate_result --margin %margin%
if errorlevel 1 exit /b 1

echo Test "%test_name%" completed successfully!
goto :eof

:error
echo Error occurred in estimate tests
exit /b 1