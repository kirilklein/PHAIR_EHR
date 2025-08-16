@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
echo ==== Running finetune... ====
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt\configs\causal\finetune\uncensored.yaml
if errorlevel 1 goto :error

echo ==== Checking performance... ====
python -m tests.pipeline.test_performance_multitarget .\outputs\causal\finetune\models\uncensored --target-bounds "exposure:min:0.9,max:1" --target-bounds "OUTCOME:min:0.9,max:1" --target-bounds "OUTCOME_2:min:0.9,max:1" --target-bounds "OUTCOME_3:min:0.9,max:1"
if errorlevel 1 goto :error


echo Pipeline completed successfully.
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