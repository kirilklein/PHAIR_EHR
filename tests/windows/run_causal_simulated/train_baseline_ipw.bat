@echo off

REM -------------------------------
REM Run the IPW-favorable pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining

:: Run Outcomes and Cohort Selection
echo ==== Running simulate_outcomes (IPW-favorable)... ====
python -m corebehrt.main_causal.simulate_from_sequence --config_path corebehrt\configs\causal\simulate_ipw_favorable.yaml
if errorlevel 1 goto :error

echo ==== Running select_cohort (IPW-favorable)... ====
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract_simulated_ipw.yaml
if errorlevel 1 goto :error

echo ======================================
echo ==== Running prepare_finetune_data (IPW-favorable)... ====
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt\configs\causal\finetune\prepare\simulated_ipw.yaml
if errorlevel 1 goto :error

echo ==== Running finetune (IPW-favorable)... ====
python -m corebehrt.main_causal.train_baseline --config_path corebehrt\configs\causal\finetune\simulated_ipw_bl.yaml
if errorlevel 1 goto :error

:: Run Causal Steps
echo ==== Running calibrate (IPW-favorable)... ====
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt\configs\causal\finetune\calibrate_simulated_ipw_bl.yaml
if errorlevel 1 goto :error

:: Run Estimation
echo ==== Running estimate (IPW-favorable)... ====
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate_simulated_ipw_bl.yaml
if errorlevel 1 goto :error


echo ==== Running test_estimate_result (IPW-favorable)... ====
python -m tests.test_main_causal.test_estimate_result --ci_stretch_factor 1.4 --ipw_ci_stretch_factor 1.8 --dir outputs\causal\estimate\baseline\simulated_ipw
if errorlevel 1 goto :error

echo IPW-favorable pipeline completed successfully.
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

