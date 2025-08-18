@echo off

REM -------------------------------
REM Causal test (data preparation) + XGB test - Windows version
REM Equivalent to causal_test_prep_data_xgb.yml workflow
REM -------------------------------

:: Clean outputs directory
echo ======================================
echo ==== Cleaning outputs directory... ====
rmdir /s /q outputs\causal 2>nul

:: Run the pipeline with inline error checking
echo ======================================
echo ==== Data preparation ====

echo ==== Running create_data... ====
python -m corebehrt.main.create_data --config_path corebehrt\configs\causal\prepare_and_pretrain\create_data.yaml
if errorlevel 1 goto :error

echo ======================================
echo ==== Outcomes and cohort selection ====

echo ==== Running create_outcomes... ====
python -m corebehrt.main.create_outcomes --config_path corebehrt\configs\causal\outcomes.yaml
if errorlevel 1 goto :error

echo ==== Running select_cohort_full extract... ====
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract.yaml
if errorlevel 1 goto :error

echo ==== Running select_cohort_full extract_uncensored... ====
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract_uncensored.yaml
if errorlevel 1 goto :error

echo ======================================
echo ==== Prepare Finetune Data ====

echo ==== Running prepare_ft_exp_y (simple)... ====
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt\configs\causal\finetune\prepare\simple.yaml
if errorlevel 1 goto :error

echo ==== Testing prepare_data_ft_exp_y... ====
python tests\pipeline\prepare_data_ft_exp_y.py .\outputs\causal\finetune\prepared_data
if errorlevel 1 goto :error

echo ======================================
echo ==== Train XGB baseline ====

echo ==== Training XGB baseline on censored data... ====
python tests\pipeline\train_xgb_baseline.py --data_path outputs\causal\finetune\prepared_data --targets exposure OUTCOME OUTCOME_2 OUTCOME_3 --target-bounds "exposure:min:0.6" --target-bounds "OUTCOME:min:0.8" --target-bounds "OUTCOME_2:min:0.68" --target-bounds "OUTCOME_3:min:0.68"
if errorlevel 1 goto :error

echo ======================================
echo ==== Prepare Uncensored Data ====

echo ==== Running prepare_ft_exp_y (uncensored)... ====
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt\configs\causal\finetune\prepare\uncensored.yaml
if errorlevel 1 goto :error

echo ======================================
echo ==== Train XGB on uncensored data ====

echo ==== Training XGB baseline on uncensored data... ====
python tests\pipeline\train_xgb_baseline.py --data_path outputs\causal\finetune\prepared_data_uncensored --targets exposure OUTCOME OUTCOME_2 OUTCOME_3 --target-bounds "exposure:min:0.9" --target-bounds "OUTCOME:min:0.88" --target-bounds "OUTCOME_2:min:0.88" --target-bounds "OUTCOME_3:min:0.88"
if errorlevel 1 goto :error

echo ======================================
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
