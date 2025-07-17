@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
@REM echo ======================================
@REM echo ==== Running Preprocessing and Pretraining... ====
@REM echo ==== Deleting old features... ====
@REM rmdir /s /q outputs\causal\data\features

@REM echo ==== Running create_data... ====
@REM python -m corebehrt.main.create_data --config_path corebehrt\configs\causal\prepare_and_pretrain\create_data.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Running prepare_training_data... ====
@REM python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\prepare_and_pretrain\prepare_pretrain.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Running pretrain... ====
@REM python -m corebehrt.main.pretrain --config_path corebehrt\configs\causal\prepare_and_pretrain\pretrain.yaml
@REM if errorlevel 1 goto :error

@REM :: Run Outcomes and Cohort Selection
@REM echo ==== Running create_outcomes... ====
@REM python -m corebehrt.main.create_outcomes --config_path corebehrt\configs\causal\outcomes.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Running select_cohort... ====
@REM python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract.yaml
@REM if errorlevel 1 goto :error

@REM echo ======================================
@REM echo ==== Running prepare_finetune_data... ====
@REM python -m corebehrt.main_causal.prepare_ft_exp_y --config_path corebehrt\configs\causal\finetune\prepare\simple.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Testing prepare_data_ft_exp_y... ====
@REM python tests\pipeline\prepare_data_ft_exp_y.py .\outputs\causal\finetune\prepared_data
@REM if errorlevel 1 goto :error

echo ==== Running finetune... ====
python -m corebehrt.main_causal.finetune_exp_y --config_path corebehrt\configs\causal\finetune\simple.yaml
if errorlevel 1 goto :error

echo ==== Testing ft_exp_y... ====
python tests\pipeline\ft_exp_y.py .\outputs\causal\finetune\models\simple
if errorlevel 1 goto :error

echo ==== Checking performance... ====
python -m tests.pipeline.test_performance_multitarget .\outputs\causal\finetune\models\simple --target-bounds "exposure:min:0.58,max:0.8" --target-bounds "OUTCOME:min:0.7,max:0.95" --target-bounds "OUTCOME_2:min:0.7,max:0.95" --target-bounds "OUTCOME_3:min:0.58,max:0.95"
if errorlevel 1 goto :error

:: Run Causal Steps
echo ==== Running calibrate... ====
python -m corebehrt.main_causal.calibrate_exp_y --config_path corebehrt\configs\causal\finetune\calibrate.yaml
if errorlevel 1 goto :error

:: Run Estimation
echo ==== Running estimate... ====
python -m corebehrt.main_causal.estimate --config_path corebehrt\configs\causal\estimate.yaml
if errorlevel 1 goto :error

echo ==== Checking estimate... ====
python -m tests.pipeline.test_estimate ./outputs/causal/estimate/simple example_data/synthea_meds_causal/tuning
if errorlevel 1 goto :error

@REM :: Run Criteria and Stats
@REM echo ==== Running extract_criteria... ====
@REM python -m corebehrt.main_causal.helper_scripts.extract_criteria --config_path corebehrt\configs\causal\helper\extract_criteria.yaml
@REM if errorlevel 1 goto :error

@REM echo ==== Running get_stats... ====
@REM python -m corebehrt.main_causal.helper_scripts.get_stats --config_path corebehrt\configs\causal\helper\get_stats.yaml
@REM if errorlevel 1 goto :error

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