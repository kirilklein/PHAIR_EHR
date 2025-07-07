@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
echo Delete old features
rmdir /s /q outputs\causal\data\features

echo Running create_data...
python -m corebehrt.main.create_data --config_path corebehrt\configs\causal\prepare_and_pretrain\create_data.yaml
if errorlevel 1 goto :error

echo Running prepare_training_data...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\prepare_and_pretrain\prepare_pretrain.yaml
if errorlevel 1 goto :error

echo Running pretrain...
python -m corebehrt.main.pretrain --config_path corebehrt\configs\causal\prepare_and_pretrain\pretrain.yaml
if errorlevel 1 goto :error

:: Run Outcomes and Cohort Selection
echo Running create_outcomes...
python -m corebehrt.main.create_outcomes --config_path corebehrt\configs\causal\outcomes.yaml
if errorlevel 1 goto :error

echo Running select_cohort_full...
python -m corebehrt.main_causal.select_cohort_full --config_path corebehrt\configs\causal\select_cohort_full\extract.yaml
if errorlevel 1 goto :error

:: Prepare Finetune Data
echo Running prepare_finetune_data uncensored...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\finetune\prepare\ft_exp_uncensored.yaml
if errorlevel 1 goto :error

echo Checking Uncensored Data...
echo We expect that all the triggers are the same as the outcomes.
python -m tests.pipeline.test_uncensored_prepared --processed_data_dir ./outputs/causal/finetune/processed_data/exposure_uncensored --trigger_code EXPOSURE
if errorlevel 1 goto :error

echo Running xgb baseline uncensored...
python -m tests.pipeline.train_xgb_baseline --data_path ./outputs/causal/finetune/processed_data/exposure_uncensored --min_roc_auc 0.95 --expected_most_important_concept EXPOSURE
if errorlevel 1 goto :error

echo Running prepare_finetune_data...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\finetune\prepare\ft_exp.yaml
if errorlevel 1 goto :error

echo Running xgb baseline censored...
python -m tests.pipeline.train_xgb_baseline --data_path ./outputs/causal/finetune/processed_data/exposure --min_roc_auc 0.55 --multihot
if errorlevel 1 goto :error

:: Finetune Exposure & Outcome
echo Running finetune uncensored...
python -m corebehrt.main.finetune_cv --config_path corebehrt\configs\causal\finetune\ft_exp_uncensored.yaml
if errorlevel 1 goto :error

echo Checking Performance uncensored...
python -m tests.pipeline.test_performance .\outputs\causal\finetune\models\exposure_uncensored --min 0.8
if errorlevel 1 goto :error

echo Running finetune...
python -m corebehrt.main.finetune_cv --config_path corebehrt\configs\causal\finetune\ft_exp.yaml
if errorlevel 1 goto :error

echo Checking Performance...
python -m tests.pipeline.test_performance .\outputs\causal\finetune\models\exposure --min 0.51 --max 0.8
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