@echo off

REM -------------------------------
REM Run the pipeline with inline error checking
REM -------------------------------

:: Run the pipeline with inline error checking
:: Run Preprocessing and Pretraining
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

echo Running select_cohort...
python -m corebehrt.main.select_cohort --config_path corebehrt\configs\causal\finetune\select_cohort\exposure_simple_val.yaml
if errorlevel 1 goto :error

echo Running prepare_finetune_data...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\finetune\prepare\ft_exp.yaml
if errorlevel 1 goto :error

echo Running prepare_finetune_data uncensored...
python -m corebehrt.main.prepare_training_data --config_path corebehrt\configs\causal\finetune\prepare\ft_exp_uncensored.yaml
if errorlevel 1 goto :error

echo Running finetune...
python -m corebehrt.main.finetune_cv --config_path corebehrt\configs\causal\finetune\ft_exp.yaml
if errorlevel 1 goto :error

echo Running finetune uncensored...
python -m corebehrt.main.finetune_cv --config_path corebehrt\configs\causal\finetune\ft_exp_uncensored.yaml
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