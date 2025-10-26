@echo off

REM Create outcomes
python -m corebehrt.main.create_outcomes

REM Select cohort
python -m corebehrt.main.select_cohort

REM Prepare finetune
python -m corebehrt.main.prepare_training_data --config_path ./corebehrt/configs/prepare_finetune.yaml

REM Finetune CV
python -m corebehrt.main.finetune_cv

REM Pause to allow you to see the output before the window closes
pause