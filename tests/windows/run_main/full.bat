@echo off

REM Create data
python -m corebehrt.main.create_data

REM Create data without held out
python -m corebehrt.main.create_data --config_path ./corebehrt/configs/create_data_wo_held_out.yaml

REM Prepare pretrain
python -m corebehrt.main.prepare_training_data --config_path ./corebehrt/configs/prepare_pretrain.yaml

REM Pretrain
python -m corebehrt.main.pretrain

REM Create outcomes
python -m corebehrt.main.create_outcomes

REM Select cohort
python -m corebehrt.main.select_cohort

REM Prepare finetune
python -m corebehrt.main.prepare_training_data --config_path ./corebehrt/configs/prepare_finetune.yaml

REM Finetune CV
python -m corebehrt.main.finetune_cv

REM Select cohort absolute
python -m corebehrt.main.select_cohort --config_path ./corebehrt/configs/select_cohort_absolute.yaml

REM Finetune OOT
python -m corebehrt.main.finetune_cv --config_path ./corebehrt/configs/finetune_oot.yaml

REM Pause to allow you to see the output before the window closes
pause