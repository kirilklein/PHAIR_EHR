@echo off
setlocal enabledelayedexpansion

echo ===== Causal Pipeline Performance 2 Target Test Uncensored =====

:: Clean outputs
echo Cleaning outputs...
if exist "outputs\causal" rmdir /s /q "outputs\causal"
if exist "outputs\causal_multitarget" rmdir /s /q "outputs\causal_multitarget"

:: Set error handling
set "error_occurred=false"

:: ─── Data prep & pretrain ────────────────────────────────────
echo.
echo ===== Data prep ^& pretrain =====

echo Running create_data...
python -m corebehrt.main.create_data --config-path corebehrt/configs/causal/prepare_and_pretrain/create_data.yaml
if errorlevel 1 (
    echo ERROR: create_data failed
    set "error_occurred=true"
    goto :error_exit
)

echo Running prepare_training_data...
python -m corebehrt.main.prepare_training_data --config-path corebehrt/configs/causal/prepare_and_pretrain/prepare_pretrain.yaml
if errorlevel 1 (
    echo ERROR: prepare_training_data failed
    set "error_occurred=true"
    goto :error_exit
)

echo Running pretrain...
python -m corebehrt.main.pretrain --config-path corebehrt/configs/causal/prepare_and_pretrain/pretrain.yaml
if errorlevel 1 (
    echo ERROR: pretrain failed
    set "error_occurred=true"
    goto :error_exit
)

echo Running create_outcomes...
python -m corebehrt.main.create_outcomes --config-path corebehrt/configs/causal/outcomes.yaml
if errorlevel 1 (
    echo ERROR: create_outcomes failed
    set "error_occurred=true"
    goto :error_exit
)

echo Running select_cohort_full...
python -m corebehrt.main_causal.select_cohort_full --config-path corebehrt/configs/causal/select_cohort_full/extract.yaml
if errorlevel 1 (
    echo ERROR: select_cohort_full failed
    set "error_occurred=true"
    goto :error_exit
)

:: ─── Prepare Finetune Data ────────────────────────────────────
echo.
echo ===== Prepare Finetune Data =====

echo Running prepare_training_data for multitarget...
python -m corebehrt.main_causal.prepare_training_data --config-path corebehrt/configs/causal_multitarget/prepare/simple.yaml
if errorlevel 1 (
    echo ERROR: prepare_training_data for multitarget failed
    set "error_occurred=true"
    goto :error_exit
)

:: ─── Finetune Exposure Uncensored ────────────────────────────────────
echo.
echo ===== Finetune Exposure Uncensored =====

echo Running finetune_exp_y...
python -m corebehrt.main_causal.finetune_exp_y --config-path corebehrt/configs/causal_multitarget/finetune.yaml
if errorlevel 1 (
    echo ERROR: finetune_exp_y failed
    set "error_occurred=true"
    goto :error_exit
)

echo Running performance test...
python -m tests.pipeline.test_performance outputs\causal_multitarget\finetune\models\simple --min 0.85
if errorlevel 1 (
    echo ERROR: Performance test failed
    set "error_occurred=true"
    goto :error_exit
)

:: Success
echo.
echo ===== SUCCESS: All steps completed successfully! =====
goto :end

:error_exit
echo.
echo ===== ERROR: Pipeline failed! =====
exit /b 1

:end
echo.
echo ===== Pipeline completed =====
          
          
          
         
          
