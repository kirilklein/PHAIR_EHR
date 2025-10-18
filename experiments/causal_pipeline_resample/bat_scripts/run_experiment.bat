@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Causal Pipeline RESAMPLING Experiment Runner (Baseline + BERT)
REM This version samples a different subset of the cohort for each run
REM ========================================

echo DEBUG: run_experiment.bat called with arguments: %*
echo DEBUG: First argument %%1 = "%1"
echo DEBUG: Second argument %%2 = "%2"
echo DEBUG: All arguments %%* = %*

if "%1"=="" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help

REM Arguments provided, proceed to main execution
goto :main_execution

:show_help
echo ========================================
echo Causal Pipeline RESAMPLING Experiment Runner
echo ========================================
echo.
echo Usage: run_experiment.bat ^<experiment_name^> [OPTIONS]
echo.
echo ARGUMENTS:
echo   experiment_name       Name of the experiment to run
echo.
echo OPTIONS:
echo   -h, --help              Show this help message
echo   --baseline-only         Run only baseline ^(CatBoost^) pipeline
echo   --bert-only             Run only BERT pipeline ^(requires baseline data^)
echo   --overwrite             Force re-run all steps ^(default: skip completed steps^)
echo   -e, --experiment-dir    Base directory for experiments ^(default: .\outputs\causal\sim_study_sampling\runs^)
echo   --base-configs-dir DIR  Custom base configs directory ^(default: ..\base_configs^)
echo   --base-seed N           Base seed for sampling ^(default: 42^). Actual seed = base_seed + run_number
echo   --sample-fraction F     Fraction of patients to sample ^(0 ^< F ^<= 1, mutually exclusive with --sample-size^)
echo   --sample-size N         Absolute number of patients to sample ^(takes precedence, mutually exclusive with --sample-fraction^)
echo   --meds PATH             Path to MEDS data ^(default: ./example_data/synthea_meds_causal^)
echo   --features PATH         Path to features data ^(default: ./outputs/causal/data/features^)
echo   --tokenized PATH        Path to tokenized data ^(default: ./outputs/causal/data/tokenized^)
echo   --pretrain-model PATH   Path to pretrained model ^(default: ./outputs/causal/pretrain/model^)
echo   ^(no options^)          Run both baseline and BERT pipelines
echo.
echo AVAILABLE EXPERIMENTS:
for %%f in (..\experiment_configs\*.yaml) do (
    set filename=%%~nf
    echo   - !filename!
)
echo.
echo EXAMPLES:
echo   run_experiment.bat my_experiment
echo     ^> Runs both baseline and BERT pipelines for my_experiment
echo.
echo   run_experiment.bat my_experiment --baseline-only
echo     ^> Runs only baseline pipeline
echo.
echo   run_experiment.bat my_experiment --sample-fraction 0.6
echo     ^> Samples 60%% of base cohort
echo.
echo NOTES:
echo   - This is the RESAMPLING variant: samples from MEDS data -^> simulates -^> selects cohort for each run
echo   - No pre-built cohort needed; sampling happens from raw MEDS data
echo   - Seed is calculated as: base_seed + run_number ^(extracted from run_id^)
echo   - Experiment configs are read from: ..\experiment_configs\^<experiment_name^>.yaml
echo   - Generated configs are saved to: generated_configs\^<experiment_name^>\
echo   - Results are saved to: ..\..\outputs\causal\sim_study_sampling\runs\^<experiment_name^>\
echo   - Use Ctrl+C to stop the experiment at any time
echo.
if not "%BATCH_MODE%"=="true" (
    pause
)
exit /b 1

:main_execution

set EXPERIMENT_NAME=%1
shift
echo DEBUG: EXPERIMENT_NAME set to: %EXPERIMENT_NAME%
set RUN_BASELINE=true
set RUN_BERT=true
set RUN_ID=run_01
set OVERWRITE=false
set EXPERIMENTS_DIR=.\outputs\causal\sim_study_sampling\runs
set BASE_CONFIGS_DIR=
set BASE_SEED=42
set SAMPLE_FRACTION=
set SAMPLE_SIZE=
set MEDS_DATA=./example_data/synthea_meds_causal
set FEATURES_DATA=./outputs/causal/data/features
set TOKENIZED_DATA=./outputs/causal/data/tokenized
set PRETRAIN_MODEL=./outputs/causal/pretrain/model
echo DEBUG: Initial flags - RUN_BASELINE=%RUN_BASELINE%, RUN_BERT=%RUN_BERT%, RUN_ID=%RUN_ID%, OVERWRITE=%OVERWRITE%

REM Parse additional arguments
:parse_args
if "%1"=="" goto :start_experiment
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="--baseline-only" (
    set RUN_BERT=false
    shift
    goto :parse_args
)
if "%1"=="--bert-only" (
    set RUN_BASELINE=false
    shift
    goto :parse_args
)
if "%1"=="--overwrite" (
    set OVERWRITE=true
    shift
    goto :parse_args
)
if "%1"=="--run_id" (
    if "%2"=="" (
        echo ERROR: --run_id requires a run ID ^(e.g., run_01^)
        exit /b 1
    )
    set RUN_ID=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-e" (
    if "%2"=="" (
        echo ERROR: -e requires a directory path
        exit /b 1
    )
    set EXPERIMENTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--experiment-dir" (
    if "%2"=="" (
        echo ERROR: --experiment-dir requires a directory path
        exit /b 1
    )
    set EXPERIMENTS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--base-configs-dir" (
    if "%2"=="" (
        echo ERROR: --base-configs-dir requires a directory path
        exit /b 1
    )
    set BASE_CONFIGS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--base-seed" (
    if "%2"=="" (
        echo ERROR: --base-seed requires a number
        exit /b 1
    )
    set BASE_SEED=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--sample-fraction" (
    if "%2"=="" (
        echo ERROR: --sample-fraction requires a number
        exit /b 1
    )
    set SAMPLE_FRACTION=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--sample-size" (
    if "%2"=="" (
        echo ERROR: --sample-size requires a number
        exit /b 1
    )
    set SAMPLE_SIZE=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--meds" (
    if "%2"=="" (
        echo ERROR: --meds requires a path
        exit /b 1
    )
    set MEDS_DATA=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--features" (
    if "%2"=="" (
        echo ERROR: --features requires a path
        exit /b 1
    )
    set FEATURES_DATA=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--tokenized" (
    if "%2"=="" (
        echo ERROR: --tokenized requires a path
        exit /b 1
    )
    set TOKENIZED_DATA=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--pretrain-model" (
    if "%2"=="" (
        echo ERROR: --pretrain-model requires a path
        exit /b 1
    )
    set PRETRAIN_MODEL=%2
    shift
    shift
    goto :parse_args
)
REM Unknown argument, skip it
shift
goto :parse_args

:start_experiment
set SCRIPT_DIR=%~dp0
set CONFIG_DIR=%SCRIPT_DIR%generated_configs\%EXPERIMENT_NAME%

echo ========================================
echo Running RESAMPLING Experiment: %RUN_ID%/%EXPERIMENT_NAME%
echo DEBUG: RUN_BASELINE=%RUN_BASELINE%, RUN_BERT=%RUN_BERT%, RUN_ID=%RUN_ID%
if "%RUN_BASELINE%"=="true" if "%RUN_BERT%"=="true" (
    echo Mode: BASELINE + BERT
) else if "%RUN_BASELINE%"=="true" (
    echo Mode: BASELINE ONLY
) else (
    echo Mode: BERT ONLY
)
echo Base seed: %BASE_SEED%
echo Sample size: %SAMPLE_SIZE%
echo Sample fraction: %SAMPLE_FRACTION%
echo DEBUG: After parsing - SAMPLE_SIZE=%SAMPLE_SIZE%, SAMPLE_FRACTION=%SAMPLE_FRACTION%
echo ========================================

REM Check if experiment config exists
if not exist "..\experiment_configs\%EXPERIMENT_NAME%.yaml" (
    echo ERROR: Experiment config not found: ..\experiment_configs\%EXPERIMENT_NAME%.yaml
    echo.
    echo Available experiments:
    for %%f in (..\experiment_configs\*.yaml) do (
        set filename=%%~nf
        echo   - !filename!
    )
    pause
    exit /b 1
)

REM Generate experiment-specific configs
echo Step 1: Generating experiment configs...
set CONFIG_CMD=python ..\python_scripts\generate_configs.py %EXPERIMENT_NAME% --run_id %RUN_ID% --experiments_dir %EXPERIMENTS_DIR% --base-seed %BASE_SEED% --meds %MEDS_DATA% --features %FEATURES_DATA% --tokenized %TOKENIZED_DATA% --pretrain-model %PRETRAIN_MODEL%
if not "%SAMPLE_SIZE%"=="" (
    set CONFIG_CMD=!CONFIG_CMD! --sample-size %SAMPLE_SIZE%
) else (
    if not "%SAMPLE_FRACTION%"=="" (
        set CONFIG_CMD=!CONFIG_CMD! --sample-fraction %SAMPLE_FRACTION%
    )
)
if not "%BASE_CONFIGS_DIR%"=="" (
    set CONFIG_CMD=!CONFIG_CMD! --base-configs-dir %BASE_CONFIGS_DIR%
)
echo DEBUG: About to run: !CONFIG_CMD!
!CONFIG_CMD!
set CONFIG_EXIT_CODE=!errorlevel!
echo DEBUG: Config generation exit code: !CONFIG_EXIT_CODE!
if !CONFIG_EXIT_CODE! neq 0 (
    echo ERROR: Failed to generate configs with exit code !CONFIG_EXIT_CODE!
    pause
    exit /b 1
)

REM Verify that config files were created
echo DEBUG: Checking if config files were generated...
echo DEBUG: Current working directory: %CD%
echo DEBUG: Looking for: ..\generated_configs\%EXPERIMENT_NAME%\simulation.yaml

if exist "..\generated_configs\%EXPERIMENT_NAME%\simulation.yaml" (
    echo DEBUG: simulation.yaml found in ..\generated_configs\%EXPERIMENT_NAME%\
) else (
    echo ERROR: simulation.yaml not found in ..\generated_configs\%EXPERIMENT_NAME%\
    echo DEBUG: Let's see what exists in the generated_configs directory:
    dir "..\generated_configs\%EXPERIMENT_NAME%\" 2>nul || echo "Directory does not exist"
    pause
    exit /b 1
)

echo.
echo Step 2: Running pipeline steps with resampling...
echo NOTE: Each run samples from MEDS data, then simulates and selects cohort
echo.

REM Change to project root for running pipeline commands
cd ..\..\..
echo DEBUG: Current directory after cd: %CD%
echo DEBUG: Config path will be: experiments\causal_pipeline_resample\generated_configs\%EXPERIMENT_NAME%\simulation.yaml

REM Verify the config file exists from this directory
if exist "experiments\causal_pipeline_resample\generated_configs\%EXPERIMENT_NAME%\simulation.yaml" (
    echo DEBUG: Config file exists at expected path from current directory
) else (
    echo ERROR: Config file NOT found at expected path from current directory
    echo DEBUG: Let's see what's in the generated_configs directory:
    dir "experiments\causal_pipeline_resample\generated_configs\%EXPERIMENT_NAME%\" 2>nul || echo "Directory does not exist"
    pause
    exit /b 1
)

REM --- Data Preparation with Sampling (ALWAYS RUN) ---
echo ======================================
echo PHASE 1: DATA PREPARATION WITH RESAMPLING
echo ======================================

REM Step: simulate_outcomes_with_sampling
set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\simulated_outcomes\counterfactuals.csv
if "%OVERWRITE%"=="true" (
    echo ==== Running simulate_outcomes_with_sampling... ====
    python -m corebehrt.main_causal.simulate_with_sampling --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\simulation.yaml
    if errorlevel 1 goto :error
) else (
    if exist "!TARGET_FILE!" (
        echo ==== Skipping simulate_outcomes_with_sampling ^(output already exists^) ====
    ) else (
        echo ==== Running simulate_outcomes_with_sampling... ====
        python -m corebehrt.main_causal.simulate_with_sampling --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\simulation.yaml
        if errorlevel 1 goto :error
    )
)

REM Step: select_cohort
set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\cohort\pids.pt
if "%OVERWRITE%"=="true" (
    echo ==== Running select_cohort... ====
    python -m corebehrt.main_causal.select_cohort_full --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\select_cohort.yaml
    if errorlevel 1 goto :error
) else (
    if exist "!TARGET_FILE!" (
        echo ==== Skipping select_cohort ^(output already exists^) ====
    ) else (
        echo ==== Running select_cohort... ====
        python -m corebehrt.main_causal.select_cohort_full --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\select_cohort.yaml
        if errorlevel 1 goto :error
    )
)

REM Step: prepare_finetune_data
set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\prepared_data\patients.pt
if "%OVERWRITE%"=="true" (
    echo ==== Running prepare_finetune_data... ====
    python -m corebehrt.main_causal.prepare_ft_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\prepare_finetune.yaml
    if errorlevel 1 goto :error
) else (
    if exist "!TARGET_FILE!" (
        echo ==== Skipping prepare_finetune_data ^(output already exists^) ====
    ) else (
        echo ==== Running prepare_finetune_data... ====
        python -m corebehrt.main_causal.prepare_ft_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\prepare_finetune.yaml
        if errorlevel 1 goto :error
    )
)

REM --- Baseline Pipeline ---
if "%RUN_BASELINE%"=="true" (
    echo.
    echo ======================================
    echo PHASE 2: BASELINE MODELS ^(CATBOOST^)
    echo ======================================

    REM Step: train_baseline
    set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\models\baseline\combined_predictions.csv
    if "%OVERWRITE%"=="true" (
        echo ==== Running train_baseline... ====
        python -m corebehrt.main_causal.train_baseline --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\train_baseline.yaml
        if errorlevel 1 goto :error
    ) else (
        if exist "!TARGET_FILE!" (
            echo ==== Skipping train_baseline ^(output already exists^) ====
        ) else (
            echo ==== Running train_baseline... ====
            python -m corebehrt.main_causal.train_baseline --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\train_baseline.yaml
            if errorlevel 1 goto :error
        )
    )

    REM Step: calibrate (Baseline)
    set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\calibrated\baseline\combined_calibrated_predictions.csv
    if "%OVERWRITE%"=="true" (
        echo ==== Running calibrate ^(Baseline^)... ====
        python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\calibrate.yaml
        if errorlevel 1 goto :error
    ) else (
        if exist "!TARGET_FILE!" (
            echo ==== Skipping calibrate ^(Baseline^) ^(output already exists^) ====
        ) else (
            echo ==== Running calibrate ^(Baseline^)... ====
            python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\calibrate.yaml
            if errorlevel 1 goto :error
        )
    )

    REM Step: estimate (Baseline)
    set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\estimate\baseline\estimate_results.csv
    if "%OVERWRITE%"=="true" (
        echo ==== Running estimate ^(Baseline^)... ====
        python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\estimate.yaml
        if errorlevel 1 goto :error
    ) else (
        if exist "!TARGET_FILE!" (
            echo ==== Skipping estimate ^(Baseline^) ^(output already exists^) ====
        ) else (
            echo ==== Running estimate ^(Baseline^)... ====
            python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\estimate.yaml
            if errorlevel 1 goto :error
        )
    )
)

REM --- BERT Pipeline ---
if "%RUN_BERT%"=="true" (
    echo.
    echo ======================================
    echo PHASE 3: BERT MODELS
    echo ======================================

    REM Step: finetune (BERT)
    set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\models\bert\combined_predictions.csv
    if "%OVERWRITE%"=="true" (
        echo ==== Running finetune ^(BERT^)... ====
        python -m corebehrt.main_causal.finetune_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\finetune_bert.yaml
        if errorlevel 1 goto :error
    ) else (
        if exist "!TARGET_FILE!" (
            echo ==== Skipping finetune ^(BERT^) ^(output already exists^) ====
        ) else (
            echo ==== Running finetune ^(BERT^)... ====
            python -m corebehrt.main_causal.finetune_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\finetune_bert.yaml
            if errorlevel 1 goto :error
        )
    )

    REM Step: calibrate (BERT)
    set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\calibrated\bert\combined_calibrated_predictions.csv
    if "%OVERWRITE%"=="true" (
        echo ==== Running calibrate ^(BERT^)... ====
        python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\calibrate_bert.yaml
        if errorlevel 1 goto :error
    ) else (
        if exist "!TARGET_FILE!" (
            echo ==== Skipping calibrate ^(BERT^) ^(output already exists^) ====
        ) else (
            echo ==== Running calibrate ^(BERT^)... ====
            python -m corebehrt.main_causal.calibrate_exp_y --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\calibrate_bert.yaml
            if errorlevel 1 goto :error
        )
    )

    REM Step: estimate (BERT)
    set TARGET_FILE=!EXPERIMENTS_DIR!\!RUN_ID!\!EXPERIMENT_NAME!\estimate\bert\estimate_results.csv
    if "%OVERWRITE%"=="true" (
        echo ==== Running estimate ^(BERT^)... ====
        python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\estimate_bert.yaml
        if errorlevel 1 goto :error
    ) else (
        if exist "!TARGET_FILE!" (
            echo ==== Skipping estimate ^(BERT^) ^(output already exists^) ====
        ) else (
            echo ==== Running estimate ^(BERT^)... ====
            python -m corebehrt.main_causal.estimate --config_path experiments\causal_pipeline_resample\generated_configs\!EXPERIMENT_NAME!\estimate_bert.yaml
            if errorlevel 1 goto :error
        )
    )
)

echo.
echo ========================================
echo Experiment !RUN_ID!/!EXPERIMENT_NAME! completed successfully!
echo ========================================
echo.
if not "%BATCH_MODE%"=="true" (
    pause
)
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo ERROR: Experiment !RUN_ID!/!EXPERIMENT_NAME! failed!
echo See error messages above for details.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo.
if not "%BATCH_MODE%"=="true" (
    pause
)
exit /b 1

