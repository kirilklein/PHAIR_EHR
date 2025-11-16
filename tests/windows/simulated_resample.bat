@echo off
echo Running simulated resample experiment...
python -m experiments.causal_pipeline_resample.python_scripts.run_multiple_experiments ^
    --inner_runs 2 ^
    --baseline-only ^
    --sample-fraction 0.9 ^
    --meds "./example_data/synthea_meds" ^
    --features "./outputs/causal/data/features" ^
    --tokenized "./outputs/causal/data/tokenized" ^
    --pretrain-model "./outputs/causal/pretrain/model" ^
    --experiment-dir "./outputs/causal/test_local" ^
    --overwrite ^
    ce0p62_cy0p62_y0p2_i0p2

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Simulated resample experiment failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Simulated resample experiment completed successfully
exit /b 0