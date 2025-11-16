# Two-Stage Batch Experiments Runner - Usage Guide

## Overview

The batch experiments runner now supports a two-stage workflow:

**Stage 1 (Once per outer run):**
- Sample patients from MEDS
- Simulate outcomes
- Prepare finetune data

**Stage 2 (K times per outer run):**
- Reshuffle folds (random seed)
- Finetune/train model
- Calibrate predictions
- Estimate effects

This allows you to estimate variance within each outer run (K estimates) and assess hitting the true effect across outer runs.

## Directory Structure

```
outputs/causal/experiments/
└── outer_run_01/
    └── experiment_name/
        ├── sampled_pids.pt              # Stage 1
        ├── simulated_outcomes/          # Stage 1
        ├── cohort/                      # Stage 1
        ├── prepared_data/               # Stage 1 (base folds)
        │   ├── patients.pt
        │   ├── folds.pt
        │   └── ...
        └── reshuffles/                  # Stage 2
            ├── k_01/
            │   ├── models/baseline/
            │   ├── models/bert/
            │   └── estimate/
            ├── k_02/
            │   └── ...
            └── k_05/
                └── ...
```

## Local Testing

### Quick Test (1 inner run)

```bash
python -m experiments.causal_pipeline_resample.python_scripts.run_multiple_experiments \
    --n_runs 1 \
    --inner_runs 1 \
    --baseline-only \
    --sample-fraction 0.01 \
    --meds "./example_data/synthea_meds_causal" \
    --features "./outputs/causal/data/features" \
    --tokenized "./outputs/causal/data/tokenized" \
    --pretrain-model "./outputs/causal/pretrain/model" \
    --experiment-dir "./outputs/causal/test_local" \
    --base-configs-dir "./experiments/causal_pipeline_resample/base_configs" \
    ce0p62_cy0p62_y0p2_i0p2
```

### Full Test (5 inner runs)

```bash
python -m experiments.causal_pipeline_resample.python_scripts.run_multiple_experiments \
    --n_runs 1 \
    --inner_runs 5 \
    --baseline-only \
    --sample-fraction 0.05 \
    --meds "./example_data/synthea_meds_causal" \
    --features "./outputs/causal/data/features" \
    --tokenized "./outputs/causal/data/tokenized" \
    --pretrain-model "./outputs/causal/pretrain/model" \
    --experiment-dir "./outputs/causal/test_local" \
    --base-configs-dir "./experiments/causal_pipeline_resample/base_configs" \
    ce0p62_cy0p62_y0p2_i0p2
```

## Azure Job Submission

### 1. Create Azure Config

`experiments/azure_configs/trace/batch_experiments/my_run.yaml`:

```yaml
# Experiment names to run
experiments:
  - ce0p62_cy0p62_y0p2_i0p2
  - ce0p62_cy0p62_y0p62_i0p62

paths:
  meds: "researcher_data:AKK/shared/MEDS/TRACE/v01/data"
  features: "researcher_data:AKK/shared/features/trace/v01/features"
  tokenized: "researcher_data:AKK/shared/features/trace/v01/tokenized"
  pretrain_model: "researcher_data:AKK/shared/pretrain/models/trace/small/len_512/v01"
  results: "researcher_data:AKK/shared/experiments/batch_results/run_01"
```

### 2. Submit Azure Job

```bash
python -m corebehrt.azure job run_batch_experiments CPU-20-LP \
    -e "batch_run_outer_01" \
    -c experiments/azure_configs/trace/batch_experiments/my_run.yaml \
    --bash-args "--inner_runs 5 --baseline-only --sample-fraction 0.1 --timeout-factor 10 --failfast --base-configs-dir ./experiments/causal_pipeline_resample/base_configs_azure"
```

### 3. Submit Multiple Outer Runs

For N=20 outer runs with K=5 inner runs each:

```bash
# Outer run 1
python -m corebehrt.azure job run_batch_experiments CPU-20-LP \
    -e "batch_outer_run_01" \
    -c experiments/azure_configs/trace/batch_experiments/run_01.yaml \
    --bash-args "--inner_runs 5 --baseline-only --sample-fraction 0.1"

# Outer run 2
python -m corebehrt.azure job run_batch_experiments CPU-20-LP \
    -e "batch_outer_run_02" \
    -c experiments/azure_configs/trace/batch_experiments/run_02.yaml \
    --bash-args "--inner_runs 5 --baseline-only --sample-fraction 0.1"

# ... repeat for runs 3-20
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--inner_runs`, `-k` | Number of inner reshuffles per outer run | 1 |
| `--n_runs`, `-n` | Number of outer runs (for local batch) | 1 |
| `--baseline-only` | Run only CatBoost baseline | Both |
| `--bert-only` | Run only BERT pipeline | Both |
| `--sample-fraction F` | Fraction of patients to sample | Required |
| `--sample-size N` | Absolute number of patients | Alternative to fraction |
| `--base-seed N` | Base seed for sampling | 42 |
| `--overwrite` | Force re-run all steps | Skip completed |
| `--failfast` | Stop on first failure | Continue on failure |

## Reshuffle Behavior

- **Stage 1** uses `base_seed + outer_run_number` for sampling
- **Stage 2** uses automatic time-based seeds for reshuffling (different each inner run)
- Each inner run gets independent fold splits for variance estimation

## Output Analysis

After completion, you'll have:

- **K estimates per outer run** in `reshuffles/k_01/` through `reshuffles/k_05/`
- Compute mean & SD across K estimates within each outer run
- Compare estimates across N outer runs to assess hitting true effect

## Example: 20 Outer × 5 Inner = 100 Total Estimates

```
Outer Run 1: 5 estimates → mean ± SD
Outer Run 2: 5 estimates → mean ± SD
...
Outer Run 20: 5 estimates → mean ± SD

Analysis:
- Within-run variance: SD from 5 inner runs
- Across-run variance: Distribution of 20 means
- Coverage: How many of 20 means capture true effect?
```

## Troubleshooting

### Issue: `folds.pt` not found

**Solution:** The prepare_finetune step now checks for `folds.pt` specifically. If you have old data with only `patients.pt`, use `--overwrite` to regenerate.

### Issue: Config not found for inner runs

**Solution:** Inner configs are auto-generated. Ensure base configs exist in `base_configs/` or specify custom path with `--base-configs-dir`.

### Issue: Out of memory with many inner runs

**Solution:** Run baseline and BERT separately using `--baseline-only` or `--bert-only`.

