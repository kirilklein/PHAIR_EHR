# Causal Pipeline - Resampling Variant

This directory contains a **resampling variant** of the causal inference pipeline. Unlike the standard pipeline (in `causal_pipeline/`), this version **samples a different subset of the cohort for each run** and simulates outcomes on that subsample.

## Key Differences from Standard Pipeline

| Feature | Standard Pipeline | Resampling Pipeline |
|---------|------------------|---------------------|
| **Cohort** | Build once, reuse across runs | Sample different subset per run |
| **Simulation** | Simulate once, reuse across runs | Simulate on each new sample per run |
| **Randomness** | Only from model fitting & bootstrapping | Cohort sampling + simulation + fitting + bootstrapping |
| **True Effect** | Fixed across runs | Can vary per run (drawn from distributions) |
| **Aggregation** | Typically averages over outcomes | Keeps outcomes separate, aggregates only over runs |
| **Use Case** | Repeated fits on same population | Population variability + effect variability |

## Workflow Overview

### One-Time Setup: Build Base Cohort

Before running any resampling experiments, you must build a **full base cohort** that will be sampled from:

```bash
cd ../../..  # Go to project root
python -m corebehrt.main_causal.select_cohort_full \
  --config_path experiments/causal_pipeline_resample/base_configs/base_cohort.yaml
```

This creates a cohort at `./outputs/causal/sim_study/base_cohort/pids.pt` containing all eligible patients.

### Resampling Experiment Loop

Each run follows these steps:

1. **Sample**: Randomly sample a configurable percentage (e.g., 50%) of the base cohort
2. **Simulate**: Simulate causal effects on the sampled subset (true effect can vary)
3. **Train**: Fit outcome models (baseline CatBoost and/or BERT)
4. **Calibrate**: Calibrate predicted probabilities
5. **Estimate**: Estimate causal effects with bootstrapping

**The key difference**: Steps 1-2 are repeated for each run with a different random seed, creating variation in both the population and the simulated effects.

```text
Run 1: Sample 50% (seed=43) → Simulate → Train → Calibrate → Estimate
Run 2: Sample 50% (seed=44) → Simulate → Train → Calibrate → Estimate
Run N: Sample 50% (seed=42+N) → Simulate → Train → Calibrate → Estimate
```

## Directory Structure

```bash
causal_pipeline_resample/
├── base_configs/           # Base config templates
│   ├── simulation.yaml     # Modified with sampling parameters
│   ├── base_cohort.yaml    # Config for building base cohort
│   ├── train_baseline.yaml
│   ├── calibrate.yaml
│   ├── estimate.yaml
│   └── ...
├── experiment_configs/     # Experiment definitions (same format as standard pipeline)
│   └── my_experiment.yaml
├── generated_configs/      # Auto-generated configs (per run)
├── python_scripts/
│   ├── generate_configs.py           # Modified to add sampling params & per-run seeds
│   └── analyze_experiment_results.py # Modified to keep outcomes separate
├── bash_scripts/
│   ├── run_experiment.sh             # Modified to always run sampling & simulation
│   └── run_multiple_experiments.sh   # Modified to pass per-run seeds
└── logs/
```

## Usage

### 1. Create Experiment Configs

Experiment configs use the **same format** as the standard pipeline. Create a YAML file in `experiment_configs/`:

```yaml
# experiment_configs/my_experiment.yaml
experiment_name: my_experiment
description: "Testing resampling with specific confounding"

simulation_model:
  num_shared_factors: 10
  influence_scales:
    shared_to_exposure: 0.5
    shared_to_outcome: 0.5

exposure:
  p_base: 0.3
  age_effect: 0.005

outcomes:
  OUTCOME_1:
    run_in_days: 1
    p_base: 0.2
    exposure_effect: 2.0
  OUTCOME_2:
    run_in_days: 1
    p_base: 0.2
    exposure_effect: 1.0
```

### 2. Run Single Experiment

```bash
cd bash_scripts
./run_experiment.sh my_experiment [OPTIONS]
```

**Options:**

- `--base-seed N`: Base seed for sampling (default: 42). Actual seed = base_seed + run_number
- `--sample-fraction F`: Fraction of base cohort to sample (default: 0.5)
- `--base-cohort PATH`: Path to base cohort (default: ./outputs/causal/sim_study/base_cohort)
- `--baseline-only`: Run only baseline (CatBoost) pipeline
- `--bert-only`: Run only BERT pipeline
- `--experiment-dir DIR`: Base directory for results

**Example:**

```bash
./run_experiment.sh my_experiment --run_id run_01 --sample-fraction 0.6
```

### 3. Run Multiple Runs

To run the same experiment multiple times (e.g., 100 runs for stable estimates):

```bash
cd bash_scripts
./run_multiple_experiments.sh --n_runs 100 my_experiment [OPTIONS]
```

**Options:**

- `--n_runs|-n N`: Number of runs (creates run_01, run_02, ..., run_N)
- `--base-seed N`: Base random seed (default: 42)
- `--sample-fraction F`: Fraction to sample per run (default: 0.5)
- Other options same as `run_experiment.sh`

**Example:**

```bash
./run_multiple_experiments.sh --n_runs 100 --sample-fraction 0.5 my_experiment
```

This will run 100 iterations, each with:

- Seed = 42 + run_number (43, 44, ..., 142)
- 50% of base cohort randomly sampled
- Fresh simulation on that sample
- Independent model fitting and estimation

### 4. Analyze Results

The analysis script aggregates results **per outcome** (not averaging over outcomes):

```bash
cd ../python_scripts
python analyze_experiment_results.py \
  --results_dir ../../../outputs/causal/sim_study/runs \
  --output_dir analysis_results \
  --estimator baseline bert
```

**Output structure:**

```bash
analysis_results/
├── baseline/
│   ├── OUTCOME_1/
│   │   ├── bias_plot.png
│   │   ├── relative_bias_plot.png
│   │   ├── coverage_plot.png
│   │   └── ...
│   ├── OUTCOME_2/
│   │   └── ...
├── bert/
│   └── ...
```

Each outcome gets its own directory with separate plots showing:

- Average bias ± std dev across runs
- Relative bias
- Z-scores
- Coverage probability (95% CI)
- Empirical variance

## Configuration Parameters

### Sampling Parameters (in `simulation.yaml`)

```yaml
sampling:
  enabled: true
  fraction: 0.5  # Sample 50% of base cohort
  base_cohort_path: "./outputs/causal/sim_study/base_cohort"

seed: {{SEED}}  # Replaced with base_seed + run_number
```

### Seed Calculation

The seed for each run is calculated as:

```text
seed = base_seed + run_number
```

For example, with `base_seed=42`:

- run_01 → seed=43
- run_02 → seed=44
- run_100 → seed=142

This ensures:

1. Different samples across runs
2. Reproducibility when re-running
3. Independence between runs

## When to Use This Pipeline

**Use the resampling pipeline when:**

- You want to quantify uncertainty due to **population sampling**
- You're studying how causal estimates vary across different subpopulations
- You want the true causal effect to vary per run (sampling from effect distributions)
- You need to assess finite-sample properties of estimators
- You want separate analysis per outcome (no averaging over outcomes)

**Use the standard pipeline when:**

- You want to study uncertainty only from model fitting and bootstrapping
- You need many independent fits on the same fixed population
- You want results aggregated across multiple outcomes
- Computational cost is a major concern (resampling is more expensive)

## Technical Details

### Core Components

1. **`corebehrt/functional/causal/cohort_sampler.py`**: Samples patient IDs from full cohort
2. **`corebehrt/main_causal/simulate_with_sampling.py`**: Wrapper that filters shards to sampled patients before simulation
3. **Modified scripts**: Updated configs, runners, and analysis to support per-run sampling

### Reused Components (No Modifications)

All core causal inference modules are reused as-is:

- `corebehrt.modules.simulation.realistic_simulator`
- `corebehrt.modules.causal.estimate`
- Training, calibration, and preparation modules

## Tips and Best Practices

1. **Sample Size**: Start with `--sample-fraction 0.5` and adjust based on your needs
   - Larger fractions → more stable estimates per run, less population variability
   - Smaller fractions → more population variability, noisier estimates

2. **Number of Runs**:
   - Start with 10-20 runs for quick tests
   - Use 100+ runs for final analyses and publication results
   - More runs → smoother empirical distributions

3. **Base Seed**: Keep the same `base_seed` for reproducibility. Change it only when you want a completely different experiment series.

4. **Outcome Selection**: Since outcomes are analyzed separately, you can specify fewer outcomes per experiment (4-8 instead of 13)

5. **Computational Cost**: Resampling experiments are more expensive than standard experiments because they redo simulation and data prep for each run. Consider:
   - Running fewer outcomes per experiment
   - Using only baseline models for initial exploration
   - Parallelizing runs across machines

## Troubleshooting

### "Base cohort not found"**

- Make sure you ran the one-time base cohort setup (see "One-Time Setup" above)
- Check that `--base-cohort` path is correct

### "No data after sampling"**

- Your `sample_fraction` might be too small
- Check that base cohort has enough patients

### "Analysis produces empty plots"**

- Ensure you have at least 2 runs completed
- Check that `--outcomes` filter isn't excluding all data

### "Seeds seem the same across runs"**

- Verify run IDs follow format `run_01`, `run_02`, etc.
- Check that `--base-seed` is being passed correctly

## Contact & Support

This resampling pipeline is an extension of the standard causal pipeline. For questions:

- See main project README at `../../README.md`
- Standard pipeline README at `../causal_pipeline/README.md`
