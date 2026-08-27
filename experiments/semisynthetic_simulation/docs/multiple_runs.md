# Running Multiple Replicates

The simulation is deterministic given a seed. To generate S independent replicates (e.g., for computing bias, coverage, SE calibration), run with different seeds and output directories.

## Using the Config Generator

```bash
python experiments/semisynthetic_simulation/generate_configs.py \
  my_scenario \
  --n_runs 50 \
  --base_seed 42 \
  --experiments_dir ./outputs/causal/semisynthetic_study/runs \
  --meds ./path/to/meds/data
```

This generates one config per run under `generated_configs/`:

```
generated_configs/
├── my_scenario_run_01.yaml    # seed=43, outcomes → runs/run_01/my_scenario/
├── my_scenario_run_02.yaml    # seed=44, outcomes → runs/run_02/my_scenario/
├── ...
└── my_scenario_run_50.yaml    # seed=92, outcomes → runs/run_50/my_scenario/
```

Each config is a copy of the base config with:
- `seed` set to `base_seed + run_number`
- `paths.outcomes` set to a unique per-run directory

## Running All Replicates

### Locally (sequential)

```bash
for cfg in generated_configs/my_scenario_run_*.yaml; do
  python -m corebehrt.main_causal.simulate_semisynthetic --config_path "$cfg"
done
```

### On Azure (parallel)

Submit each run as a separate Azure job:

```bash
for cfg in generated_configs/my_scenario_run_*.yaml; do
  python -m corebehrt.azure job simulate_semisynthetic CPU-20-LP \
    --config "$cfg" -e semisynthetic_study
done
```

## Output Structure

```
outputs/causal/semisynthetic_study/runs/
├── run_01/my_scenario/
│   ├── counterfactuals.csv
│   ├── ite.csv
│   └── ...
├── run_02/my_scenario/
│   └── ...
└── run_50/my_scenario/
    └── ...
```

## Evaluation Metrics

Across S runs, compute:
- **Bias**: mean(theta_hat) - theta_true
- **Empirical SD**: std(theta_hat)
- **Mean estimated SE**: mean(SE_hat)
- **SE calibration**: SD_emp / mean(SE_hat) (target: 1)
- **Coverage**: fraction of runs where 95% CI contains theta_true

The true effect theta_true is computed from the known P(Y(0)) and P(Y(1)) in each run's `counterfactuals.csv`.

## Scenario Grid

A typical experiment varies 3 dimensions:

| Dimension | Levels | Config parameter |
|-----------|--------|-----------------|
| Baseline prevalence | low (~5%), moderate (~15%), high (~30%) | `beta_0` |
| Outcome complexity | simple (few features), rich (all features + interactions) | `coefficients`, `interactions` |
| Treatment effect | null (delta=0), non-null (delta=1) | `treatment_effect.delta` |

Create one experiment config per scenario (e.g., `low_simple_null.yaml`, `low_rich_nonnull.yaml`), then run each scenario for S replicates.
