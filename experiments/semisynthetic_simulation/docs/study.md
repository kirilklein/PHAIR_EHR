# Running the Multi-Run Study on Azure

A "study" runs **N independent outer runs** (each = one semi-synthetic simulation
with its own seed) and, within each, **K inner reshuffle fits** (re-drawn CV folds
for variance estimation):

```
for each outer run (run_01 … run_NN):          # one parallel Azure job each
    simulate → select_cohort → prepare          # Stage 1, once
    for k in 1 … K:                             # Stage 2, K times
        finetune → calibrate → estimate         # folds reshuffled each time
```

Each Azure job is **one outer run**; `submit_runs.sh` submits the N jobs.

## What to set (once)

Edit `experiments/semisynthetic_simulation/job_config_template.yaml` — the datastore
paths for `meds`, `features`, `tokenized`, `pretrain_model`. Leave `results` ending in
`__RUN__` (the submit script fills in `run_01`, `run_02`, …).

The simulated outcomes (and effect sizes) live in
`experiments/semisynthetic_simulation/base_configs/simulate.yaml`:
- `OUTCOME_NULL`  → `delta: 0.0` (no effect)
- `OUTCOME_MEDIUM` → `delta: 0.5` (~6pp risk difference at a ~12% baseline)
- For the full study add `OUTCOME_LARGE` with `delta: 1.0`, and add
  `OUTCOME_LARGE.csv` to `outcome_files` in `base_configs/prepare.yaml`.

## What to run

### 1. Smoke test (do this first)

Two outer runs, two inner fits each, 10% of patients sampled:

```bash
./experiments/semisynthetic_simulation/bash_scripts/submit_runs.sh -n 2 -k 2 -f 0.1
```

Check one job's `estimate/bert/estimate_results.csv` against the true effects in
`simulated_outcomes/simulation_stats.csv` (NULL ≈ 0, MEDIUM ≈ 0.06 risk difference).

### 2. Full study (after the smoke test looks right)

Ten outer runs, ten inner fits each, full population (no `-f`):

```bash
./experiments/semisynthetic_simulation/bash_scripts/submit_runs.sh -n 10 -k 10
```

### Flags

| flag | meaning | default |
|------|---------|---------|
| `-n` | number of outer runs (parallel jobs) | 1 |
| `-k` | inner reshuffle fits per run | 2 |
| `-f` | sample this fraction of patients (smoke tests) | none (full) |

Override the compute/experiment via env: `POOL=CPU-20-LP EXPERIMENT=my_exp ./submit_runs.sh ...`

## What a single job runs (equivalent manual command)

```bash
python -m corebehrt.azure job run_semisynthetic_study CPU-20-LP \
  -e semisynthetic_study \
  -c experiments/semisynthetic_simulation/generated_job_configs/run_01.yaml \
  --bash-args "--run-id run_01 --inner-runs 2 --sample-fraction 0.1"
```

## Outputs (per outer run)

```
<results>/run_NN/
├── simulated_outcomes/        # exposure.csv, OUTCOME_*.csv, counterfactuals.csv, ite.csv, stats, figs
├── cohort/                    # selected cohort + folds
├── prepared_data/             # tokenized finetune data
├── _configs/                  # exact per-step configs used (for reproducibility)
└── reshuffles/
    ├── k_01/{models/bert, estimate/bert}/
    └── k_02/...
```

Seeds are independent across runs (`seed = base_seed + run_number`), so the N outer
runs are genuinely independent replicates.
