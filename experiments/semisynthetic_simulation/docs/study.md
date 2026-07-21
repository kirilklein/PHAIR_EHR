# Running the Study on Azure

The study runs on a **fixed, pre-existing cohort** (a `select_cohort_full` output,
e.g. a diabetes cohort). The cohort's `index_dates.csv` defines the patients and
their index dates; **treatment is the real exposure**. Only the outcome is
simulated, with a single fixed true effect per scenario (`noise_scale = 0`).

```
Fixed shared cohort  →  index_dates.csv + cohort_config.yaml + pids
        │
   per outer run s (own seed):
        simulate outcomes  →  prepare
        →  K refits:  fit → calibrate → B patient bootstraps
                     (BERT and/or CatBoost baseline)
        │
        summarize.py → bias / SD / SE-calibration / coverage, per estimator × outcome
```

- **Outer runs** redraw the simulated outcomes (Monte Carlo over the DGP). 1 is
  enough for "does it recover the effect"; ~10–20 for a coverage sanity check.
- **Inner refits (`-k`)** use different model seeds and reshuffled CV folds,
  yielding K independently fitted propensity-score models.
- **Patient bootstraps (`-b`, default 100)** resample the original analysis
  cohort with replacement conditional on each fitted model. The summarizer
  combines the K × B draws into one SE and CI per outer simulation replicate.

## What to set (once)

Edit `job_config_template.yaml` — the datastore paths for `meds`, `features`,
`tokenized`, `pretrain_model`, and **`cohort`** (your diabetes cohort dir). Leave
`results` ending in `__RUN__`.

In `base_configs/simulate.yaml`, set **`exposure_code`** to whatever marks a
treated patient in your MEDS, and check **`splits`** covers your cohort's patients.
Effect sizes are `OUTCOME_NULL` (δ=0), `OUTCOME_MEDIUM` (δ=0.5), `OUTCOME_LARGE` (δ=1.0).

## What to run

### 1. Phase 1 — first shot (does it recover the effect?)

One run, single fit, method + baseline:

```bash
./experiments/semisynthetic_simulation/bash_scripts/submit_runs.sh
```

Then summarize and check θ̂ vs θ* and CI coverage:

```bash
python -m experiments.semisynthetic_simulation.python_scripts.summarize \
  --study-dir <results>/run_01
```

### 2. Full run

A few outer runs, K seeded refits and B patient bootstraps each:

```bash
./experiments/semisynthetic_simulation/bash_scripts/submit_runs.sh -n 5 -k 10 -b 100
```
(rename `smoketest`→`full` in the template's `results` first). Then
`summarize.py --study-dir <results>` over all runs for the full table.

### Flags

| flag | meaning | default |
|------|---------|---------|
| `-n` | outer runs (parallel jobs) | 1 |
| `-k` | independently seeded model refits per run | 1 |
| `-b` | patient bootstrap samples per refit | 100 |
| `--baseline-only` / `--bert-only` | restrict to one model | both |

Override compute/experiment via env: `POOL=<gpu> EXPERIMENT=my_exp ./submit_runs.sh ...`
(BERT fits want a GPU pool; the CatBoost baseline runs on CPU.)

## Outputs (per outer run)

```
<results>/run_NN/
├── simulated_outcomes/   exposure.csv, OUTCOME_*.csv, counterfactuals.csv, stats, figs
├── prepared_data/        tokenized finetune data (+ folds)
├── _configs/             exact per-step configs used
└── reshuffles/k_NN/
    ├── models/{bert,baseline}/...
    └── estimate/{bert,baseline}/
        ├── estimate_results.csv
        └── bootstrap_results.csv   ← B patient-level bootstrap estimates
```

Running `summarize.py` writes `replicate_estimates.csv` (one K × B aggregate
per outer run) and `summary.csv` (bias, empirical SD, mean SE, SE calibration,
and coverage across outer runs). Common support is trimmed at the 0.1st and
99.9th percentiles within treatment arms before patient resampling.
