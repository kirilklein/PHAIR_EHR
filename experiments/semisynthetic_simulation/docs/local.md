# Running Locally

## Single Run

```bash
python -m corebehrt.main_causal.simulate_semisynthetic \
  --config_path corebehrt/configs/causal/simulate_semisynthetic.yaml
```

This reads the MEDS data, extracts real treatment assignments, computes oracle features, simulates outcomes, and saves results to the configured `paths.outcomes` directory.

## Output Files

```
outputs/causal/semisynthetic_outcomes/
├── counterfactuals.csv          # subject_id, exposure, outcome_X, Y0_X, Y1_X, P0_X, P1_X
├── ite.csv                      # subject_id, ite_X (individual treatment effects)
├── OUTCOME.csv                  # Event records for patients with Y_obs=1
├── index_date_matching.csv      # Exposed/control matching
├── simulation_stats.csv         # Patient counts, exposure rates, outcome prevalences
├── theoretical_max_roc_auc.csv  # Best achievable AUC from the DGP
└── figs/
    ├── probability_distributions.png
    └── true_effects_vs_risk_differences.png
```

## Config Structure

```yaml
paths:
  data: ./example_data/synthea_meds_causal    # MEDS shard directory
  splits: ["tuning"]                           # Which splits to use
  outcomes: ./outputs/causal/semisynthetic_outcomes

seed: 42
min_num_codes: 3
exposure_code: "EXPOSURE"       # Code string identifying treatment events

features:
  code_prefixes:
    diagnosis: "D/"
    medication: "M/"
    procedure: "P/"
    admission: "ADM/"
  lookback_days: 365            # General lookback window
  recent_window_days: 90        # "Recent" events window
  burst_window_days: 30         # Short-term burst window
  motif_window_days: 30         # Max gap for sequential motifs
  standardize: true             # Z-score features before applying coefficients

outcomes:
  OUTCOME:                      # One block per simulated outcome
    outcome_model:
      run_in_days: 1
      beta_0: -2.0              # Intercept (controls baseline prevalence)
      coefficients:             # Feature name -> coefficient (logit scale)
        disease_burden: 0.2
        age: 0.4
        event_recency: -0.15
      interactions:
        - features: [disease_burden, age]
          coefficient: 0.1
      noise_scale: 0.1          # Logit-scale noise std dev
    treatment_effect:
      mode: constant            # "constant" or "heterogeneous"
      delta: 1.0                # Constant treatment effect (logit scale)
```

## MEDS Data Requirements

The simulator expects MEDS-format parquet files with columns:
- `subject_id`: Patient identifier
- `time`: Event timestamp (datetime)
- `code`: Medical code string (e.g., `D/12345`, `M/67890`, `DOB`, `EXPOSURE`)
- `assigned_index_date`: Per-patient index date (datetime, can be NaT for patients to exclude)

Patients with `EXPOSURE` events are treated (A=1). Others are control (A=0).
