# Calibrating Simulation Parameters

Before running a full experiment, use the calibration script to check that your coefficients and intercept produce realistic probability distributions.

## Running the Calibration Script

```bash
python -m corebehrt.main_causal.calibrate_semisynthetic \
  --config_path corebehrt/configs/causal/simulate_semisynthetic.yaml
```

On Azure:

```bash
python -m corebehrt.azure job calibrate_semisynthetic CPU-20-LP \
  --config corebehrt/configs/causal/simulate_semisynthetic.yaml \
  -e calibrate_semisynthetic
```

## What It Reports

### Feature diagnostics (printed table)

Per feature: mean, std, min, p5, p25, p50, p75, p95, max, and **SMD** (standardized mean difference between treated and control groups).

SMD tells you which features are associated with real treatment assignment. Features with |SMD| > 0.1 are meaningfully imbalanced and will create confounding in the simulation.

### Probability diagnostics

For each outcome:
- P(Y(0)) and P(Y(1)): mean, std, min, max, median
- Fraction of extreme probabilities (< 0.01 or > 0.80)
- Expected factual outcome prevalence

### Causal effect diagnostics

- True ATE = mean(P1 - P0)
- True ATT = mean(P1 - P0 | A=1)
- True ATC = mean(P1 - P0 | A=0)
- True RR = mean(P1) / mean(P0)

### Plots (saved to `outcomes/figs/`)

- P(Y(0)) and P(Y(1)) histograms
- ITE distribution per outcome
- SMD love plot (feature balance between treated/control)

## Calibration Workflow

### 1. Start with a target baseline prevalence

Decide the untreated outcome rate:
- ~5% for rare outcomes
- ~10-20% for moderate
- ~30% for common

### 2. Set beta_0 to approximate that prevalence

`sigmoid(beta_0)` is roughly the baseline risk when all features are at their mean (which is 0 after standardization). So:
- beta_0 = -3.0 -> ~5% baseline
- beta_0 = -2.0 -> ~12% baseline
- beta_0 = -1.0 -> ~27% baseline
- beta_0 = 0.0  -> ~50% baseline

### 3. Keep coefficients moderate

With standardized features, a coefficient of:
- 0.1-0.2: small effect
- 0.3-0.5: moderate effect
- 0.8+: large effect (use sparingly)

Multiple large coefficients will push logits to extremes and saturate probabilities near 0 or 1.

### 4. Run calibration and check

Good signs:
- Most P(Y(0)) values between 0.01 and 0.80
- Mean prevalence close to your target
- ATE is detectable but not absurdly large
- Fraction of extreme probabilities < 10%

Bad signs:
- Many probabilities near 0 or 1 (reduce coefficients)
- ATE too small to detect (increase delta)
- ATE too large (every method will succeed trivially)

### 5. Check feature SMDs

If the SMD love plot shows most features near zero, then the features don't create meaningful confounding and the simulation is too easy. Look for features with |SMD| > 0.1 — these are the ones that make the problem realistic.

### 6. Iterate

Adjust beta_0, coefficients, and delta, re-run calibration, until the distributions look sensible.
