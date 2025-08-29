# IPW-Favorable Simulation

This directory contains configurations for running a simulation scenario designed to favor Inverse Probability Weighting (IPW) over Targeted Maximum Likelihood Estimation (TMLE).

## Background

In the original simulation (`simulate_realistic.yaml`), TMLE consistently outperformed IPW in terms of bias and accuracy. This new simulation (`simulate_ipw_favorable.yaml`) creates conditions where IPW should perform better by addressing the key issue of **overadjustment**.

### The Overadjustment Problem

When the propensity score model includes instrumental variables (features that affect treatment but not outcome), it leads to overadjustment bias. The PS model conditions on variables it shouldn't, actually introducing bias rather than removing it.

### Solution: Pure Confounding Structure

This simulation creates a scenario where **all observable features are true confounders** (affecting both exposure and outcome):

1. **No Instrumental Variables** (0 exposure-only factors): Prevents overadjustment in propensity score modeling
2. **No Outcome-Only Predictors** (0 outcome-only factors): Simplifies outcome modeling, making it comparable to PS modeling
3. **All Confounders** (15 shared factors): Every observable feature is a legitimate target for PS adjustment
4. **Balanced Influence**: Both exposure and outcome are well-predicted by the same set of confounders

## Key Parameter Changes

| Parameter | Original | IPW-Favorable (v2) | Rationale |
|-----------|----------|-------------------|-----------|
| `num_shared_factors` | 10 | 15 | More confounders to adjust for |
| `num_exposure_only_factors` | 10 | **0** | **Eliminate instruments - prevent overadjustment** |
| `num_outcome_only_factors` | 10 | **0** | **Eliminate outcome-only predictors** |
| `factor_mapping.scale` | 1.2 | 1.0 | Moderate, predictable effects |
| `factor_mapping.sparsity_factor` | 0.98 | 0.90 | Each code relates to more confounders |
| `shared_to_exposure` | 2.0 | 2.5 | Strong but balanced PS signal |
| `shared_to_outcome` | 2.0 | 2.0 | Balanced outcome prediction |
| `exposure_only_to_exposure` | 0.4 | **0.0** | **No instrument effects** |
| `outcome_only_to_outcome` | 0.4 | **0.0** | **No outcome-only effects** |
| `logit_noise_scale` | 0 | 0.1 | Minimal noise |

## Files Structure

- `simulate_ipw_favorable.yaml` - Main simulation configuration
- `select_cohort_full/extract_simulated_ipw.yaml` - Cohort extraction config
- `finetune/prepare/simulated_ipw.yaml` - Data preparation config
- `finetune/simulated_ipw_bl.yaml` - Model training config
- `finetune/calibrate_simulated_ipw_bl.yaml` - Model calibration config
- `estimate_simulated_ipw_bl.yaml` - Effect estimation config

## Running the Simulation

Use the provided batch script:

```bash
tests/windows/run_causal_simulated/train_baseline_ipw.bat
```

This runs the complete pipeline:

1. Simulate outcomes with IPW-favorable parameters
2. Extract cohort
3. Prepare training data
4. Train baseline models
5. Calibrate predictions
6. Estimate causal effects
7. Run validation tests

## Output Comparison

Results are saved to `./outputs/causal/estimate/baseline/simulated_ipw/`

Use the comparison script to analyze differences:

```python
python experiments/effects/visualize/compare_ipw_tmle_simulations.py
```

This will generate visualizations showing how IPW vs TMLE performance differs between the two simulation scenarios.

## Expected Results

In this IPW-favorable simulation, you should observe:

- **IPW estimates much closer to true effects** because it's adjusting for the correct variables (only confounders)
- **No overadjustment bias** since there are no instrumental variables in the feature set
- **TMLE may still perform well** but without the advantage it had when outcome modeling was easier than PS modeling
- Demonstration that **variable selection matters more than method choice** for propensity score methods

### Key Insight

This simulation demonstrates that the **type of variables included** in your feature set is crucial:

- Including instruments → IPW overadjustment bias
- Only confounders → IPW performs optimally
- This highlights the importance of understanding your data's causal structure when choosing between causal inference methods.
