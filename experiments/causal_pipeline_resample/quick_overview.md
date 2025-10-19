# Causal Pipeline Overview (Resampling Variant)

## Pipeline Structure

Each experimental run follows this workflow:

### 1. Data Generation & Sampling

- Sample a subset of patients (e.g., 10%) from raw MEDS data using run-specific seed
- Simulate realistic exposure assignment and multiple outcome scenarios with known ground-truth causal effects
- Generate counterfactual outcomes for all patients under both exposure conditions

### 2. Cohort Selection

- Apply temporal eligibility criteria (minimum lookback, follow-up periods, index date ranges)
- Filter patients based on inclusion/exclusion criteria (age, comorbidities, prior medications)
- Create matched control cohort by drawing index dates from exposed patients (with age adjustment)

### 3. Data Preparation

- Split cohort into cross-validation folds (e.g., 5-fold)
- Prepare tokenized sequences for BERT and tabular features for baseline models
- Maintain fold structure for consistent train/validation splits

### 4. Model Training (Baseline & BERT)

- Train propensity score models (exposure prediction) and outcome models using k-fold CV
- **Baseline**: CatBoost with hyperparameter tuning via Optuna
- **BERT**: Fine-tuned transformer on patient event sequences
- Generate out-of-fold predictions ensuring all patients have predictions from models not trained on their data

### 5. Calibration

- Calibrate predictions using isotonic regression on validation folds
- Ensures predicted probabilities match empirical event rates
- Critical for accurate causal effect estimation (especially TMLE)

### 6. Causal Effect Estimation

- Use calibrated predictions for propensity scores (exposure) and outcome probabilities
- Apply multiple estimators: IPW (Inverse Propensity Weighting), G-computation, and TMLE (Targeted Maximum Likelihood Estimation)
- Estimate standard errors via bootstrap resampling and theoretical variance formulas
- Calculate treatment effects: Risk Difference (RD), Risk Ratio (RR), and confidence intervals

### 7. Multi-Run Replication

- Repeat steps 1-6 for N independent runs with different random seeds
- Each run samples a new cohort subset, simulates new outcomes, and trains new models
- Ensures statistical independence across runs

### 8. Performance Aggregation

- Calculate bias (mean deviation from ground-truth effect across runs)
- Estimate variance (spread of effect estimates across runs)
- Compute coverage (proportion of CIs containing true effect)
- Calculate z-scores and MSE for systematic evaluation of estimator performance

## Key Features

- **Known Ground Truth**: Simulated effects enable direct bias/variance calculation
- **Realistic Confounding**: Simulation preserves EHR structure with configurable confounding strength
- **Model Comparison**: Parallel evaluation of baseline (CatBoost) vs. deep learning (BERT) approaches
- **Resampling Independence**: Each run samples fresh patients ensuring true statistical replication
- **Robust Uncertainty**: Combines cross-validation, calibration, and bootstrap for reliable inference
