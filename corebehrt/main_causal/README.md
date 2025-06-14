# Transformer-based Pipeline for doubly-robust causal estimates

## Overview

This pipeline enables rigorous causal inference analysis for patient outcomes by combining transformer-based representation learning and probability estimation with established causal inference methodologies.

### Core Workflow

1. **Select cohort** to select exposed/controls based on incl./excl. criteria
2. **Prepare data** to create censored and truncated sequences and binary targets
3. **Finetune** to model both exposure+outcome and make counterfactual estimates
4. **Calibrate** to calibrate exposure and outcome probas
5. **Estimate causal effects** using methods like IPW, AIPW, and TMLE

## Main Pipeline Components

First, run the usual pipeline in main up to select data: create_data, prepare_training_data, and pretrain.

### 1. Select Cohort Full

`select_cohort_full.py` - Identify exposed/control, draw index dates, apply criteria, and save cohort.

### 2. Prepare data

`prepare_ft_exp_y.py` - Prepare data for finetuning on exposure and outcomes. Creates patients with two targets.

### 3. Finetune exposure+target

`finetune_exp_y.py` - Train a model on exposure and outcome. For outcome, produce counterfactual estimates as well.

### 4. Calibrate exposure+target probas

`calibrate_exp_y.py` - Calibrate probabilities.

### 5. Causal Effect Estimation

`estimate.py` - Implements various causal inference methods to estimate treatment effects

## Additional Scripts

### Extract Criteria

`extract_criteria.py` - Given index_dates (run select cohort full first), extract criteria from the data. We can also define additional criteria during cohort selection which are not applied but stored for use with `get_stats.py`

```bash
python -m corebehrt.main_causal.extract_criteria
```

**Purpose:**

- Identifies conditions present in a time window before or after exposure
- Creates binary indicators for specific medical codes/combinations of medical codes and criteria

### Get stats

`helper_scripts/get_stats.py` - Get stats from criteria table produced e.g. with select_cohort or extract_criteria. Optionally use propensity scores to get weighted stats as well.

**Purpose:**

- Produces tables comparing exposed vs. unexposed groups
- Calculates weighted statistics using propensity scores
- Evaluates balance between treatment groups

## Detailed Pipeline Steps

### 1. Cohort Selection

`select_cohort_full.py` - Identifies exposed and control patients, draws index dates, and applies selection criteria.

```bash
python -m corebehrt.main_causal.select_cohort_full
```

**Purpose:**

- Identifies treatment and control groups
- Applies inclusion/exclusion criteria
- Creates a balanced cohort for analysis

### 2. Data Preparation

`prepare_ft_exp_y.py` - Prepares data for joint exposure and outcome modeling.

```bash
python -m corebehrt.main_causal.prepare_ft_exp_y
```

**Purpose:**

- Creates patient records with both exposure and outcome targets
- Prepares data for joint modeling approach

### 3. Joint Exposure-Outcome Modeling

`finetune_exp_y.py` - Trains a model to jointly predict exposure and outcomes.

```bash
python -m corebehrt.main_causal.finetune_exp_y
```

**Purpose:**

- Trains a single model for both exposure and outcome prediction
- Generates counterfactual outcome estimates
- Produces patient representations for causal inference

### 4. Probability Calibration

`calibrate_exp_y.py` - Calibrates the predicted probabilities from the joint model.

```bash
python -m corebehrt.main_causal.calibrate_exp_y
```

**Purpose:**

- Ensures predicted probabilities are well-calibrated
- Improves reliability of propensity scores and outcome predictions

### 5. Treatment Effect Estimation

`estimate.py` - Implements various causal inference methods to estimate treatment effects.

```bash
python -m corebehrt.main_causal.estimate
```

**Methods:**

- Inverse Probability Weighting (IPW)
- Augmented Inverse Probability Weighting (AIPW)
- Targeted Maximum Likelihood Estimation (TMLE)
- Matching-based methods

**Outputs:**

- `estimate_results.csv`: Estimated causal effects
- `experiment_stats.csv`: Statistics of the experiment

## Usage Workflow

1. **Select Cohort:** Use `select_cohort_full.py`
2. **Prepare Finetune Exposure+Outcome:** Use `prepare_ft_exp_y.py`
3. **Finetune Exposure + Outocme:** Use `finetune_exp_y.py`
4. **Calibrate:** Use `calibrate_exp_y.py`
5. **Estimate effects:** Use `estimate.py` with selected causal methods

## Configuration Files

Example configurations are available in the `corebehrt/configs/causal` folder
