# Causal Inference Pipeline for Patient Outcomes

## Overview

This pipeline enables rigorous causal inference analysis for patient outcomes by combining machine learning with established causal inference methodologies. It estimates treatment effects using patient encodings, outcome simulations, and various causal inference techniques.

### Core Workflow

1. **Train propensity score models** to estimate exposure probabilities
2. **Calibrate predictions** to ensure proper probability estimation
3. **Extract patient encodings** from the propensity score model as confounders
4. **Train outcome models** on these encodings to predict outcomes
5. **Estimate causal effects** using methods like IPW, AIPW, and TMLE
6. **Optional: Simulate outcomes** with known causal effects to test robustness

## Main Pipeline Components

### 0. Select Cohort Advanced (Optional)

`select_cohort_advanced.py` - Selects a cohort of patients based on advanced criteria. Is run in addition to [select_cohort.py](../main/select_cohort.py) to create a CONSORT diagram. See module description in [select_cohort_advanced.py](./select_cohort_advanced.py) for more details.

### 1. Propensity Score Estimation

`finetune_cv.py` - Trains models to estimate propensity scores using cross-validation

### 2. Probability Calibration

`calibrate.py` - Calibrates model predictions to produce reliable probability estimates

### 3. Patient Encoding

`encode.py` - Extracts patient-level vector representations (encodings) from the propensity score model

### 4. Outcome Modeling

`train_mlp.py` - Trains outcome prediction models using patient encodings and makes counterfactual predictions

### 5. Causal Effect Estimation

`estimate.py` - Implements various causal inference methods to estimate treatment effects

### 6. Outcome Simulation (Optional)

`simulate.py` - Used for robustness testing - simulates outcomes with known causal effects to validate estimation methods

## Detailed Pipeline Steps

### 1. Build Tree (Helper)

`helper_scripts/build_tree.py` - Organizes medical codes into a hierarchical structure for analysis.

```bash
python -m corebehrt.main_causal.helper_scripts.build_tree --type [diagnoses|medications] --level [INT]
```

**Parameters:**

- `--type`: Type of data (diagnoses or medications)
- `--level`: Hierarchical level for tree construction

**Outputs:**

- Tree dictionary at `./outputs/trees/[type]_tree_level_[level].pkl`

### 2. Generate Outcomes Config (Helper)

`helper_scripts/generate_outcomes_config.py` - Creates outcome configuration files from tree dictionaries.

```bash
python -m corebehrt.main_causal.helper_scripts.generate_outcomes_config \
    --input ./outputs/trees/[type]_tree_level_[level].pkl \
    --output ./outputs/causal/outcomes/generated_outcomes.yaml
```

**Parameters:**

- `--input`: Path to tree dictionary
- `--output`: Path for saving the config
- `--match_how`: Code matching method (startswith, contains, exact)
- `--prepend`: String to prepend to outcome names

**Outputs:**

- YAML configuration file with outcome definitions

### 3. Propensity Score Model Training

`finetune_cv.py` - Trains models to estimate exposure probabilities. See main [README.md](../README.md) for more details.

### 4. Calibration

`calibrate.py` - Ensures propensity scores represent true probabilities. Uses beta calibration.

```bash
python -m corebehrt.main_causal.calibrate
```

**Configuration:**

```yaml
paths:
## INPUTS  
  finetune_model: ./path/to/finetune_model

## OUTPUTS
  calibrated_predictions: ./path/to/save/calibrated_predictions
```

**Outputs:**

- `predictions_and_targets.csv`: Propensity scores and exposure (for completeness)
- `predictions_and_targets_calibrated.csv`: Calibrated probability estimates

### 5. Encoding

`encode.py` - Uses the finetuned model to encode the patient data.

```bash
python -m corebehrt.main_causal.encode
```

**Configuration:**

```yaml
paths:
## INPUTS  
  finetune_model: ./path/to/finetune_model
  prepared_data: ./path/to/prepared_data # same as used in finetune_cv

## OUTPUTS
  encoded_data: path/to/save/encoded_data
```

**Outputs:**

- `encodings.pt`: Patient-level vector representations
- `pids.pt`: Patient IDs

### 6. Outcome Model Training

Trains models to predict outcomes based on patient encodings.

```bash
python -m corebehrt.main_causal.train_mlp
```

**Configuration:**

```yaml
paths:
  ## INPUTS  
  encoded_data: ./path/to/encoded_data
  calibrated_predictions: ./path/to/calibrated_predictions # to extract exposure
  cohort: ./path/to/cohort

  outcomes: ./path/to/outcomes
  outcome: <Outcome_name>.csv

  ## OUTPUTS
  trained_mlp: ./path/to/trained_mlp

outcome:
  n_hours_start_follow_up: 1 # start follow up (considering outcomes) time after index date (negative means before)
  n_hours_end_follow_up: null # end follow up (considering outcomes) time after index date (negative means before)
# + model and trainer args
```

**Outputs:**

- `mlp_probas.pt`: Predicted outcome probabilities
- `mlp_predictions.pt`: Binary predictions
- Counterfactual predictions

### 7. Outcome Simulation (Optional)

`simulate.py` - Simulates outcomes with known causal effects for validation.

```bash
python -m corebehrt.main_causal.simulate
```

**Configuration:**

```yaml
simulation:
  exposure_coef: 4 # exposure coefficient. Determines the strength of the treatment effect
  enc_coef: .0001 # treatment patient embeddings coefficient. Determines the confounding effect of the treatment
  intercept: -2 # intercept, determines the baseline outcome level
  enc_sparsity: 0.7 # proportion of treatment patient embeddings that will have non-zero coefficients, in order to simulate that only some treatment features are associated with the outcome
  enc_scale: 0.00011 # scale of the normal distribution for treatment patient embeddings coefficients
```

**Outputs:**

- `outcome_with_timestamps.csv`: Patient outcomes with known effects. Same output as from `create_outcomes.py`
- `probas_and_outcomes.csv`: Counterfactual probabilities and outcomes

### 8. Treatment Effect Estimation

Implements causal inference methods to estimate effects.

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

## Additional Helper Scripts

### Extract Criteria

`extract_criteria.py` - Extracts patient characteristics related to exposure.

```bash
python -m corebehrt.main_causal.extract_criteria
```

**Purpose:**

- Identifies conditions present in a time window before or after exposure
- Creates binary indicators for specific medical codes/combinations of medical codes and criteria

### Get Statistics

`get_stats.py` - Generates statistical summaries of patient cohorts.

```bash
python -m corebehrt.main_causal.get_stats
```

**Purpose:**

- Produces tables comparing exposed vs. unexposed groups
- Calculates weighted statistics using propensity scores
- Evaluates balance between treatment groups

## Usage Workflow

1. **Optional:** Build tree and generate outcomes config if analyzing many outcomes
2. **Train propensity score model:** Use `finetune_cv.py`
3. **Calibrate predictions:** Use `calibrate.py`
4. **Extract encodings:** Use `encode.py`
5. **For robustness testing:** Use `simulate.py` to simulate outcomes with known effects
6. **Train outcome model:** Use `train_mlp.py` on real or simulated data
7. **Estimate effects:** Use `estimate.py` with selected causal methods
8. **Analyze results:** Use `simulate.py` to compare estimated vs. true effects (if simulated)

## Configuration Files

Example configurations are available in the `corebehrt/configs/causal` folder
