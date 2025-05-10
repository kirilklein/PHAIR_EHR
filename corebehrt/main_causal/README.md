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

`finetune_cv.py` - Trains models to estimate exposure probabilities.

```bash
python -m corebehrt.main_causal.finetune_cv
```

**Outputs:**

- Cross-validated propensity scores
- Model predictions and parameters

### 4. Calibration

`calibrate.py` - Ensures propensity scores represent true probabilities.

```bash
python -m corebehrt.main_causal.calibrate
```

**Outputs:**

- Calibrated probability estimates

### 5. Encoding

`encode.py` - Extracts patient representations from the propensity score model.

```bash
python -m corebehrt.main_causal.encode
```

**Outputs:**

- `encodings.pt`: Patient-level vector representations

### 6. Outcome Model Training

Trains models to predict outcomes based on patient encodings.

```bash
python -m corebehrt.main_causal.train_mlp
```

**Configuration:**

```yaml
model_args:
  num_layers: 3
  hidden_dims: [256, 128, 64]

trainer_args:
  batch_size: 128
  epochs: 50
  early_stopping: 5
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
outcome_model:
  type: sigmoid
  params: {a: 1.5, b: 0.5, c: 0.2}

counterfactual:
  generate: true
  method: "inverse probability weighting"
```

**Outputs:**

- `simulated_outcomes.csv`: Patient outcomes with known effects
- `counterfactual_probas.csv`: Counterfactual probabilities

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

- `treatment_effects.csv`: Estimated causal effects
- `bootstrap_results.pt`: Uncertainty estimates (optional)

## Additional Helper Scripts

### Extract Criteria

`extract_criteria.py` - Extracts patient characteristics related to exposure.

```bash
python -m corebehrt.main_causal.extract_criteria
```

**Purpose:**

- Identifies conditions present before exposure
- Creates binary indicators for specific medical codes

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

1. **Optional:** Build tree and generate outcomes config if analyzing multiple outcomes
2. **Train propensity score model:** Use `finetune_cv.py`
3. **Calibrate predictions:** Use `calibrate.py`
4. **Extract encodings:** Use `encode.py`
5. **For robustness testing:** Use `simulate.py` to simulate outcomes with known effects
6. **Train outcome model:** Use `train_mlp.py` on real or simulated data
7. **Estimate effects:** Use `estimate.py` with selected causal methods
8. **Analyze results:** Use `simulate.py` to compare estimated vs. true effects (if simulated)

## Configuration Files

Example configurations are available in the `configs` folder.