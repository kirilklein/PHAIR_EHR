# Pipeline: Causal Inference for Patient Outcomes

! **This pipeline is under development and this is a template for the pipeline. Not all steps are available yet.**

This guide walks through the steps required to **estimate treatment effects** using patient encodings, outcome simulation, and causal inference techniques. The pipeline consists of:

1. [**Build Tree**](#1-build-tree)
2. [**Generate Outcomes Config**](#2-generate-outcomes-config)
3. [**Encode**](#3-encode)
4. [**Simulate Outcome**](#4-simulate-outcome)
5. [**Train MLP (on encodings)**](#5-train-mlp-on-encodings)
6. [**Estimate Treatment Effects**](#6-estimate-treatment-effects)

---

## 1. Build Tree

The `build_tree.py` script builds a hierarchical tree structure of diagnoses or medications at a specified level and saves it as a pickle file.
This tree representation helps organize medical codes into a structured format for further analysis. The purpose of this step is to generate a tree dictionary that can be used to generate an outcomes config file.

### 1.1 Usage

```bash
python -m corebehrt.main_causal.build_tree --type [diagnoses|medications] --level [INT]
```

### 1.2 Parameters

- `--type`: Type of data to build tree for (diagnoses or medications)
- `--level`: Level at which to build the tree (integer value)

### 1.3 Purpose

The tree structure enables:

- Organizing codes into a hierarchical representation
- Grouping related diagnoses or medications
- Creating structured outcomes for causal analysis
- Facilitating the generation of outcomes configurations for thousands of codes simultaneously

### 1.4 Outputs

- **Pickle file**: Contains the dictionary representation of the tree at the specified level
- Location: `./outputs/trees/[type]_tree_level_[level].pkl`

---

## 2. Generate Outcomes Config

The `generate_outcomes_config.py` script automatically creates an outcomes configuration YAML file from the tree dictionary generated in the previous step. This allows for the creation of thousands of outcome definitions without manual specification.

### 2.1 Usage

```bash
python -m corebehrt.main_causal.generate_outcomes_config \
    --input ./outputs/trees/[type]_tree_level_[level].pkl \
    --output ./outputs/causal/outcomes/generated_outcomes.yaml \
    --match_how startswith \
    --prepend "CODE_"
```

### 2.2 Parameters

- `--input`: Path to the tree dictionary pickle file (required)
- `--output`: Path to save the generated outcomes config file (default: `./outputs/causal/outcomes/generated_outcomes.yaml`)
- `--prepend`: Optional string to prepend to outcome names
- `--match_how`: Match method to use (choices: `startswith`, `contains`, `exact`; default: `startswith`)
- `--case_sensitive`: Flag to enable case-sensitive matching

### 2.3 Purpose

This script enables automated creation of outcomes configurations for large numbers of medical codes.

### 2.4 Outputs

- **YAML file**: Contains the generated outcomes configuration
- Location: Specified by the `--output` parameter
- Structure: Follows the standard outcomes configuration format with an entry for each code in the tree dictionary

---

## 3. Encode

The `encode` script extracts patient encodings using a **fine-tuned model** and the **processed data** generated during the fine-tuning step.

### 3.1 Inputs

- Fine-tuned model
- Processed patient data from the fine-tuning script

### 3.2 Outputs

- **`encodings.pt`**: Patient-level vector representations

These encodings serve as inputs for downstream causal inference tasks.

---

## 4. Simulate Outcome

The `simulate` script generates synthetic patient outcomes using:

- Encodings from the **encode** step
- Predictions and targets from the fine-tuned model (files: `probas` and `predictions_and_targets`)

### 4.1 Configuration

Edit the **simulation configuration file**:

```yaml
# Simulation Parameters
outcome_model:
  type: sigmoid  # Sigmoid transformation of inputs
  params: {a: 1.5, b: 0.5, c: 0.2}  # Coefficients for simulation

# Counterfactual Generation
counterfactual:
  generate: true
  method: "inverse probability weighting"
```

### 4.2 Outcome Simulation Outputs

- **`simulated_outcomes.csv`**: Generated patient outcomes
- **`counterfactual_probas.csv`** (if enabled): Counterfactual outcome probabilities

---

## 5. Train MLP (on encodings)

The `train_mlp` script trains shallow **multi-layer perceptrons (MLPs)** on the patient encodings to predict:

- **Simulated outcomes** (or)
- **Real target outcomes**

### 5.1 Finetuning Configuration

Edit the **training configuration file**:

```yaml
# Model Parameters
model_args:
  num_layers: 3
  hidden_dims: [256, 128, 64] # input is determined by the encoding size, output is one

trainer_args:
  batch_size: 128
  epochs: 50
  early_stopping: 5
  optimizer: adamw
  loss_function: binary_cross_entropy
```

### 5.2 Training Outputs

- **`mlp_probas.pt`**: Predicted probabilities from the shallow MLPs
- **`mlp_predictions.pt`**: Model predictions

---

## 6. Estimate Treatment Effects

The `estimate` script combines multiple sources of information to compute treatment effect estimates:

- **Fine-tuned model predictions and targets**
- **Shallow MLP predictions and targets**
- **Simulated counterfactual outcomes**

### 6.1 Estimation Methods

The script supports multiple causal inference techniques:

- **Inverse Probability Weighting (IPW)**
- **Augmented Inverse Probability Weighting (AIPW)**
- **Targeted Maximum Likelihood Estimation (TMLE)**
- **Matching-based methods**

### 6.2 Estimation Outputs

- **`treatment_effects.csv`**: Estimated treatment effects
- **`bootstrap_results.pt`** (if enabled): Bootstrapped standard errors

---

## Summary

| Step                     | Script           | Key Configs | Output Files |
|--------------------------|-----------------|-------------|-------------|
| **1. Build Tree** | `build_tree.py` | Type and level | Pickle file |
| **2. Generate Outcomes Config** | `generate_outcomes_config.py` | Input and output paths, match method, prepend string | YAML file |
| **3. Encode** | `encode` | Fine-tuned model | `encodings.pt` |
| **4. Simulate Outcome** | `simulate` | Outcome simulation, counterfactuals | `simulated_outcomes.csv`, `counterfactual_probas.csv` |
| **5. Train MLP** | `train_mlp` | MLP parameters, early stopping | `mlp_probas.pt`, `mlp_predictions.pt` |
| **6. Estimate Effects** | `estimate` | Causal inference methods, bootstrap | `treatment_effects.csv`, `bootstrap_results.pt` |

---

📖 **A good starting point is the `configs` folder, where examples for each step are provided.**
