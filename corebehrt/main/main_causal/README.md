# Pipeline: Causal Inference for Patient Outcomes

! **This pipeline is under development and this is a template for the pipeline. Not all steps are available yet.**

This guide walks through the steps required to **estimate treatment effects** using patient encodings, outcome simulation, and causal inference techniques. The pipeline consists of:

1. [**Encode**](#1-encode)
2. [**Simulate Outcome**](#2-simulate-outcome)
3. [**Train MLP (on encodings)**](#3-train-mlp-on-encodings)
4. [**Estimate Treatment Effects**](#4-estimate-treatment-effects)

---

## 1. Encode

The `encode` script extracts patient encodings using a **fine-tuned model** and the **processed data** generated during the fine-tuning step.

### Inputs

- Fine-tuned model
- Processed patient data from the fine-tuning script

### Outputs

- **`encodings.pt`**: Patient-level vector representations

These encodings serve as inputs for downstream causal inference tasks.

---

## 2. Simulate Outcome

The `simulate` script generates synthetic patient outcomes using:

- Encodings from the **encode** step
- Predictions and targets from the fine-tuned model (files: `probas` and `predictions_and_targets`)

### Configuration

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

### Outcome Simulation Outputs

- **`simulated_outcomes.csv`**: Generated patient outcomes
- **`counterfactual_probas.csv`** (if enabled): Counterfactual outcome probabilities

---

## 3. Train MLP (on encodings)

The `train_mlp` script trains shallow **multi-layer perceptrons (MLPs)** on the patient encodings to predict:

- **Simulated outcomes** (or)
- **Real target outcomes**

### Finetuning Configuration

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

### Training Outputs

- **`mlp_probas.pt`**: Predicted probabilities from the shallow MLPs
- **`mlp_predictions.pt`**: Model predictions

---

## 4. Estimate Treatment Effects

The `estimate` script combines multiple sources of information to compute treatment effect estimates:

- **Fine-tuned model predictions and targets**
- **Shallow MLP predictions and targets**
- **Simulated counterfactual outcomes**

### Estimation Methods

The script supports multiple causal inference techniques:

- **Inverse Probability Weighting (IPW)**
- **Augmented Inverse Probability Weighting (AIPW)**
- **Targeted Maximum Likelihood Estimation (TMLE)**
- **Matching-based methods**

### Estimation Outputs

- **`treatment_effects.csv`**: Estimated treatment effects
- **`bootstrap_results.pt`** (if enabled): Bootstrapped standard errors

---

## Summary

| Step                     | Script           | Key Configs | Output Files |
|--------------------------|-----------------|-------------|-------------|
| **1. Encode** | `encode` | Fine-tuned model | `encodings.pt` |
| **2. Simulate Outcome** | `simulate` | Outcome simulation, counterfactuals | `simulated_outcomes.csv`, `counterfactual_probas.csv` |
| **3. Finetune Fast** | `finetune_fast` | MLP parameters, early stopping | `mlp_probas.pt`, `mlp_predictions.pt` |
| **4. Estimate Effects** | `estimate` | Causal inference methods, bootstrap | `treatment_effects.csv`, `bootstrap_results.pt` |
| **5. Train MLP** | `train_mlp` | MLP parameters, early stopping | `mlp_probas.pt`, `mlp_predictions.pt` |

---

📖 **A good starting point is the `configs` folder, where examples for each step are provided.**
