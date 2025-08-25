# Causal Analysis Configuration Guide

This guide provides step-by-step configuration details for running causal analysis pipelines. Each section covers the required configuration files and their parameters for different stages of the causal inference workflow.

## Overview

The causal analysis pipeline consists of several sequential steps:

1. **Cohort Selection** - Identify exposed/control patients with proper matching
2. **Data Preparation** - Prepare data for joint exposure+outcome modeling  
3. **Finetuning** - Train models for exposure and outcome prediction
4. **Calibration** - Calibrate predicted probabilities
5. **Effect Estimation** - Estimate causal effects using various methods

## Step 1: Cohort Selection

**Script:** `select_cohort_full.py`  
**Main Config:** `select_cohort_full/extract.yaml`  
**Criteria Config:** `select_cohort_full/definitions.yaml`

The cohort selection step identifies exposed and control patients, applies inclusion/exclusion criteria, and performs index date matching to create balanced cohorts for causal analysis.

### Main Configuration (`extract.yaml`)

#### Logging

```yaml
logging:
  level: INFO                           # Log level (DEBUG, INFO, WARNING, ERROR)
  path: ./outputs/logs/causal          # Directory for log files
```

#### Input Paths

```yaml
paths:
  ### Inputs
  features: ./outputs/causal/data/features/              # Patient features directory
  meds: ./example_data/synthea_meds_causal              # MEDS format medical data
  splits: [tuning]                                      # Data splits to process
  exposures: ./outputs/causal/finetune/outcomes/        # Exposure definitions directory
  exposure: EXPOSURE.csv                                # Exposure file name
  criteria_config: ./corebehrt/configs/causal/select_cohort_full/definitions.yaml
```

- **features**: Directory containing preprocessed patient features
- **meds**: Source medical data in MEDS format
- **splits**: List of data splits to include (e.g., `[train, tuning, test]`)
- **exposures**: Directory containing exposure event definitions
- **exposure**: CSV file defining exposure events (fallback to outcome if not provided)
- **criteria_config**: Path to inclusion/exclusion criteria definitions

#### Output Paths

```yaml
paths:
  ### Outputs  
  cohort: ./outputs/causal/finetune/cohorts/full/       # Output directory for cohort
```

#### Time Windows

```yaml
time_windows:
  data_end:                    # End of available data period. Used to filter for sufficient follow-up time.
    year: 2020
    month: 01  
    day: 01
  data_start:                  # Start of available data period
    year: 1950
    month: 1
    day: 1
  min_follow_up:              # Minimum time from index date to data end
    days: 1000                # Ensures sufficient outcome observation time
  min_lookback:               # Minimum time from data start to index date  
    days: 365                 # Ensures sufficient baseline covariate period
```

#### Cross-Validation Setup

```yaml
cv_folds: 2                   # Number of CV folds (1 = simple train/val split)
val_ratio: 0.1               # Validation set ratio (used when cv_folds = 1)
test_ratio: 0                # Test set ratio (0 = no test set)
```

#### Index Date Matching

```yaml
index_date_matching:
  birth_year_tolerance: 3     # Years tolerance for age matching
  redraw_attempts: 2          # Attempts to find valid control index dates
  age_adjusted: true          # Enable age-adjusted sampling
```

- **birth_year_tolerance**: Maximum birth year difference for age matching
- **redraw_attempts**: Number of retry attempts for invalid index dates
- **age_adjusted**: Whether to perform age-adjusted control sampling

### Criteria Configuration (`definitions.yaml`)

This file defines the inclusion/exclusion criteria for cohort selection.

#### Criteria Expressions

```yaml
inclusion: min_age_2 | criteria_1    # Logical expression for inclusion
exclusion: min_age_95                # Logical expression for exclusion
```

Use logical operators:

- `|` or `or`: OR condition
- `&` or `and`: AND condition  
- `~`: NOT condition
- Parentheses for grouping: `(criteria_1 & criteria_2) | criteria_3`

#### Criteria Definitions

```yaml
criteria_definitions:
  min_age_2:                         # Age-based criterion
    min_age: 2                       # Minimum age at index date
    
  min_age_95:                        # Upper age limit
    min_age: 95
    
  criteria_1:                        # Code-based criterion
    codes:
      - ^D/431855005.*              # Regex pattern for diagnosis codes
    start_days: -1                   # Days before index date (negative = before)
    end_days: 10_000                # Days after index date (positive = after)
```

**Criterion Types:**

1. **Age-based criteria:**

   ```yaml
   min_age_criterion:
     min_age: 18                     # Minimum age at index date
   ```

2. **Code-based criteria:**

   ```yaml
   diagnosis_criterion:
     codes:
       - ^D/431855005.*             # Regex patterns for medical codes
       - ^P/12345.*                 # Multiple patterns allowed
     start_days: -365               # Look-back period (days before index)
     end_days: 0                    # Look-forward period (days after index)
   ```

3. **Composite criteria:**

   ```yaml
   inclusion: (min_age_18 & diabetes) | prior_medication
   exclusion: pregnancy | kidney_disease
   ```

### Configuration Variants

The repository includes several pre-configured variants:

- **`extract.yaml`**: Standard configuration for real-world data
- **`extract_simulated.yaml`**: Configuration for simulated data (relaxed time windows)
- **`extract_uncensored.yaml`**: Configuration for uncensored analysis (minimal restrictions)

### Usage Example

```bash
# Run cohort selection with default config
python -m corebehrt.main_causal.select_cohort_full

# Run with custom config
python -m corebehrt.main_causal.select_cohort_full --config_path custom_extract.yaml
```

### Outputs

The cohort selection step produces:

- **`patient_ids.pt`**: List of selected patient IDs
- **`folds.json`**: Cross-validation fold assignments  
- **`index_dates.csv`**: Index dates for all patients
- **`stats/`**: Directory with cohort statistics and visualizations
- **`criteria_flags.csv`**: Criteria evaluation results for each patient

---

## Step 2: Data Preparation

**Script:** `prepare_ft_exp_y.py`  
**Config:** `finetune/prepare/simple.yaml` (or variants)

The data preparation step creates datasets for joint exposure and outcome modeling. It processes the selected cohort to generate training sequences with both exposure and outcome targets, preparing the data for the finetuning stage.

### Main Configuration Parameters

#### Logging & Paths

```yaml
logging:
  level: INFO
  path: ./outputs/logs/causal

paths:
  ## INPUTS
  features: ./outputs/causal/data/features             # Preprocessed patient features
  tokenized: ./outputs/causal/data/tokenized          # Tokenized sequences
  cohort: ./outputs/causal/finetune/cohorts/full      # Selected cohort directory
  
  outcomes: ./outputs/causal/finetune/outcomes        # Outcome definitions
  outcome_files:                                      # List of outcome CSV files
    - OUTCOME.csv
    - OUTCOME_2.csv
    - OUTCOME_3.csv
    
  exposures: ./outputs/causal/finetune/outcomes       # Exposure definitions  
  exposure: EXPOSURE.csv                              # Main exposure file
  
  ## OUTPUTS
  prepared_data: ./outputs/causal/finetune/prepared_data  # Output directory
```

#### Data Configuration

```yaml
data:
  type: finetune                      # Dataset type
  truncation_len: 64                  # Maximum sequence length
  min_len: 2                         # Minimum sequence length
  cv_folds: 2                        # Cross-validation folds
  min_instances_per_class: 10        # Minimum required positive samples for each outcome class. Ignore targets with less than this number of positive samples. Useful for one-to-many.
```

- **truncation_len**: Maximum number of tokens in patient sequences
- **min_len**: Filter out sequences shorter than this
- **cv_folds**: Number of cross-validation folds for training
- **min_instances_per_class**: Minimum required positive samples for each outcome class. Ignore targets with less than this number of positive samples. Useful for one-to-many.

#### Exposure Configuration

```yaml
exposure:
  n_hours_censoring: -1              # Censoring time relative to index date
  n_hours_start_follow_up: -1        # Follow-up start time 
  n_hours_end_follow_up: 10          # Follow-up end time
```

- **n_hours_censoring**: When to censor data (negative = before index date)
- **n_hours_start_follow_up**: Start of exposure observation window
- **n_hours_end_follow_up**: End of exposure observation window

#### Outcome Configuration

```yaml
outcome:
  n_hours_start_follow_up: 1         # Outcome observation start
  n_hours_end_follow_up: null        # Outcome observation end (null = no limit)
  n_hours_compliance: null           # Compliance-based follow-up adjustment
  group_wise_follow_up: true         # Use group-specific follow-up times
  delay_death_hours: 336             # Death coding delay adjustment (hours)
```

- **n_hours_start_follow_up**: Start observing outcomes after index date
- **n_hours_end_follow_up**: Stop observing outcomes (null = until data end)
- **group_wise_follow_up**: Whether to use treatment-group specific follow-up
- **delay_death_hours**: Account for delayed death coding (e.g., 2 weeks = 336h)

### Usage Example (Prepare_ft_exp_y.py)

```bash
# Run data preparation with default config
python -m corebehrt.main_causal.prepare_ft_exp_y

# Run with specific variant
python -m corebehrt.main_causal.prepare_ft_exp_y --config_path ./corebehrt/configs/causal/finetune/prepare/simulated.yaml
```

### Outputs (Prepare_ft_exp_y.py)

The data preparation step produces:

- **`folds.json`**: Cross-validation fold assignments for prepared data
- **`cohort_config.yaml`**: Copy of cohort configuration for reference
- **Prepared datasets**: Tokenized sequences with exposure and outcome labels
- **Data splits**: Train/validation splits ready for model training

---

## Step 3: Joint Exposure-Outcome Finetuning

**Script:** `finetune_exp_y.py`  
**Config:** `finetune/simple.yaml` (or variants)

The finetuning step trains a transformer model to jointly predict both exposure propensity and outcome probabilities. This approach enables counterfactual reasoning by learning shared representations while predicting both treatment assignment and outcomes.

### Main Configuration Parameters (finetune_exp_y.py)

#### Logging & Paths (finetune_exp_y.py)

```yaml
logging:
  level: INFO
  path: ./outputs/logs/causal

paths:
  ## INPUTS
  pretrain_model: ./outputs/causal/pretrain/model      # Pre-trained transformer model
  prepared_data: ./outputs/causal/finetune/prepared_data  # Prepared dataset
  
  ## OUTPUTS
  model: ./outputs/causal/finetune/models/simple       # Output model directory
```

#### Model Architecture

```yaml
model:
  head:
    shared_representation: true       # Share representations between exposure/outcome. Shared-> more focus on confounders -> less adjustment for instruments and outcome only-> might reduce performance but more accurate causal estimates in theory.
    bidirectional: true              # Use bidirectional GRU for sequence pooling
    bottleneck_dim: 64               # Dimensionality of bottleneck layer. Lower -> more focus on confounders.
    l1_lambda: 0.2                   # L1 regularization strength. Higher -> more focus on confounders.
    pooling_strategy: gru            # Pooling method: 'gru' or 'cls'. GRU gives better performance but also more expensive. Use for smaller experiments. Set to cls for many outcomes.
```

- **shared_representation**: Whether exposure and outcome heads share representations. Shared-> more focus on confounders -> less adjustment for instruments and outcome only-> might reduce performance but more accurate causal estimates in theory.
- **bidirectional**: Use bidirectional GRU for sequence encoding
- **bottleneck_dim**: Hidden dimension of prediction head bottleneck. Lower -> more focus on confounders.
- **l1_lambda**: L1 regularization to encourage sparsity. Higher -> more focus on confounders.
- **pooling_strategy**: How to pool sequence representations (`gru` or `cls` token). GRU gives better performance but also more expensive. Use for smaller experiments. Set to cls for many outcomes.

#### Training Configuration

```yaml
trainer_args:
  batch_size: 128                    # Training batch size
  val_batch_size: 256               # Validation batch size
  effective_batch_size: 128         # Effective batch size for gradient accumulation
  epochs: 5                         # Maximum training epochs
  early_stopping: 3                 # Early stopping patience
  stopping_criterion: roc_auc       # Metric for early stopping
  
  # Layer freezing
  n_layers_to_freeze: 1             # Number of transformer layers to freeze
  freeze_encoder_on_plateau: true   # Freeze encoder when validation plateaus
  freeze_encoder_on_plateau_threshold: 0.01  # Threshold for plateau detection
  freeze_encoder_on_plateau_patience: 4      # Patience before freezing
  
  # Multi-task learning
  use_pcgrad: true                  # Use PCGrad for multi-task optimization/ doesn't scale well, turn off for >10 outcomes
```

**Key Training Features:**

- **Gradient-based multi-task learning**: PCGrad helps balance exposure and outcome learning
- **Progressive freezing**: Freeze transformer layers when validation plateaus
- **Early stopping**: Prevents overfitting using validation metrics

#### Loss & Optimization

```yaml
trainer_args:
  loss_weight_function:
    _target_: corebehrt.modules.trainer.utils.PositiveWeight.sqrt

optimizer:
  lr: 3e-3                          # Learning rate
  eps: 1e-6                         # Adam epsilon

scheduler:
  _target_: transformers.get_linear_schedule_with_warmup
  num_training_epochs: 15
  num_warmup_epochs: 2
```

- **loss_weight_function**: Handle class imbalance (sqrt weighting)
- **scheduler**: Linear warmup followed by linear decay

#### Metrics & Monitoring

```yaml
metrics:
  roc_auc:
    _target_: corebehrt.modules.monitoring.metrics.ROC_AUC
  pr_auc:
    _target_: corebehrt.modules.monitoring.metrics.PR_AUC

trainer_args:
  plot_histograms: true             # Plot prediction distributions
  plot_all_targets: false           # Plot metrics for all outcomes
  num_targets_to_log: 3            # Number of outcome targets to visualize
```

### Usage Example (finetune_exp_y.py)

```bash
# Run finetuning with default config
python -m corebehrt.main_causal.finetune_exp_y

# Run with simulated data config
python -m corebehrt.main_causal.finetune_exp_y --config_path ./corebehrt/configs/causal/finetune/simulated.yaml
```

### Outputs (finetune_exp_y.py)

The finetuning step produces:

- **Cross-validation models**: Separate model for each CV fold
- **`outcome_names.pt`**: List of outcome names for reference
- **Performance metrics**: ROC-AUC, PR-AUC for exposure and outcomes
- **Training curves**: Loss and metric plots for monitoring
- **Model checkpoints**: Best models based on validation performance

### Multi-Task Learning Strategy

The joint modeling approach offers several advantages:

1. **Shared representations**: Learn common patterns between exposure and outcome
2. **Regularization**: Exposure prediction acts as auxiliary task
3. **Counterfactual reasoning**: Model can predict outcomes under different exposures
4. **Efficiency**: Single model instead of separate exposure/outcome models

---

## Step 4: Probability Calibration

**Script:** `calibrate_exp_y.py`  
**Config:** `finetune/calibrate.yaml` (or variants)

The calibration step adjusts the predicted probabilities from the finetuned model to ensure they are well-calibrated. This is crucial for causal inference methods that rely on accurate probability estimates, such as propensity score weighting and outcome regression.

### Main Configuration Parameters (calibrate_exp_y.py)

#### Logging & Paths (calibrate_exp_y.py)

```yaml
logging:
  level: INFO
  path: ./outputs/logs/causal

paths:
  ## INPUTS
  finetune_model: ./outputs/causal/finetune/models/simple  # Trained model directory
  
  ## OUTPUTS
  calibrated_predictions: ./outputs/causal/finetune/models/simple/calibrated  # Calibrated outputs
```

- **finetune_model**: Directory containing the trained model from Step 3
- **calibrated_predictions**: Output directory for calibrated predictions and artifacts

#### Plotting Configuration

Be carefull with plotting when run on many outcomes.

```yaml
plotting:
  plot_all_outcomes: true           # Generate calibration plots for all outcomes
  num_outcomes_to_plot: 5          # Limit number of outcomes to plot (optional)
```

- **plot_all_outcomes**: Whether to create calibration plots for every outcome
- **num_outcomes_to_plot**: Limit plotting to top N outcomes (by frequency/importance)

### Calibration Process

The calibration step performs several key operations:

1. **Prediction Collection**: Gather raw predictions from all CV folds
2. **Calibration Fitting**: Fit calibration functions (e.g., Platt scaling, isotonic regression)
3. **Probability Adjustment**: Apply calibration to transform raw probabilities
4. **Validation**: Generate calibration plots and reliability diagrams

### Usage Example (calibrate_exp_y.py)

```bash
# Run calibration with default config
python -m corebehrt.main_causal.calibrate_exp_y

# Run with simulated data config
python -m corebehrt.main_causal.calibrate_exp_y --config_path ./corebehrt/configs/causal/finetune/calibrate_simulated.yaml
```

### Outputs (calibrate_exp_y.py)

The calibration step produces:

- **Calibrated predictions**: Adjusted probability estimates for all patients
- **Calibration models**: Fitted calibration functions for each outcome
- **Reliability diagrams**: Plots showing calibration quality before/after adjustment
- **Calibration metrics**: Brier score, calibration error, and other diagnostic metrics
- **Cross-fold consistency**: Calibration performance across different CV folds

### Calibration Methods

The pipeline supports multiple calibration approaches:

1. **Platt Scaling**: Sigmoid function fitting for binary outcomes
2. **Isotonic Regression**: Non-parametric monotonic calibration
3. **Temperature Scaling**: Single parameter adjustment for neural networks

### Quality Assessment

Key metrics for evaluating calibration quality:

- **Calibration Error**: Mean absolute difference between predicted and observed frequencies
- **Brier Score**: Proper scoring rule combining calibration and discrimination
- **Reliability Diagrams**: Visual assessment of probability-frequency correspondence
- **Sharpness**: Distribution of predicted probabilities (resolution)

### Importance for Causal Inference

Well-calibrated probabilities are essential for:

- **Propensity Score Methods**: Accurate treatment probability estimates
- **Outcome Regression**: Reliable counterfactual outcome predictions
- **Doubly Robust Methods**: Both propensity and outcome model accuracy
- **Uncertainty Quantification**: Confident interval estimation

---

## Step 5: Causal Effect Estimation

**Script:** `estimate.py`  
**Config:** `estimate.yaml` (or variants)

The effect estimation step implements various causal inference methods to estimate treatment effects using the calibrated predictions from previous steps. It supports multiple estimators and provides comprehensive analysis with statistical inference and visualization.

### Main Configuration Parameters (estimate.py)

#### Logging & Paths (estimate.py)

```yaml
logging:
  level: INFO
  path: ./outputs/logs/causal

paths:
  ## INPUTS
  calibrated_predictions: ./outputs/causal/finetune/models/simple/calibrated/  # Calibrated model predictions
  counterfactual_outcomes: ./outputs/causal/simulated_outcomes                 # True effects (optional, for validation)
  
  ## OUTPUTS
  estimate: ./outputs/causal/estimate/simple                                   # Effect estimation results
```

- **calibrated_predictions**: Directory with calibrated probabilities from Step 4
- **counterfactual_outcomes**: Optional true counterfactual outcomes for validation (simulated data)
- **estimate**: Output directory for effect estimates and analysis

#### Estimator Configuration

```yaml
estimator:
  methods: ["IPW", "TMLE"]          # Causal inference methods to use
  effect_type: "ATE"                # Type of causal effect to estimate
  n_bootstrap: 30                   # Number of bootstrap samples for confidence intervals
  common_support_threshold: 0.001   # Threshold for common support filtering
```

**Estimation Methods:**

- **IPW**: Inverse Probability Weighting using propensity scores
- **TMLE**: Targeted Maximum Likelihood Estimation (doubly robust)
- **AIPW**: Augmented Inverse Probability Weighting (doubly robust)

**Effect Types:**

- **ATE**: Average Treatment Effect (population-level effect)
- **ATT**: Average Treatment Effect on the Treated
- **ATC**: Average Treatment Effect on the Controls

**Bootstrap Configuration:**

- **n_bootstrap**: Number of bootstrap resamples for uncertainty quantification
- **common_support_threshold**: Minimum propensity score for inclusion (avoids extreme weights)

#### Plotting Configuration (estimate.py)

```yaml
plot:
  contingency_table:
    max_outcomes_per_figure: 10     # Outcomes per contingency table plot
    max_number_of_figures: 10       # Maximum number of figures to generate
  effect_size:
    max_outcomes_per_figure: 10     # Outcomes per effect size plot
    max_number_of_figures: 10       # Maximum number of effect plots
    plot_individual_effects: true   # Create individual effect plots
  adjustment:
    max_outcomes_per_figure: 8      # Outcomes per adjustment analysis plot
    max_number_of_figures: 10       # Maximum adjustment plots
```

- **contingency_table**: Patient count visualizations by treatment/outcome status
- **effect_size**: Treatment effect magnitude comparisons across methods
- **adjustment**: Analysis of covariate balance and adjustment quality

#### Bias Simulation (`estimate_simulated_with_bias.yaml`)

This should help investigate how applying a bias to ps or outcome probas affects the final effect estimates.

```yaml
estimator:
  methods: ["IPW", "TMLE"]
  effect_type: "ATE"
  n_bootstrap: 30
  common_support_threshold: 0.001
  bias:                             # Systematic bias simulation
    ps_bias_type: additive          # Propensity score bias type
    y_bias_type: additive           # Outcome model bias type  
    ps_values: [0.0, 0.5, 1.0]      # Propensity bias magnitudes
    y_values: [0.0, 0.5, 1.0]       # Outcome bias magnitudes
```

**Bias Simulation Parameters:**

- **ps_bias_type**: How to introduce propensity score bias (`additive`, `multiplicative`)
- **y_bias_type**: How to introduce outcome model bias  
- **ps_values**: Range of propensity score bias magnitudes to test
- **y_values**: Range of outcome model bias magnitudes to test

This configuration runs a systematic robustness analysis across different bias scenarios.

### Causal Inference Methods

#### Inverse Probability Weighting (IPW)

Uses propensity scores to reweight observations, creating a pseudo-population where treatment is randomized.

**Advantages:**

- Simple and interpretable
- Only requires propensity score model
- Efficient for large datasets

**Limitations:**

- Sensitive to propensity score misspecification
- Can have high variance with extreme weights

#### Targeted Maximum Likelihood Estimation (TMLE)

Doubly robust method that combines propensity scores and outcome regression with targeted bias reduction.

**Advantages:**

- Doubly robust (consistent if either propensity or outcome model is correct)
- Efficient and less sensitive to model misspecification
- Provides valid confidence intervals

**Limitations:**

- More complex than IPW
- Requires both propensity and outcome models

#### Augmented Inverse Probability Weighting (AIPW)

Another doubly robust method that augments IPW with outcome regression predictions.

**Advantages:**

- Doubly robust like TMLE
- Often more stable than plain IPW
- Straightforward implementation

**Limitations:**

- Can be less efficient than TMLE
- Still sensitive to extreme propensity scores

### Usage Example (estimate.py)

```bash
# Run effect estimation with default config  
python -m corebehrt.main_causal.estimate

# Run with simulated data validation
python -m corebehrt.main_causal.estimate --config_path ./corebehrt/configs/causal/estimate_simulated.yaml

# Run bias sensitivity analysis
python -m corebehrt.main_causal.estimate --config_path ./corebehrt/configs/causal/estimate_simulated_with_bias.yaml
```

### Outputs (estimate.py)

The effect estimation step produces:

#### Core Results

- **`estimate_results.csv`**: Treatment effect estimates with confidence intervals
- **`experiment_stats.csv`**: Cohort statistics and covariate balance
- **`tmle_analysis.csv`**: Detailed TMLE adjustment analysis
- **`patients.pt`**: Final analysis cohort patient IDs

#### Visualizations

- **`effects.png`**: Heatmap of treatment effects across outcomes and methods
- **`contingency_table/`**: Patient count tables by treatment/outcome status  
- **`effects_scatter/`**: Effect magnitude comparisons across methods
- **`adjustment_analysis/`**: Covariate balance and adjustment quality plots

#### Validation (Simulated Data)

- **`true_effects.png`**: True effect magnitudes for comparison
- **`diff.png`**: Differences between estimated and true effects

#### Bias Analysis

- **`bias_simulation_results.csv`**: Effect estimates across bias scenarios

### Statistical Inference

The pipeline provides comprehensive uncertainty quantification:

1. **Bootstrap Confidence Intervals**: Non-parametric uncertainty estimation
2. **Common Support Analysis**: Overlap assessment and extreme weight handling  
3. **Covariate Balance**: Pre/post-adjustment balance evaluation
4. **Sensitivity Analysis**: Robustness to model misspecification (with bias simulation)

### Quality Assessment (estimate.py)

Key diagnostics for evaluating causal estimates:

- **Effect Consistency**: Agreement across different methods (IPW, TMLE, AIPW)
- **Confidence Interval Coverage**: Appropriate uncertainty quantification
- **Covariate Balance**: Successful confounder adjustment
- **Common Support**: Adequate overlap in propensity score distributions
- **Bias Sensitivity**: Robustness to modeling assumptions (when applicable)

