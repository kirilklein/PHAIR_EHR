# Pipeline Tests

The BONSAI framework includes comprehensive automated testing through GitHub Actions workflows that validate different aspects of the pipeline functionality. These tests ensure reliability and catch regressions across various use cases.

## Core Pipeline Tests

### Standard Pipeline Test (`pipeline.yml`)

**Purpose**: Validates the complete end-to-end BONSAI pipeline functionality including data processing, model training, and evaluation.

**Components tested**:

- **Data Creation**: Converts MEDS format to tokenized features (with and without held-out sets)
- **Data Preparation**: Prepares training data for pretraining
- **Pretraining**: BERT-style masked language modeling on EHR sequences
- **Outcome Generation**: Creates supervised learning outcomes
- **Cohort Selection**: Applies inclusion/exclusion criteria and generates cross-validation folds
- **Finetuning**: K-fold cross-validation with early stopping
- **Out-of-Time Evaluation**: Tests temporal generalization with absolute date cohorts
- **Held-out Evaluation**: Final model assessment on reserved test data

**Performance Tests**: Includes a second job that specifically tests model performance with "good" vs "bad" censoring scenarios and compares BERT models against XGBoost baselines.

## Causal Inference Pipeline Tests

### Causal Pipeline Test (`causal_pipeline.yml`)

**Purpose**: Tests the complete causal inference workflow for estimating treatment effects from observational EHR data.

**Workflow**:

1. **Data Preparation & Pretraining**: Standard MEDS processing and BERT pretraining
2. **Outcome & Cohort Selection**: Creates outcomes and selects cohorts for causal analysis
3. **Exposure/Outcome Prediction**: Prepares data and finetunes models to predict both treatment exposure and outcomes
4. **Performance Validation**: Ensures models meet minimum performance thresholds for reliable causal inference
5. **Calibration**: Calibrates prediction probabilities for accurate uncertainty estimation
6. **Causal Estimation**: Estimates treatment effects using calibrated models
7. **Criteria Extraction & Statistics**: Extracts cohort criteria and generates descriptive statistics

### Data Preparation + XGBoost Test (`causal_test_prep_data_xgb.yml`)

**Purpose**: Specifically tests the data preparation pipeline followed by XGBoost baseline training, providing a comprehensive validation of the preprocessing steps and non-transformer baseline performance.

**Key steps**:

1. **Data Preparation**: Processes MEDS data into analysis-ready format
2. **Outcome & Cohort Selection**: Creates both censored and uncensored cohorts
3. **Feature Preparation**: Prepares tabular features for traditional ML algorithms
4. **XGBoost Training**: Trains gradient boosting models on prepared features
   - Tests on both censored and uncensored data variants
   - Validates performance meets minimum thresholds (AUC > 0.6-0.9 depending on scenario)
   - Compares performance across multiple outcomes (exposure, OUTCOME, OUTCOME_2, OUTCOME_3)
5. **Performance Validation**: Ensures feature engineering produces data suitable for high-performance ML models

This test is particularly valuable because:

- **Validates data quality**: If XGBoost can't achieve good performance, there may be data preprocessing issues
- **Provides baseline comparison**: Establishes non-transformer baseline performance
- **Tests feature engineering**: Ensures tabular feature extraction works correctly
- **Censoring validation**: Compares performance between censored and uncensored scenarios

#### Causal Pipeline with Simulated Outcomes (`causal_pipeline_sim_outcomes.yml`)

**Purpose**: Tests causal inference using simulated outcomes with known ground truth effects.

**Key features**:

- **Simulated Data Generation**: Creates synthetic outcomes with known causal relationships
- **Ground Truth Validation**: Enables validation against known true effects
- **Bias Testing**: Includes estimation with and without bias to test robustness
- **XGBoost Baseline**: Compares transformer-based causal inference against traditional ML

#### Uncensored Causal Pipeline (`causal_pipeline_uncensored.yml`)

**Purpose**: Tests causal inference on uncensored data where all outcomes are observed.

**Validation**:

- **Higher Performance Thresholds**: Expects AUC > 0.9 since censoring is removed
- **Data Validation**: Includes specific tests to validate uncensored data preparation
- **Baseline Comparison**: XGBoost training on uncensored features

### Specialized Tests

#### Causal Estimation Test (`causal_estimate.yml`)

**Purpose**: Focuses specifically on the causal estimation step using generated synthetic data.

**Testing approach**:

- **Multiple Scenarios**: Tests different noise levels and effect sizes
- **Parameter Sensitivity**: Validates estimation under various data conditions
- **Confidence Interval Testing**: Ensures proper uncertainty quantification
- **Robustness Testing**: Tests estimation with different data generation parameters

#### Code Counts Test (`code_counts.yml`)

**Purpose**: Validates medical code counting and rare code mapping functionality.

**Components**:

- **Code Frequency Analysis**: Counts occurrences of medical codes
- **Rare Code Mapping**: Maps infrequent codes to common categories
- **Output Validation**: Compares against expected outputs to catch regressions

### Test Data and Validation

All pipeline tests use synthetic MEDS-format data located in `example_data/synthea_meds_causal/`, ensuring:

- **Reproducible Testing**: Consistent synthetic data across all test runs
- **Privacy Compliance**: No real patient data in testing
- **Controlled Scenarios**: Known data characteristics enable targeted validation

### Performance Validation

The tests include sophisticated performance validation:

- **Target Bounds**: Each test specifies minimum and maximum expected performance (e.g., "exposure:min:0.6,max:0.8")
- **Multi-target Testing**: Validates performance across multiple prediction targets simultaneously
- **Scenario-specific Thresholds**: Different performance expectations for censored vs uncensored data
- **Confidence Interval Validation**: Tests proper uncertainty quantification in causal estimates

These comprehensive tests ensure that BONSAI maintains reliability across its complex multi-step pipeline and sophisticated causal inference capabilities.
