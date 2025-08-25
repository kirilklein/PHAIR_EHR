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
  data_end:                    # End of available data period
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
  min_instances_per_class: 10        # Minimum samples per outcome class
```

- **truncation_len**: Maximum number of tokens in patient sequences
- **min_len**: Filter out sequences shorter than this
- **cv_folds**: Number of cross-validation folds for training
- **min_instances_per_class**: Minimum required samples for each outcome class

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

*Next: Finetuning Configuration (coming soon)*
