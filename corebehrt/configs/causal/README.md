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

*Next: Data Preparation Configuration (coming soon)*
