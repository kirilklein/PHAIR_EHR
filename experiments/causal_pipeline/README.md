# Causal Pipeline Experiment System

This directory contains a flexible experiment system for running causal inference pipelines with different simulation settings.

## Quick Start

1. Define your experiment settings in `experiment_configs/`
2. Run an experiment:
   - **Windows:** `bat_scripts\run_experiment.bat <experiment_name>`
   - **Linux/Unix:** `bash_scripts/run_experiment.sh <experiment_name>`

## Structure

- `experiment_configs/` - Define experiment-specific simulation parameters
- `base_configs/` - Template configs for each pipeline step  
- `generated_configs/` - Auto-generated configs (don't edit manually)
- `bat_scripts/` - Windows batch scripts for running experiments
  - `run_experiment.bat` - Run baseline experiments only
  - `run_experiment_full.bat` - Run full pipeline (baseline + BERT)
  - `run_all_experiments.bat` - Run ALL baseline experiments sequentially
  - `run_all_experiments_full.bat` - Run ALL full experiments sequentially
  - `run_experiments_ordered.bat` - Run experiments in custom order
  - `run_multiple_experiments.bat` - Run multiple specific experiments with variance support
  - `monitor_experiments.bat` - Real-time monitoring of experiment progress
  - `create_new_experiment.bat` - Helper to create new experiment templates
  - `list_experiments.bat` - List all available experiments
  - `analyze_results.bat` - Analyze experiment results
- `bash_scripts/` - Linux/Unix bash scripts for running experiments
  - `run_experiment.sh` - Run baseline experiments only
  - `run_experiment_full.sh` - Run full pipeline (baseline + BERT)
  - `run_all_experiments.sh` - Run ALL baseline experiments sequentially
  - `run_all_experiments_full.sh` - Run ALL full experiments sequentially
  - `run_experiments_ordered.sh` - Run experiments in custom order
  - `monitor_experiments.sh` - Real-time monitoring of experiment progress
  - `create_new_experiment.sh` - Helper to create new experiment templates
  - `list_experiments.sh` - List all available experiments
  - `analyze_results.sh` - Analyze experiment results
- `python_scripts/` - Helper Python scripts for config generation and analysis

## Example Usage

### Windows (Batch Scripts)

```batch
# Run a single experiment with default paths
bat_scripts\run_experiment.bat ce0_cy0_y0_i0

# Run with custom data paths
bat_scripts\run_experiment.bat ce0_cy0_y0_i0 --meds C:\data\meds --features C:\data\features

# Run with specific run ID (for multiple runs)
bat_scripts\run_experiment.bat ce0_cy0_y0_i0 --run_id run_02

# Run only baseline pipeline
bat_scripts\run_experiment.bat ce0_cy0_y0_i0 --baseline-only

# Run only BERT pipeline (requires baseline data)
bat_scripts\run_experiment.bat ce0_cy0_y0_i0 --bert-only

# Run ALL experiments sequentially
bat_scripts\run_all_experiments.bat

# Run ALL experiments with custom paths
bat_scripts\run_all_experiments.bat --meds C:\data\meds --pretrain-model C:\models\bert

# Run multiple runs of all experiments (for variance studies)
bat_scripts\run_all_experiments.bat --n_runs 5

# Skip existing experiments
bat_scripts\run_all_experiments.bat --skip-existing

# Run only baseline for all experiments
bat_scripts\run_all_experiments.bat --baseline-only

# Run only BERT for all experiments
bat_scripts\run_all_experiments.bat --bert-only

# Run multiple specific experiments
bat_scripts\run_multiple_experiments.bat ce0_cy0_y0_i0 ce0p5_cy0p5_y0_i0 --n_runs 3

# Monitor experiment progress (real-time)
bat_scripts\monitor_experiments.bat

# Create a new experiment template
bat_scripts\create_new_experiment.bat my_new_experiment

# List available experiments
bat_scripts\list_experiments.bat

# Analyze experiment results for specific outcomes
bat_scripts\analyze_results.bat --results_dir .\outputs\causal\sim_study\runs --outcomes "OUTCOME_01 OUTCOME_02"
```

### Linux/Unix (Bash Scripts)

```bash
# Run a single experiment with default paths
bash_scripts/run_experiment.sh ce0_cy0_y0_i0

# Run with custom data paths
bash_scripts/run_experiment.sh ce0_cy0_y0_i0 --meds /data/meds --features /data/features

# Run with specific run ID (for multiple runs)
bash_scripts/run_experiment.sh ce0_cy0_y0_i0 --run_id run_02

# Run only baseline pipeline
bash_scripts/run_experiment.sh ce0_cy0_y0_i0 --baseline-only

# Run only BERT pipeline (requires baseline data)
bash_scripts/run_experiment.sh ce0_cy0_y0_i0 --bert-only

# Run ALL experiments sequentially
bash_scripts/run_all_experiments.sh

# Run ALL experiments with custom paths
bash_scripts/run_all_experiments.sh --meds /data/meds --pretrain-model /models/bert

# Run multiple runs of all experiments (for variance studies)
bash_scripts/run_all_experiments.sh --n_runs 5

# Skip existing experiments
bash_scripts/run_all_experiments.sh --skip-existing

# Run only baseline for all experiments
bash_scripts/run_all_experiments.sh --baseline-only

# Run only BERT for all experiments
bash_scripts/run_all_experiments.sh --bert-only

# Run multiple specific experiments
bash_scripts/run_multiple_experiments.sh ce0_cy0_y0_i0 ce0p5_cy0p5_y0_i0 --n_runs 3

# Monitor experiment progress (real-time)
bash_scripts/monitor_experiments.sh

# Create a new experiment template
bash_scripts/create_new_experiment.sh my_new_experiment

# List available experiments
bash_scripts/list_experiments.sh

# Analyze experiment results for specific outcomes
bash_scripts/analyze_results.sh --results_dir ./outputs/causal/sim_study/runs --outcomes OUTCOME_01 OUTCOME_02
```

## Pre-configured Experiments

- **no_influence** - All influence scales set to 0 (tests null effects)
- **strong_confounding** - Strong confounding scenario (challenging)  
- **minimal_confounding** - Minimal confounding scenario (easier)

## Adding New Experiments

### Option 1: Use the helper script

**Windows:**

```bat
bat_scripts\create_new_experiment.bat my_experiment
# Edit the generated file: ../experiment_configs/my_experiment.yaml
bat_scripts\run_experiment.bat my_experiment
```

**Linux/Unix:**

```bash
bash_scripts/create_new_experiment.sh my_experiment

bash_scripts/run_experiment.sh my_experiment
```

### Option 2: Manual creation

1. Create a new file in `experiment_configs/` (e.g., `my_experiment.yaml`)
2. Define your simulation parameters (see examples)
3. Run with:
   - **Windows:** `bat_scripts\run_experiment.bat my_experiment`
   - **Linux/Unix:** `bash_scripts/run_experiment.sh my_experiment`

## Experiment Configuration

Each experiment config can override:

- **simulation_model**: Latent factor structure and influence scales
- **outcomes**: Outcome definitions and effect sizes
- **exposure**: Exposure model parameters

Example minimal config:

```yaml
experiment_name: my_test
description: "My test experiment"

simulation_model:
  influence_scales:
    shared_to_exposure: 2.0
    shared_to_outcome: 2.0

outcomes:
  OUTCOME:
    exposure_effect: 1.5
```

## Output Structure

All experiment outputs are saved in:

```shell
outputs/causal/experiments/<experiment_name>/
├── simulated_outcomes/     # Simulated data (shared)
├── cohort/                 # Selected cohort (shared)
├── prepared_data/          # Prepared features (shared)
├── models/
│   ├── baseline/           # CatBoost baseline model
│   └── bert/               # BERT fine-tuned model
└── estimate/
    ├── baseline/           # Baseline causal estimates
    │   └── estimate_results.csv
    └── bert/               # BERT causal estimates
        └── estimate_results.csv
```

**Note**: The data preparation steps (simulation, cohort selection, data preparation) are shared between baseline and BERT pipelines to ensure fair comparison.

## What the System Does

The system automatically:

1. **Generates configs** - Creates all necessary config files with correct paths
2. **Runs pipeline** - Executes: simulation → cohort → prepare → finetune → calibrate → estimate → test
3. **Organizes outputs** - Saves all results in organized experiment directories
4. **Error handling** - Stops on errors with clear messaging

No need to manually edit paths or duplicate config files!

## Configurable Data Paths

All scripts support configurable data paths for flexibility across different environments:

### Available Path Arguments

- `--meds` - Path to MEDS data (default: `./example_data/synthea_meds_causal`)
- `--features` - Path to features data (default: `./outputs/causal/data/features`)
- `--tokenized` - Path to tokenized data (default: `./outputs/causal/data/tokenized`)
- `--pretrain-model` - Path to pretrained BERT model (default: `./outputs/causal/pretrain/model`)
- `--experiment-dir` / `-e` - Base directory for experiment outputs (default: `./outputs/causal/sim_study/runs`)

### Examples

**Windows:**

```batch
# Run with all custom paths
bat_scripts\run_experiment.bat ce0_cy0_y0_i0 ^
  --meds C:\custom\data\meds ^
  --features C:\custom\features ^
  --tokenized C:\custom\tokenized ^
  --pretrain-model C:\models\my_bert ^
  --experiment-dir C:\experiments\outputs

# Run all experiments with custom MEDS data
bat_scripts\run_all_experiments.bat --meds C:\data\meds_v2 --n_runs 10

```

**Linux/Unix:**

```bash
# Run with all custom paths
bash_scripts/run_experiment.sh ce0_cy0_y0_i0 \
  --meds /data/meds \
  --features /data/features \
  --tokenized /data/tokenized \
  --pretrain-model /models/bert \
  --experiment-dir /outputs/experiments

# Run all experiments with custom MEDS data
bash_scripts/run_all_experiments.sh --meds /data/meds_v2 --n_runs 10
```

**Note:** Path arguments are automatically propagated through the entire pipeline (config generation → simulation → cohort selection → model training → estimation).

## Multiple Runs and Variance Studies

The system supports running multiple repetitions of experiments for variance estimation:

### Run ID System

Each experiment run is assigned a unique ID (e.g., `run_01`, `run_02`, `run_03`):

- `--run_id run_XX` - Specify a specific run ID for an experiment
- `--n_runs N` - Automatically run N repetitions (creates `run_01` through `run_0N`)
- `--reuse-data` / `-r` - Reuse prepared data from `run_01` for all subsequent runs (default: `true`)
- `--no-reuse-data` - Force regenerate data for each run (useful for testing data generation variance)

### Data Reuse for Variance Studies

**Important:** By default, `run_02` and onwards **reuse the prepared data** from `run_01`. This ensures:

- **Identical patient populations** across all runs
- **Same features and cohorts** for fair comparison
- **Only model training randomness** varies between runs
- **Proper measurement** of model performance variance

#### Windows Examples 1

```batch
# Run 10 repetitions of all experiments (for variance estimation)
bat_scripts\run_all_experiments.bat --n_runs 10

# Run 5 repetitions of specific experiments
bat_scripts\run_multiple_experiments.bat ce0_cy0_y0_i0 ce1_cy1_y0_i0 --n_runs 5

# Run specific run ID (e.g., re-run failed run_03)
bat_scripts\run_experiment.bat ce0_cy0_y0_i0 --run_id run_03

# Disable data reuse (test full pipeline variance)
bat_scripts\run_all_experiments.bat --n_runs 5 --no-reuse-data
```

#### Linux/Unix Examples 1

```bash
# Run 10 repetitions of all experiments
bash_scripts/run_all_experiments.sh --n_runs 10

# Run 5 repetitions of specific experiments
bash_scripts/run_multiple_experiments.sh ce0_cy0_y0_i0 ce1_cy1_y0_i0 --n_runs 5

# Run specific run ID
bash_scripts/run_experiment.sh ce0_cy0_y0_i0 --run_id run_03

# Disable data reuse
bash_scripts/run_all_experiments.sh --n_runs 5 --no-reuse-data
```

### Output Structure with Multiple Runs

```text
outputs/causal/sim_study/runs/
├── run_01/
│   └── ce0_cy0_y0_i0/
│       ├── simulated_outcomes/
│       ├── cohort/
│       ├── prepared_data/
│       └── estimate/
├── run_02/
│   └── ce0_cy0_y0_i0/
│       ├── cohort/           # symlink to run_01 (if reuse enabled)
│       ├── prepared_data/    # symlink to run_01 (if reuse enabled)
│       └── estimate/         # unique model results
└── run_03/
    └── ce0_cy0_y0_i0/
        └── estimate/         # unique model results
```

## Sequential Experiment Execution

The system provides robust sequential execution with proper error handling:

### `run_all_experiments` Scripts

**Windows:** `bat_scripts\run_all_experiments.bat`  
**Linux/Unix:** `bash_scripts/run_all_experiments.sh`

- Automatically finds and runs ALL experiments in `experiment_configs/`
- Supports multiple runs with `--n_runs` for variance studies
- Waits for each experiment to complete before starting the next
- Continues even if individual experiments fail
- Creates timestamped log files with detailed execution records
- Provides comprehensive summary at the end
- Supports `--skip-existing` to skip completed experiments

### `run_multiple_experiments` Scripts

**Windows:** `bat_scripts\run_multiple_experiments.bat`  
**Linux/Unix:** `bash_scripts/run_multiple_experiments.sh`

- Run specific experiments in the order you define
- Supports multiple runs with `--n_runs` for variance studies
- Useful for running subsets or specific sequences
- Same robust error handling as `run_all_experiments` scripts
- Example: `run_multiple_experiments.bat exp1 exp2 exp3 --n_runs 5`

### `monitor_experiments` Scripts

**Windows:** `bat_scripts\monitor_experiments.bat`  
**Linux/Unix:** `bash_scripts/monitor_experiments.sh`

- Real-time monitoring of experiment progress
- Shows currently running Python processes
- Lists recent experiment outputs and log files
- Auto-refreshes every 10 seconds

### Logging

All sequential runs create detailed log files:

- Format: `run_all_experiments_YYYY-MM-DD_HH-MM-SS.log`
- Records start/end times for each experiment
- Tracks success/failure status
- Final summary with counts and failed experiment list

## Analyzing Results

The `analyze_results` scripts provide comprehensive analysis and visualization of experiment results across all runs.

### `analyze_results` Scripts

**Windows:** `bat_scripts\analyze_results.bat`  
**Linux/Unix:** `bash_scripts/analyze_results.sh`

### Features

- **Aggregate results** across all runs and experiments
- **Generate plots** showing performance vs confounding strength, instrument strength, etc.
- **Filter by outcomes** to analyze specific outcomes only
- **Configurable paths** for results and output directories

### Arguments

- `--results_dir` - Directory containing experiment results (default: `./outputs/causal/sim_study/runs`)
- `--output_dir` - Directory to save analysis outputs (default: `./outputs/causal/sim_study/analysis`)
- `--outcomes` - Space-separated list of outcomes to analyze (analyzes all if not specified)

#### Windows Examples

```batch
# Analyze all results with default settings
bat_scripts\analyze_results.bat

# Analyze specific outcomes only
bat_scripts\analyze_results.bat --outcomes "OUTCOME_01 OUTCOME_02"

# Analyze results from custom directory
bat_scripts\analyze_results.bat ^
  --results_dir C:\experiments\results ^
  --output_dir C:\experiments\analysis

# Full custom analysis
bat_scripts\analyze_results.bat ^
  --results_dir .\outputs\causal\sim_study\runs_v2 ^
  --output_dir .\outputs\causal\sim_study\analysis_v2 ^
  --outcomes "OUTCOME_01 OUTCOME_03"
```

#### Linux/Unix Examples

```bash
# Analyze all results with default settings
bash_scripts/analyze_results.sh

# Analyze specific outcomes only
bash_scripts/analyze_results.sh --outcomes OUTCOME_01 OUTCOME_02

# Analyze results from custom directory
bash_scripts/analyze_results.sh \
  --results_dir /experiments/results \
  --output_dir /experiments/analysis

# Full custom analysis
bash_scripts/analyze_results.sh \
  --results_dir ./outputs/causal/sim_study/runs_v2 \
  --output_dir ./outputs/causal/sim_study/analysis_v2 \
  --outcomes OUTCOME_01 OUTCOME_03
```

### Output

The analysis generates:

- **Aggregated CSV files** with summary statistics
- **Performance plots** (bias, coverage, RMSE, etc.) vs confounding/instrument strength
- **Comparison plots** between baseline and BERT models
- **Comprehensive reports** for each outcome
