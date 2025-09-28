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
  - `run_multiple_experiments.bat` - Run multiple experiments sequentially (legacy)
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
# Run a single baseline experiment
bat_scripts\run_experiment.bat ce0_cy0_y0_i0

# Run a single full experiment (baseline + BERT)
bat_scripts\run_experiment_full.bat ce0_cy0_y0_i0

# Run only baseline pipeline for an experiment
bat_scripts\run_experiment_full.bat ce0_cy0_y0_i0 --baseline-only

# Run only BERT pipeline for an experiment (requires baseline data)
bat_scripts\run_experiment_full.bat ce0_cy0_y0_i0 --bert-only

# Run ALL baseline experiments sequentially
bat_scripts\run_all_experiments.bat

# Run ALL full experiments sequentially (baseline + BERT)
bat_scripts\run_all_experiments_full.bat

# Skip existing experiments
bat_scripts\run_all_experiments_full.bat --skip-existing

# Run only baseline for all experiments
bat_scripts\run_all_experiments_full.bat --baseline-only

# Run only BERT for all experiments
bat_scripts\run_all_experiments_full.bat --bert-only

# Run experiments in a specific order
bat_scripts\run_experiments_ordered.bat ce0_cy0_y0_i0 ce0p5_cy0p5_y0_i0 ce1_cy1_y0_i0

# Monitor experiment progress (real-time)
bat_scripts\monitor_experiments.bat

# Create a new experiment template
bat_scripts\create_new_experiment.bat my_new_experiment

# List available experiments
bat_scripts\list_experiments.bat

# Analyze experiment results
bat_scripts\analyze_results.bat
```

### Linux/Unix (Bash Scripts)

```bash
# Run a single baseline experiment
bash_scripts/run_experiment.sh ce0_cy0_y0_i0

# Run a single full experiment (baseline + BERT)
bash_scripts/run_experiment_full.sh ce0_cy0_y0_i0

# Run only baseline pipeline for an experiment
bash_scripts/run_experiment_full.sh ce0_cy0_y0_i0 --baseline-only

# Run only BERT pipeline for an experiment (requires baseline data)
bash_scripts/run_experiment_full.sh ce0_cy0_y0_i0 --bert-only

# Run ALL baseline experiments sequentially
bash_scripts/run_all_experiments.sh

# Run ALL full experiments sequentially (baseline + BERT)
bash_scripts/run_all_experiments_full.sh

# Skip existing experiments
bash_scripts/run_all_experiments_full.sh --skip-existing

# Run only baseline for all experiments
bash_scripts/run_all_experiments_full.sh --baseline-only

# Run only BERT for all experiments
bash_scripts/run_all_experiments_full.sh --bert-only

# Run experiments in a specific order
bash_scripts/run_experiments_ordered.sh ce0_cy0_y0_i0 ce0p5_cy0p5_y0_i0 ce1_cy1_y0_i0

# Monitor experiment progress (real-time)
bash_scripts/monitor_experiments.sh

# Create a new experiment template
bash_scripts/create_new_experiment.sh my_new_experiment

# List available experiments
bash_scripts/list_experiments.sh

# Analyze experiment results
bash_scripts/analyze_results.sh
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
# Edit the generated file: experiment_configs/my_experiment.yaml
bat_scripts\run_experiment.bat my_experiment
```

**Linux/Unix:**

```bash
bash_scripts/create_new_experiment.sh my_experiment
# Edit the generated file: experiment_configs/my_experiment.yaml
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

## Sequential Experiment Execution

The system provides robust sequential execution with proper error handling:

### `run_all_experiments` Scripts

**Windows:** `bat_scripts\run_all_experiments.bat`  
**Linux/Unix:** `bash_scripts/run_all_experiments.sh`

- Automatically finds and runs ALL experiments in `experiment_configs/`
- Waits for each experiment to complete before starting the next
- Continues even if individual experiments fail
- Creates timestamped log files with detailed execution records
- Provides comprehensive summary at the end

### `run_experiments_ordered` Scripts

**Windows:** `bat_scripts\run_experiments_ordered.bat`  
**Linux/Unix:** `bash_scripts/run_experiments_ordered.sh`

- Run experiments in a specific order you define
- Useful for running subsets or specific sequences
- Same robust error handling and logging as `run_all_experiments` scripts

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
