# Configuration-Based Causal Effect Simulation

This directory contains configuration files for running causal effect simulations using the `induce_causal_effect.py` script.

## Quick Start

### Using a Configuration File

```bash
# Run with default configuration
python tests/data_generation/induce_causal_effect.py --config tests/data_generation/configs/induce_causal_effect.yaml
```

### Configuration File Structure

The YAML configuration files are organized into logical sections:

```yaml
# Required paths
paths:
  source_dir: "./data/input_shards"
  write_dir: "./data/output_shards"

# Exposure configuration
exposure:
  code: "EXPOSURE"
  run_in_days: 30
  compliance_interval_days: 10
  daily_stop_prob: 0.01
  p_base: 0.1
  trigger_codes: [
    "D/25675004",  # confounder for outcome 1
    "D/431855005",
    "D/90781000119102",  # instrument (only affects exposure)
    "D/43878008"
  ]
  trigger_weights: [0.5, 0.1, 0.8, 0.6]

# Multiple outcomes configuration
outcomes:
  outcome_1:
    code: "OUTCOME"
    run_in_days: 30
    p_base: 0.1
    trigger_codes: [
      "D/25675004",  # confounder (also in exposure)
      "D/431855005",
      "D/125605004"  # prognostic (only affects this outcome)
    ]
    trigger_weights: [0.5, 0.1, 0.3]
    exposure_effect: 1.0
  
  outcome_2:
    code: "OUTCOME_2"
    run_in_days: 30
    p_base: 0.01
    trigger_codes: ["D/105531004", "D/65363002"]
    trigger_weights: [0.5, 0.1]
    exposure_effect: 2.0
```

## Causal Relationships

The causal structure is **implicit** based on code placement:

- **Confounders**: Codes that appear in both exposure and outcome sections
- **Instruments**: Codes that only appear in exposure section
- **Prognostic factors**: Codes that only appear in outcome sections
- **Exposure effects**: Direct causal effect of exposure on each outcome

## Available Configuration Files

### `induce_causal_effect.yaml`

- Basic configuration with default parameters
- Multiple outcomes with different causal structures
- Good starting point for most simulations

## Creating Custom Configurations

1. Copy an existing configuration file
2. Modify the parameters for your use case
3. Run the script with your custom config

### Key Parameters to Consider

- **Effect sizes (trigger_weights)**: Range [-3, 3]. Positive = increases probability, negative = decreases
- **Base probabilities (p_base)**: Event rates without triggers (0.01-0.4 typical)
- **Exposure effects**: Direct causal effect of exposure on each outcome (0.0 = no effect, 2.0 = strong effect)
- **Code lists**: Must have same length as their corresponding weight lists

## Output Files

When using configuration files, the simulation will generate:

- `config.yaml`: Copy of the configuration used
- `.ite.csv`: Individual Treatment Effects for each outcome (columns: `ite_{outcome_code}`)
- `.ate.txt`: Average Treatment Effects for each outcome
- `.ate.json`: ATE results in JSON format
- Output parquet files with simulated data organized by split

## Validation

The script validates that:

- Configuration file exists and has valid YAML syntax
- Required paths are provided
- Code and weight lists have matching lengths
- Parameter values are within reasonable ranges

## Troubleshooting

### Common Issues

1. **Missing required paths**: Ensure `source_dir` and `write_dir` are specified
2. **Mismatched list lengths**: Code lists and weight lists must have same length
3. **Invalid YAML**: Check indentation and syntax
4. **File not found**: Verify config file path is correct

### Expected Output Structure

```text
write_dir/
├── config.yaml
├── train/
│   ├── *.parquet
├── tuning/
│   ├── *.parquet
├── held_out/
│   ├── *.parquet
├── .ite.csv
├── .ate.txt
└── .ate.json
```

## Examples

### Simple Simulation

```bash
python tests/data_generation/induce_causal_effect.py \
  --config configs/induce_causal_effect.yaml
```

### Custom Configuration

Create a custom YAML file with your specific trigger codes and weights:

```yaml
exposure:
  code: "DRUG_EXPOSURE"
  p_base: 0.05
  trigger_codes: ["DIABETES", "HYPERTENSION", "PROVIDER_PREF"]
  trigger_weights: [1.2, 0.8, 2.0]

outcomes:
  adverse_event:
    code: "ADVERSE_EVENT"
    p_base: 0.02
    trigger_codes: ["DIABETES", "KIDNEY_DISEASE"]
    trigger_weights: [0.5, 1.1]
    exposure_effect: 1.5
```

In this example:

- `DIABETES` is a confounder (affects both exposure and outcome)
- `HYPERTENSION` and `PROVIDER_PREF` are instruments (only affect exposure)
- `KIDNEY_DISEASE` is prognostic (only affects outcome)
