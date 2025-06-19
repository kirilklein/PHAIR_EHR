# Causal Inference Pipeline for Large Scale Adverse Event Detection (ADVENT)

## Helper Scripts

### Build Tree (Helper)

`helper_scripts/build_tree.py` - Organizes medical codes into a hierarchical structure for analysis.

```bash
python -m corebehrt.main_causal.helper_scripts.build_tree --type [diagnoses|medications] --level [INT]
```

**Parameters:**

- `--type`: Type of data (diagnoses or medications)
- `--level`: Hierarchical level for tree construction

**Outputs:**

- Tree dictionary at `./outputs/trees/[type]_tree_level_[level].pkl`

### Generate Outcomes Config (Helper)

`helper_scripts/generate_outcomes_config.py` - Creates outcome configuration files from tree dictionaries.

```bash
python -m corebehrt.mai_causal.helper_scripts.generate_outcomes_config \
    --input ./outputs/trees/[type]_tree_level_[level].pkl \
    --output ./outputs/causal/outcomes/generated_outcomes.yaml
```

**Parameters:**

- `--input`: Path to tree dictionary
- `--output`: Path for saving the config
- `--match_how`: Code matching method (startswith, contains, exact)
- `--prepend`: String to prepend to outcome names

**Outputs:**

- YAML configuration file with outcome definitions for multi-task learning