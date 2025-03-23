
# README_AI.md

```markdown
# COREBEHRT: Causal Outcomes Repository for EHR-Based Treatment Effects

## REPOSITORY_STRUCTURE

```yaml
corebehrt/
├── configs/
│   └── causal/
│       └── outcomes.yaml       # Example outcome configuration
├── main_causal/               # Main causal inference pipeline components
│   ├── build_tree.py          # Generate hierarchical code trees
│   ├── generate_outcomes_config.py  # Generate outcome configurations
│   ├── encode.py              # Generate patient encodings
│   ├── simulate.py            # Simulate outcomes
│   ├── train_mlp.py           # Train prediction models
│   ├── estimate.py            # Estimate treatment effects
│   └── helper/                # Helper modules for main scripts
├── modules/                   # Core functionality modules
│   └── tree/                  # Tree building functionality
└── outputs/                   # Pipeline outputs
    ├── trees/                 # Tree dictionary outputs
    ├── causal/                # Causal analysis outputs
    │   └── outcomes/          # Outcome configurations
    ├── features/              # Feature outputs
    └── logs/                  # Log files
```

## PIPELINE_COMPONENTS

### [COMPONENT_1]: build_tree.py

- **FUNCTION**: Builds hierarchical tree structure of medical codes
- **INPUTS**:
  - CSV file containing diagnosis/medication codes
  - Level parameter (integer) specifying tree depth
  - Type parameter (string) specifying "diagnoses" or "medications"
- **OUTPUTS**:
  - Pickle file with tree dictionary at specified level: `./outputs/trees/[type]_tree_level_[level].pkl`
- **PARAMETERS**:
  - `--type`: "diagnoses" or "medications"
  - `--level`: Integer specifying tree depth
- **EXECUTION**:
  
  ```bash
  python -m corebehrt.main_causal.build_tree --type [diagnoses|medications] --level [INT]
  ```

- **DEPENDENCIES**: TreeBuilder class from corebehrt.modules.tree.tree

### [COMPONENT_2]: generate_outcomes_config.py

- **FUNCTION**: Generates YAML outcome configurations from tree dictionaries
- **INPUTS**:
  - Tree dictionary pickle file from build_tree.py
- **OUTPUTS**:
  - YAML file with outcome configurations: `./outputs/causal/outcomes/generated_outcomes.yaml`
- **PARAMETERS**:
  - `--input`: Path to tree dictionary pickle (required)
  - `--output`: Path to save outcome config (default: `./outputs/causal/outcomes/generated_outcomes.yaml`)
  - `--prepend`: String to prepend to outcome names (optional)
  - `--match_how`: Match method ("startswith", "contains", "exact")
  - `--case_sensitive`: Flag for case sensitivity
- **EXECUTION**:

  ```bash
  python -m corebehrt.main_causal.generate_outcomes_config --input [PATH] --output [PATH] --match_how [METHOD] --prepend [STRING]
  ```

- **DEPENDENCIES**: pickle, yaml

### [COMPONENT_3]: encode.py

- **FUNCTION**: Extracts patient vector representations
- **INPUTS**:
  - Fine-tuned model
  - Processed patient data
- **OUTPUTS**:
  - Patient encodings: `encodings.pt`
- **DEPENDENCIES**: Not fully specified in provided documentation

### [COMPONENT_4]: simulate.py

- **FUNCTION**: Generates synthetic patient outcomes
- **INPUTS**:
  - Patient encodings
  - Model predictions and targets
- **OUTPUTS**:
  - Simulated outcomes: `simulated_outcomes.csv`
  - Counterfactual probabilities: `counterfactual_probas.csv` (optional)
- **CONFIGURATION**:
  - Outcome model type
  - Simulation parameters
  - Counterfactual generation settings
- **DEPENDENCIES**: Not fully specified in provided documentation

### [COMPONENT_5]: train_mlp.py

- **FUNCTION**: Trains multi-layer perceptrons for outcome prediction
- **INPUTS**:
  - Patient encodings
  - Real or simulated outcomes
- **OUTPUTS**:
  - Model probabilities: `mlp_probas.pt`
  - Model predictions: `mlp_predictions.pt`
- **CONFIGURATION**:
  - Model architecture parameters
  - Training hyperparameters
- **DEPENDENCIES**: Not fully specified in provided documentation

### [COMPONENT_6]: estimate.py

- **FUNCTION**: Estimates treatment effects using causal inference methods
- **INPUTS**:
  - Model predictions and targets
  - Counterfactual outcomes
- **OUTPUTS**:
  - Treatment effect estimates: `treatment_effects.csv`
  - Bootstrap results: `bootstrap_results.pt` (optional)
- **METHODS**:
  - Inverse Probability Weighting (IPW)
  - Augmented Inverse Probability Weighting (AIPW)
  - Targeted Maximum Likelihood Estimation (TMLE)
  - Matching-based methods
- **DEPENDENCIES**: Not fully specified in provided documentation

## DATA_FLOW

```yaml
build_tree.py → TreeDict.pkl → generate_outcomes_config.py → outcomes.yaml
                                                              |
fine-tuned model → encode.py → encodings.pt → simulate.py → simulated_outcomes.csv
                                             → train_mlp.py → mlp_predictions.pt
                                                              |
                                                              ↓
                                                          estimate.py → treatment_effects.csv
```

## CONFIG_FORMATS

### outcomes.yaml

```yaml
logging:
  level: INFO
  path: ./outputs/logs

paths:
  data: ./example_data/example_MEDS_data_w_labs
  outcomes: ./outputs/causal/outcomes
  features: ./outputs/features/

outcomes:
  OUTCOME_NAME:
    type: [code]
    match: [['CODE_VALUE']]
    match_how: startswith|contains|exact
    case_sensitive: true|false
```

### simulation_config.yaml

```yaml
outcome_model:
  type: sigmoid
  params: {a: 1.5, b: 0.5, c: 0.2}

counterfactual:
  generate: true
  method: "inverse probability weighting"
```

### mlp_config.yaml

```yaml
model_args:
  num_layers: 3
  hidden_dims: [256, 128, 64]

trainer_args:
  batch_size: 128
  epochs: 50
  early_stopping: 5
  optimizer: adamw
  loss_function: binary_cross_entropy
```

## TECHNICAL_DETAILS

### Tree Structure

- Tree represents hierarchical organization of medical codes
- Cutoff level: 8 (maximum depth of tree)
- Extend level: Specified by user + 1
- Tree is converted to dictionary at specified level for use in outcome generation

### Outcome Definition

- Each outcome defined by:
  - Name: String identifier
  - Type: Code-based identification
  - Match: Array of code patterns to match
  - Match method: startswith, contains, or exact
  - Case sensitivity: Boolean flag

### Causal Inference Methods

- IPW: Inverse Probability Weighting - weights observations by inverse probability of treatment
- AIPW: Augmented IPW - combines outcome regression with IPW for double robustness
- TMLE: Targeted Maximum Likelihood Estimation - targeted bias reduction approach
- Matching: Pairs similar treated/untreated subjects for effect estimation

## EXECUTION_SEQUENCE

1. Build tree: `python -m corebehrt.main_causal.build_tree --type diagnoses --level 3`
2. Generate outcomes: `python -m corebehrt.main_causal.generate_outcomes_config --input ./outputs/trees/diagnoses_tree_level_3.pkl`
3. Extract encodings: `python -m corebehrt.main_causal.encode`
4. Simulate outcomes: `python -m corebehrt.main_causal.simulate`
5. Train prediction models: `python -m corebehrt.main_causal.train_mlp`
6. Estimate treatment effects: `python -m corebehrt.main_causal.estimate`
