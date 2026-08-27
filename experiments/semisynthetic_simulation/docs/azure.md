# Running on Azure

## Single Job

```bash
python -m corebehrt.azure job simulate_semisynthetic CPU-20-LP \
  --config corebehrt/configs/causal/simulate_semisynthetic.yaml \
  -e semisynthetic_sim
```

The Azure component is at `corebehrt/azure/components/simulate_semisynthetic.py`.

### Azure Config Paths

Replace local paths with Azure datastore paths:

```yaml
paths:
  data: "researcher_data:path/to/MEDS/data"
  splits: ["tuning"]
  outcomes: "researcher_data:path/to/semisynthetic_outcomes"
```

## Calibration Job

Run the calibration script to inspect probability distributions before committing to a full experiment:

```bash
python -m corebehrt.azure job calibrate_semisynthetic CPU-20-LP \
  --config corebehrt/configs/causal/simulate_semisynthetic.yaml \
  -e semisynthetic_calibrate
```

## Pipeline Integration

The semisynthetic simulator produces the same output format as `simulate_from_sequence`, so it can be swapped into the existing `FINETUNE_ESTIMATE_SIMULATED` pipeline by replacing the simulation component.

To use it in a custom pipeline, the component signature is:

```python
INPUTS = {"data": {"type": "uri_folder"}}
OUTPUTS = {"outcomes": {"type": "uri_folder"}}
```
