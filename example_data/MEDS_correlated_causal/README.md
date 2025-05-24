# Simulated Data Generation

The dataset was generated in two steps to simulate a controlled causal structure for downstream analysis.

## Step 1: Simulate Causal Effect Between `EXPOSURE` and `OUTCOME`

In the first step, we simulate a causal relationship between the `EXPOSURE` variable and the `OUTCOME`. The resulting data is saved to a specified directory.

```bash
python tests/data_generation/induce_causal_effect.py \
    --source_dir example_data/correlated_MEDS_data/{split} \
    --write_dir example_data/MEDS_correlated_causal/{split}
```

For splits `train`, `tuning`, and `held_out`.

```bash
python tests/data_generation/induce_causal_effect.py --source_dir example_data/correlated_MEDS_data/train --write_dir example_data/MEDS_correlated_causal/train
python tests/data_generation/induce_causal_effect.py --source_dir example_data/correlated_MEDS_data/tuning --write_dir example_data/MEDS_correlated_causal/tuning
python tests/data_generation/induce_causal_effect.py --source_dir example_data/correlated_MEDS_data/held_out --write_dir example_data/MEDS_correlated_causal/held_out
```
