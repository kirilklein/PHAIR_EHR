<p align="center"><img src="docs/logo_light.png" width="480" alt="BONSAI Causal"></p>

[![Pipeline tests](https://github.com/kirilklein/bonsai-causal/actions/workflows/pipeline.yml/badge.svg)](https://github.com/kirilklein/bonsai-causal/actions/workflows/pipeline.yml)
[![Causal pipeline tests](https://github.com/kirilklein/bonsai-causal/actions/workflows/causal_pipeline.yml/badge.svg)](https://github.com/kirilklein/bonsai-causal/actions/workflows/causal_pipeline.yml)
[![Unittests](https://github.com/kirilklein/bonsai-causal/actions/workflows/unittests.yml/badge.svg)](https://github.com/kirilklein/bonsai-causal/actions/workflows/unittests.yml)
[![Format](https://github.com/kirilklein/bonsai-causal/actions/workflows/format.yml/badge.svg)](https://github.com/kirilklein/bonsai-causal/actions/workflows/format.yml)
[![Lint](https://github.com/kirilklein/bonsai-causal/actions/workflows/lint.yml/badge.svg)](https://github.com/kirilklein/bonsai-causal/actions/workflows/lint.yml)
![Doc Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/kirilklein/9414903a757f9536ee69438142b66184/raw/docstr-coverage.json)
![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/kirilklein/9414903a757f9536ee69438142b66184/raw/covbadge.json)

**BONSAI Causal** extends [BONSAI](https://github.com/FGA-DIKU/BONSAI), a ModernBERT pipeline for Electronic Health Records in [MEDS](https://github.com/Medical-Event-Data-Standard/meds) format, with a full causal-inference chain. BONSAI answers *"will this patient have outcome Y?"*; this repo answers *"what is the effect of exposure A on outcome Y?"* by using the same transformer as propensity and outcome model inside standard causal estimators.

## What it does

- **Cohort selection.** Inclusion/exclusion criteria as logical expressions over codes, ages and lab values. Control index dates are drawn from the exposed distribution, with optional age matching.
- **Joint finetuning.** One transformer predicts exposure propensity and counterfactual outcome probabilities from the same representation, with K-fold cross-validation.
- **Calibration.** Predicted probabilities are calibrated before estimation.
- **Effect estimation.** Propensities and outcome predictions go to [CausalEstimate](https://github.com/kirilklein/CausalEstimate): IPW, AIPW and TMLE with bootstrap CIs, or `TMLE_TH` for single-shot TMLE with analytic influence-curve CIs. Methods are chosen per run in the estimate config.
- **Validation with known effects.** `simulate_semisynthetic` and `simulate_from_sequence` generate outcomes with a specified true effect on real patient histories, so the whole chain can be checked for bias before use on real outcomes.
- **Runs anywhere.** The bundled synthetic data runs the full chain on a laptop in minutes; [Azure ML](corebehrt/azure/README.md) for full-scale data.

## Quickstart

Requires Python 3.12.

```bash
git clone https://github.com/kirilklein/bonsai-causal.git
cd bonsai-causal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the full chain on the bundled synthetic MEDS data (`example_data/synthea_meds_causal`). This is exactly what CI runs:

```bash
C=corebehrt/configs/causal
# Base pipeline: sequences, pretrained model, outcome and exposure files
python -m corebehrt.main.create_data                --config_path $C/prepare_and_pretrain/create_data.yaml
python -m corebehrt.main.prepare_training_data      --config_path $C/prepare_and_pretrain/prepare_pretrain.yaml
python -m corebehrt.main.pretrain                   --config_path $C/prepare_and_pretrain/pretrain.yaml
python -m corebehrt.main.create_outcomes            --config_path $C/outcomes.yaml
# Causal pipeline
python -m corebehrt.main_causal.select_cohort_full  --config_path $C/select_cohort_full/extract.yaml   # exposed/control cohort, index-date matching
python -m corebehrt.main_causal.prepare_ft_exp_y    --config_path $C/finetune/prepare/simple.yaml     # sequences with exposure + outcome targets
python -m corebehrt.main_causal.finetune_exp_y      --config_path $C/finetune/simple.yaml             # joint propensity + outcome model
python -m corebehrt.main_causal.calibrate_exp_y     --config_path $C/finetune/calibrate.yaml          # calibrate predicted probabilities
python -m corebehrt.main_causal.estimate            --config_path $C/estimate.yaml                    # IPW + TMLE (bootstrap CIs) and TMLE_TH (analytic CI)
```

Effect estimates land in `outputs/causal/estimate/simple/` as a CSV with one row per method and outcome.

<p align="center">
  <img src="docs/causal_COREBEHRT_overview.jpg" alt="Causal pipeline overview" width="820">
</p>

The base steps (data creation, pretraining, outcomes, cohorts, plain finetuning) are the BONSAI pipeline; see the [main README](corebehrt/main/README.md) for their options and [ehr2meds](https://github.com/FGA-DIKU/ehr2meds) for converting raw EHR data to MEDS. For the causal steps see the [causal pipeline README](corebehrt/main_causal/README.md) and the [config guide](corebehrt/configs/causal/README.md).

## Repository layout

- [`corebehrt/main`](corebehrt/main/README.md): base pipeline scripts (create_data, pretrain, finetune, ...)
- [`corebehrt/main_causal`](corebehrt/main_causal/README.md): causal scripts (select_cohort_full, finetune_exp_y, calibrate, estimate, simulate)
- [`corebehrt/modules`](corebehrt/modules/overview.md) and [`corebehrt/functional`](corebehrt/functional/overview.md): model, data processing and utilities
- [`corebehrt/configs`](corebehrt/configs/causal/README.md): YAML configs for every stage
- [`corebehrt/azure`](corebehrt/azure/README.md): cloud execution
- `experiments`: Azure configs for the studies run with this repo (semaglutide, TRACE)

## Azure Integration

To run at scale on Azure ML (SDK v2), see the [Azure guide](corebehrt/azure/README.md): configuration, data stores, environment and job submission.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and formatting
- Testing requirements
- Pull request process
- Issue reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BONSAI Causal in your research, please cite this repository:

```bibtex
@software{klein2026bonsaicausal,
  author = {Klein, Kiril Vadimovic},
  title  = {BONSAI Causal: transformer-based causal inference on electronic health records},
  year   = {2026},
  url    = {https://github.com/kirilklein/bonsai-causal}
}
```

The underlying sequence model builds on [BONSAI](https://github.com/FGA-DIKU/BONSAI) and the [CORE-BEHRT](https://arxiv.org/abs/2404.15201) paper (Odgaard, Klein et al., 2024); please cite those as well when using the modeling pipeline.
