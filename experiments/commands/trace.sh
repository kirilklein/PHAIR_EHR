# trace

## prepare
python -m corebehrt.azure job get_code_counts CPU-20-LP -e trace_stats -c azure_configs/trace/create_data/get_counts.yaml
python -m corebehrt.azure job map_rare_codes CPU-20-LP -e trace_prep -c azure_configs/trace/create_data/map_rare_codes.yaml

# create data
python -m corebehrt.azure job create_data CPU-16C -e trace_prep -c azure_configs/trace/create_data/for_training.yaml

## prepare and pretrain
python -m corebehrt.azure job prepare_training_data CPU-20-LP -e trace_prep -c azure_configs/trace/pretrain/prepare.yaml
python -m corebehrt.azure job pretrain GPU-A100-Small -e trace_pretrain -c azure_configs/trace/pretrain/small_mid.yaml


## outcomes
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_lc_prep_ft -c azure_configs/trace/outcomes/death.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_lc_prep_ft -c azure_configs/trace/outcomes/lung_cancer.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_atc_prep_ft -c azure_configs/trace/outcomes/mammography.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_atc_prep_ft -c azure_configs/trace/outcomes/atc_level_3.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_atc_prep_ft -c azure_configs/trace/outcomes/atc_level_5.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_atc_prep_ft -c azure_configs/trace/outcomes/atc_level_6.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_atc_prep_ft -c azure_configs/trace/outcomes/frequent_drugs.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e trace_atc_prep_ft -c azure_configs/trace/outcomes/frequent_drugs_small.yaml

## Next: run pipeline