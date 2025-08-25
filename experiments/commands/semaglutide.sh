# ozempic

## prepare
python -m corebehrt.azure job get_code_counts CPU-20-LP -e sgl_prep -c azure_configs/semaglutide/create_data/get_counts.yaml
python -m corebehrt.azure job map_rare_codes CPU-20-LP -e sgl_prep -c azure_configs/semaglutide/create_data/map_rare_codes.yaml

## Data creation
python -m corebehrt.azure job create_data CPU-16C -e sgl_prep -c azure_configs/semaglutide/create_data/for_training.yaml
python -m corebehrt.azure job create_data CPU-300 -e sgl_prep_all_ft -c azure_configs/semaglutide/create_data/for_cohort.yaml

## prepare and pretrain
python -m corebehrt.azure job prepare_training_data CPU-20-LP -e sgl_prep -c azure_configs/semaglutide/pretrain/prepare.yaml
python -m corebehrt.azure job pretrain GPU-A100-Small -e sgl_pretrain -c azure_configs/semaglutide/pretrain/small_mid.yaml

## create outcomes
python -m corebehrt.azure job create_outcomes CPU-300 -e sgl_outcomes -c azure_configs/semaglutide/outcomes/all.yaml
python -m corebehrt.azure job create_outcomes CPU-300 -e sgl_outcomes -c azure_configs/semaglutide/outcomes/semaglutide.yaml

## cohort selection
python -m corebehrt.azure job select_cohort_full CPU-300 -e sgl_prep_ft -c azure_configs/semaglutide/cohort_sustain/extract.yaml

python -m corebehrt.azure job extract_criteria CPU-300 -e sgl_stats -c azure_configs/semaglutide/cohort_sustain/stats/extract_criteria.yaml
python -m corebehrt.azure job get_stats CPU-300 -e sgl_stats -c azure_configs/semaglutide/cohort_sustain/stats/get_stats.yaml

## Extract criteria and get stats
python -m corebehrt.azure job extract_criteria CPU-20-LP -e ozempic_adult_diab_extract -c azure_configs/ozempic/stats/extract.yaml
python -m corebehrt.azure job get_stats CPU-20-LP -e sgl_stats -c azure_configs/semaglutide/stats/adult_diab2_no_cvd.yaml

## Next: run pipeline