python -m corebehrt.azure pipeline FINETUNE_ESTIMATE CPU-300 experiments/azure_configs/trace/pipeline \
  --meds           researcher_data:AKK/shared/MEDS/TRACE/v01/data \
  --pretrain_model researcher_data:AKK/shared/pretrain/models/trace/small/len_512/v01 \
  --features       researcher_data:AKK/shared/features/trace/v01/features \
  --tokenized      researcher_data:AKK/shared/features/trace/v01/tokenized \
  --outcomes       researcher_data:AKK/shared/outcomes/trace/atc_level_6 \
  --exposures      researcher_data:AKK/shared/outcomes/trace/lung_cancer \
  -cp finetune_exp_y=GPU-A100-Single \
  -e  trace_ppl_lc_atc6_test \
  -c prepare_finetune_exp_y=experiments/azure_configs/trace/pipeline/prepare/lc_2y.yaml \
  -c select_cohort_full=experiments/azure_configs/trace/pipeline/cohort/lc.yaml

# --cohort         researcher_data:AKK/experiments/ozempic/cohorts/tte/fu_2y/v02 \