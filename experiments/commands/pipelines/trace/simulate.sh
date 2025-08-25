python -m corebehrt.azure pipeline FINETUNE_ESTIMATE_SIMULATED CPU-300 azure_configs/trace/pipeline_simulate \
  --meds           researcher_data:AKK/shared/MEDS/TRACE/v01/data \
  --pretrain_model researcher_data:AKK/shared/pretrain/models/trace/small/len_512/v01 \
  --features       researcher_data:AKK/shared/features/trace/v01/features \
  --tokenized      researcher_data:AKK/shared/features/trace/v01/tokenized \
  -cp finetune_exp_y=GPU-A100-single \
  -e  trace_sim_ppl_test \
  -c prepare_finetune_exp_y=azure_configs/trace/pipeline_simulate/prepare/simple.yaml \
  -c select_cohort_full=azure_configs/trace/pipeline_simulate/cohort/extract.yaml \
  -c simulate_from_sequence=azure_configs/trace/pipeline_simulate/simulate/test.yaml
