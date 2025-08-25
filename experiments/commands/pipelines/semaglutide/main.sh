python -m corebehrt.azure pipeline FINETUNE_ESTIMATE CPU-300 azure_configs/semaglutide/pipeline \
  --meds           researcher_data:AKK/shared/MEDS/SEMAGLUTIDE/training/data \
  --pretrain_model researcher_data:AKK/shared/pretrain/models/semaglutide/small/len_512/no_sep_no_cls/v01 \
  --features       researcher_data:AKK/shared/features/semaglutide/training/v03/features \
  --tokenized      researcher_data:AKK/shared/features/semaglutide/training/v03/tokenized \
  --outcomes       researcher_data:AKK/shared/outcomes/semaglutide/all/v02 \
  --exposures      researcher_data:AKK/shared/outcomes/semaglutide/semaglutide/v01 \
  --cohort        researcher_data:AKK/experiments/semaglutide/cohorts/tte/fu_2y/v04 \
  -cp finetune_exp_y=GPU-A100-Small \
  -e  semaglutide_tte_ppl_cohort_fixed_bce \
