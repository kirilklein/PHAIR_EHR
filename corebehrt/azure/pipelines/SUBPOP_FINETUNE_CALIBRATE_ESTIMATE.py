"""
Subpopulation Finetune-Calibrate-Estimate pipeline.
Continues fine-tuning on a subpopulation using checkpoints from a main run,
then calibrates and estimates causal effects.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta

SUBPOP_FINETUNE_CALIBRATE_ESTIMATE = PipelineMeta(
    name="SUBPOP_FINETUNE_CALIBRATE_ESTIMATE",
    help="Continue fine-tuning on a subpopulation from a main run, then calibrate and estimate.",
    inputs=[
        PipelineArg(
            name="prepared_data",
            help="Path to the prepared data (from the main run).",
            required=True,
        ),
        PipelineArg(
            name="finetune_model",
            help="Path to the finetuned model from the main run (used as restart_model).",
            required=True,
        ),
        PipelineArg(
            name="subpopulation_pids",
            help="Path to file with subpopulation patient IDs (.pt).",
            required=True,
        ),
        PipelineArg(
            name="counterfactual_outcomes",
            help="Path to counterfactual outcomes (optional, for simulated data).",
            required=False,
        ),
        PipelineArg(
            name="secondary_cohort_config",
            help="Path to secondary cohort config YAML file (optional).",
            required=False,
        ),
    ],
)


def create(component: callable):
    """Define the Subpopulation Finetune-Calibrate-Estimate pipeline."""
    from azure.ai.ml import Input, dsl

    def _common_pipeline_steps(
        prepared_data: Input,
        finetune_model: Input,
        subpopulation_pids: Input,
        counterfactual_outcomes: Input = None,
        secondary_cohort_config: Input = None,
    ) -> dict:
        finetune_subpop = component(
            "finetune_subpop",
        )(
            prepared_data=prepared_data,
            restart_model=finetune_model,
            subpopulation_pids=subpopulation_pids,
        )

        calibrate_exp_y = component(
            "calibrate_exp_y",
        )(
            finetune_model=finetune_subpop.outputs.model,
        )

        estimate_kwargs = {
            "calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
        }
        if counterfactual_outcomes is not None:
            estimate_kwargs["counterfactual_outcomes"] = counterfactual_outcomes

        estimate = component(
            "estimate",
        )(**estimate_kwargs)

        get_stats_kwargs = {
            "ps_calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
        }
        if secondary_cohort_config is not None:
            get_stats_kwargs["secondary_cohort_config"] = secondary_cohort_config

        get_stats = component(
            "get_stats",
        )(**get_stats_kwargs)

        return {
            "estimate": estimate.outputs.estimate,
            "calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
            "stats": get_stats.outputs.stats,
        }

    pipeline_configs = {}

    @dsl.pipeline(
        name="subpop_ft_cal_est_w_cf",
        description="Subpopulation pipeline with counterfactual outcomes",
    )
    def _pipeline_with_counterfactual(
        prepared_data: Input,
        finetune_model: Input,
        subpopulation_pids: Input,
        counterfactual_outcomes: Input,
    ) -> dict:
        return _common_pipeline_steps(
            prepared_data,
            finetune_model,
            subpopulation_pids,
            counterfactual_outcomes=counterfactual_outcomes,
        )

    pipeline_configs["has_counterfactual"] = _pipeline_with_counterfactual

    @dsl.pipeline(
        name="subpop_ft_cal_est_wo_cf",
        description="Subpopulation pipeline without counterfactual outcomes",
    )
    def _pipeline_without_counterfactual(
        prepared_data: Input,
        finetune_model: Input,
        subpopulation_pids: Input,
    ) -> dict:
        return _common_pipeline_steps(prepared_data, finetune_model, subpopulation_pids)

    pipeline_configs["does_not_have_counterfactual"] = _pipeline_without_counterfactual

    @dsl.pipeline(
        name="subpop_ft_cal_est_w_secondary",
        description="Subpopulation pipeline with secondary cohort config",
    )
    def _pipeline_with_secondary_cohort(
        prepared_data: Input,
        finetune_model: Input,
        subpopulation_pids: Input,
        secondary_cohort_config: Input,
    ) -> dict:
        return _common_pipeline_steps(
            prepared_data,
            finetune_model,
            subpopulation_pids,
            secondary_cohort_config=secondary_cohort_config,
        )

    pipeline_configs["has_secondary_cohort"] = _pipeline_with_secondary_cohort

    @dsl.pipeline(
        name="subpop_ft_cal_est_w_cf_and_secondary",
        description="Subpopulation pipeline with counterfactual outcomes and secondary cohort config",
    )
    def _pipeline_with_both(
        prepared_data: Input,
        finetune_model: Input,
        subpopulation_pids: Input,
        counterfactual_outcomes: Input,
        secondary_cohort_config: Input,
    ) -> dict:
        return _common_pipeline_steps(
            prepared_data,
            finetune_model,
            subpopulation_pids,
            counterfactual_outcomes=counterfactual_outcomes,
            secondary_cohort_config=secondary_cohort_config,
        )

    pipeline_configs["has_both"] = _pipeline_with_both

    def pipeline_factory(**kwargs: Dict[str, Any]):
        has_counterfactual = (
            "counterfactual_outcomes" in kwargs
            and kwargs["counterfactual_outcomes"] is not None
        )
        has_secondary_cohort = (
            "secondary_cohort_config" in kwargs
            and kwargs["secondary_cohort_config"] is not None
        )

        if has_counterfactual and has_secondary_cohort:
            selected_pipeline = pipeline_configs["has_both"]
        elif has_secondary_cohort:
            selected_pipeline = pipeline_configs["has_secondary_cohort"]
        elif has_counterfactual:
            selected_pipeline = pipeline_configs["has_counterfactual"]
        else:
            selected_pipeline = pipeline_configs["does_not_have_counterfactual"]

        from inspect import signature

        pipeline_params = signature(selected_pipeline).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in pipeline_params}

        return selected_pipeline(**filtered_kwargs)

    return pipeline_factory
