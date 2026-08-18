"""
Re-run clean validation inference, then calibrate and estimate.

Uses the trained finetune model checkpoints to regenerate prediction artifacts on
clean validation folds, then feeds the result into the normal calibrate/estimate
pipeline.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta


RERUN_VAL_INFERENCE_CALIBRATE_ESTIMATE = PipelineMeta(
    name="RERUN_VAL_INFERENCE_CALIBRATE_ESTIMATE",
    help=(
        "Re-run clean validation inference from a finetuned model, then calibrate "
        "and estimate causal effects."
    ),
    inputs=[
        PipelineArg(
            name="prepared_data",
            help="Path to the prepared data used to rebuild clean validation folds.",
            required=True,
        ),
        PipelineArg(
            name="finetune_model",
            help="Path to the finetuned model whose fold checkpoints should be reused.",
            required=True,
        ),
        PipelineArg(
            name="subpopulation_pids",
            help="Optional path to subpopulation patient IDs (.pt) for subpopulation reruns.",
            required=False,
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
    """Define the rerun-inference-calibrate-estimate pipeline."""
    from azure.ai.ml import Input, dsl

    @dsl.pipeline(
        name="rerun_val_inference_cal_est",
        description="Rerun clean validation inference before calibrate and estimate",
    )
    def _pipeline(
        prepared_data: Input,
        finetune_model: Input,
        subpopulation_pids: Input = None,
        counterfactual_outcomes: Input = None,
        secondary_cohort_config: Input = None,
    ) -> dict:
        rerun_kwargs = {
            "prepared_data": prepared_data,
            "finetune_model": finetune_model,
        }
        if subpopulation_pids is not None:
            rerun_kwargs["subpopulation_pids"] = subpopulation_pids

        rerun_inference = component(
            "rerun_clean_val_inference",
        )(**rerun_kwargs)

        calibrate_exp_y = component(
            "calibrate_exp_y",
        )(
            finetune_model=rerun_inference.outputs.model,
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
            "rerun_model": rerun_inference.outputs.model,
            "calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
            "estimate": estimate.outputs.estimate,
            "stats": get_stats.outputs.stats,
        }

    def pipeline_factory(**kwargs: Dict[str, Any]):
        from inspect import signature

        pipeline_params = signature(_pipeline).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in pipeline_params}
        return _pipeline(**filtered_kwargs)

    return pipeline_factory
