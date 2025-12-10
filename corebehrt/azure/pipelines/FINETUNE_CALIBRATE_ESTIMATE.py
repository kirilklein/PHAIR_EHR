"""
Finetune-Calibrate-Estimate pipeline implementation.
Starts from prepared data and runs finetuning, calibration, and estimation.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta
from os.path import join
FINETUNE_CALIBRATE_ESTIMATE = PipelineMeta(
    name="FINETUNE_CALIBRATE_ESTIMATE",
    help="Run finetune, calibrate, and estimate, and get stats starting from prepared data.",
    inputs=[
        PipelineArg(
            name="prepared_data",
            help="Path to the prepared data (output of prepare_ft_exp_y).",
            required=True,
        ),
        PipelineArg(
            name="pretrain_model", help="Path to the pretrained model.", required=True
        ),
        PipelineArg(
            name="counterfactual_outcomes",
            help="Path to counterfactual outcomes (optional, for simulated data).",
            required=False,
        ),
    ],
)


def create(component: callable):
    """
    Define the Finetune-Calibrate-Estimate pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import Input, dsl

    # Core implementation of pipeline steps
    def _common_pipeline_steps(
        prepared_data: Input,
        pretrain_model: Input,
        counterfactual_outcomes: Input = None,
    ) -> dict:
        """Helper function containing common pipeline implementation steps"""

        finetune_exp_y = component(
            "finetune_exp_y",
        )(
            prepared_data=prepared_data,
            pretrain_model=pretrain_model,
        )

        calibrate_exp_y = component(
            "calibrate_exp_y",
        )(
            finetune_model=finetune_exp_y.outputs.model,
        )

        # Build estimate kwargs
        estimate_kwargs = {
            "calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
        }

        get_stats = component(
            "get_stats",
        )(
            ps_calibrated_predictions=calibrate_exp_y.outputs.calibrated_predictions,
        )

        # Add counterfactual outcomes if provided (for simulated data)
        if counterfactual_outcomes is not None:
            estimate_kwargs["counterfactual_outcomes"] = counterfactual_outcomes

        estimate = component(
            "estimate",
        )(**estimate_kwargs)

        return {
            "estimate": estimate.outputs.estimate,
            "calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
        }

    # Define the two pipeline variants (with and without counterfactual outcomes)
    pipeline_configs = {}

    # 1. With counterfactual outcomes (simulated data)
    @dsl.pipeline(
        name="ft_cal_est_w_cf",
        description="Pipeline with counterfactual outcomes",
    )
    def _pipeline_with_counterfactual(
        prepared_data: Input,
        pretrain_model: Input,
        counterfactual_outcomes: Input,
    ) -> dict:
        return _common_pipeline_steps(
            prepared_data, pretrain_model, counterfactual_outcomes
        )

    pipeline_configs[True] = _pipeline_with_counterfactual

    # 2. Without counterfactual outcomes (real data)
    @dsl.pipeline(
        name="ft_cal_est_wo_cf",
        description="Pipeline without counterfactual outcomes",
    )
    def _pipeline_without_counterfactual(
        prepared_data: Input,
        pretrain_model: Input,
    ) -> dict:
        return _common_pipeline_steps(prepared_data, pretrain_model)

    pipeline_configs[False] = _pipeline_without_counterfactual

    # Factory function to select the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        """
        Creates the appropriate pipeline based on whether counterfactual_outcomes is provided.

        Args:
            **kwargs: Pipeline inputs (prepared_data, pretrain_model, counterfactual_outcomes)
        Returns:
            Configured pipeline instance
        """
        has_counterfactual = (
            "counterfactual_outcomes" in kwargs
            and kwargs["counterfactual_outcomes"] is not None
        )

        # Select the appropriate pipeline based on what's provided
        selected_pipeline = pipeline_configs[has_counterfactual]

        # Filter kwargs to only include parameters the selected pipeline accepts
        from inspect import signature

        pipeline_params = signature(selected_pipeline).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in pipeline_params}

        return selected_pipeline(**filtered_kwargs)

    return pipeline_factory
