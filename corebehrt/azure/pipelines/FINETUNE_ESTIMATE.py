"""
Estimate pipeline implementation with optional cohort.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta
from corebehrt.constants.causal.data import COHORT

FINETUNE_ESTIMATE = PipelineMeta(
    name="FINETUNE_ESTIMATE",
    help="Run the estimate pipeline.",
    inputs=[
        PipelineArg(name="meds", help="Path to the raw input data.", required=True),
        PipelineArg(name="features", help="Path to the features data.", required=True),
        PipelineArg(
            name="tokenized", help="Path to the tokenized data.", required=True
        ),
        PipelineArg(
            name="pretrain_model", help="Path to the pretrained model.", required=True
        ),
        PipelineArg(name="outcomes", help="Path to the outcomes data.", required=True),
        PipelineArg(
            name="exposures", help="Path to the exposures data.", required=True
        ),
        PipelineArg(
            name="cohort",
            help="Path to the cohort data. If not provided, will be created from meds, features and exposures.",
            required=False,
        ),
    ],
)


def create(component: callable):
    """
    Define the Estimate pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import Input, dsl

    # Core implementation of pipeline steps - not a pipeline itself
    def _common_pipeline_steps(
        meds: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
        cohort: Input = None,
    ) -> dict:
        """Helper function containing common pipeline implementation steps"""
        # Get cohort from input or generate it
        if cohort is None:
            select_cohort = component(
                "select_cohort_full",
            )(
                meds=meds,
                features=features,
                exposures=exposures,
            )
            resolved_cohort = select_cohort.outputs.cohort
        else:
            resolved_cohort = cohort

        prepare_finetune_exp_y = component(
            "prepare_ft_exp_y",
            name="prepare_finetune_exp_y",
        )(
            features=features,
            tokenized=tokenized,
            cohort=resolved_cohort,
            exposures=exposures,
            outcomes=outcomes,
        )

        finetune_exp_y = component(
            "finetune_exp_y",
        )(
            prepared_data=prepare_finetune_exp_y.outputs.prepared_data,
            pretrain_model=pretrain_model,
        )
        calibrate_exp_y = component(
            "calibrate_exp_y",
        )(
            finetune_model=finetune_exp_y.outputs.model,
        )
        estimate = component(
            "estimate",
        )(
            calibrated_predictions=calibrate_exp_y.outputs.calibrated_predictions,
        )

        return {
            "estimate": estimate.outputs.estimate,
            "calibrated_predictions": calibrate_exp_y.outputs.calibrated_predictions,
        }

    # Define the two pipeline variants (with and without cohort)
    pipeline_configs = {}

    # 1. All inputs provided (including cohort)
    @dsl.pipeline(
        name="ft_estimate_w_cohort",
        description="Pipeline with provided cohort",
    )
    def _pipeline_with_cohort(
        meds: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
        cohort: Input,
    ) -> dict:
        return _common_pipeline_steps(
            meds, features, tokenized, pretrain_model, exposures, outcomes, cohort
        )

    pipeline_configs[True] = _pipeline_with_cohort

    # 2. Without cohort (will be generated)
    @dsl.pipeline(
        name="ft_estimate_wo_cohort",
        description="Pipeline without cohort (will be generated)",
    )
    def _pipeline_without_cohort(
        meds: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
    ) -> dict:
        return _common_pipeline_steps(
            meds, features, tokenized, pretrain_model, exposures, outcomes
        )

    pipeline_configs[False] = _pipeline_without_cohort

    # Factory function to select the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        """
        Creates the appropriate pipeline based on whether cohort is provided.

        Args:
            **kwargs: Pipeline inputs (meds, features, tokenized, pretrain_model, exposures, outcomes, cohort)
        Returns:
            Configured pipeline instance
        """
        has_cohort = COHORT in kwargs and kwargs[COHORT] is not None

        # Select the appropriate pipeline based on what's provided
        selected_pipeline = pipeline_configs[has_cohort]

        # Filter kwargs to only include parameters the selected pipeline accepts
        from inspect import signature

        pipeline_params = signature(selected_pipeline).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in pipeline_params}

        return selected_pipeline(**filtered_kwargs)

    return pipeline_factory
