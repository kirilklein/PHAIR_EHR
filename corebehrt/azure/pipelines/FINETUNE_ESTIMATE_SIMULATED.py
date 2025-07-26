"""
Estimate pipeline implementation with optional cohort.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta

FINETUNE_ESTIMATE_SIMULATED = PipelineMeta(
    name="FINETUNE_ESTIMATE_SIMULATED",
    help="Run the estimate pipeline with simulated cohort.",
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

    @dsl.pipeline(
        name="ft_estimate_simulated",
        description="Pipeline with simulated cohort",
    )
    def _pipeline(
        meds: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
    ) -> dict:
        """Helper function containing common pipeline implementation steps"""
        # Get cohort from input or generate it
        simulate_from_sequence = component(
            "simulate_from_sequence",
        )(data=meds)

        select_cohort = component(
            "select_cohort_full",
        )(
            meds=meds,
            features=features,
            exposures=simulate_from_sequence.outputs.outcomes,
        )
        resolved_cohort = select_cohort.outputs.cohort

        prepare_finetune_exp_y = component(
            "prepare_ft_exp_y",
            name="prepare_finetune_exp_y",
        )(
            features=features,
            tokenized=tokenized,
            cohort=resolved_cohort,
            exposures=simulate_from_sequence.outputs.outcomes,
            outcomes=simulate_from_sequence.outputs.outcomes,
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


    # Factory function to select the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        """
        Creates the appropriate pipeline based on whether cohort is provided.

        Args:
            **kwargs: Pipeline inputs (meds, features, tokenized, pretrain_model, exposures, outcomes, cohort)
        Returns:
            Configured pipeline instance
        """
        return _pipeline(**kwargs)

    return pipeline_factory
