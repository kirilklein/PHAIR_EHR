"""
Estimate pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

FINETUNE_ESTIMATE = PipelineMeta(
    name="FINETUNE_ESTIMATE",
    help="Run the estimate pipeline.",
    inputs=[
        PipelineArg(
            name="data",
            help="Path to the raw input data. Only required if outcomes is not provided.",
            required=False,
        ),
        PipelineArg(name="features", help="Path to the features data.", required=True),
        PipelineArg(
            name="tokenized", help="Path to the tokenized data.", required=True
        ),
        PipelineArg(
            name="pretrain_model", help="Path to the pretrained model.", required=True
        ),
        PipelineArg(name="outcomes", help="Path to the outcomes data.", required=False),
        PipelineArg(
            name="exposures",
            help="Path to the exposures data. Used to train propensity model.",
            required=True,
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
    from azure.ai.ml import dsl, Input
    from typing import Dict, Any

    def _common_pipeline_steps(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
    ) -> dict:
        select_cohort = component(
            "select_cohort",
        )(
            features=features,
            outcomes=exposures,
        )

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        finetune = component(
            "finetune_cv",
        )(
            prepared_data=prepare_finetune.outputs.prepared_data,
            pretrain_model=pretrain_model,
        )
        calibrate = component(
            "calibrate",
        )(
            finetune_model=finetune.outputs.model,
        )
        encode = component(
            "encode",
        )(
            finetune_model=finetune.outputs.model,
            prepared_data=prepare_finetune.outputs.prepared_data,
        )

        train_mlp = component(
            "train_mlp",
        )(
            encoded_data=encode.outputs.encoded_data,
            calibrated_predictions=calibrate.outputs.calibrated_predictions,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        train_xgb = component(
            "train_xgb",
        )(
            encoded_data=encode.outputs.encoded_data,
            calibrated_predictions=calibrate.outputs.calibrated_predictions,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        estimate = component(
            "estimate",
        )(
            exposure_predictions=calibrate.outputs.calibrated_predictions,
            outcome_predictions=train_mlp.outputs.trained_mlp,
        )

        return {
            "estimate": estimate.outputs.estimate,
            "ps_model": calibrate.outputs.ps_model,
            "outcome_mlp": train_mlp.outputs.trained_mlp,
            "outcome_xgb": train_xgb.outputs.trained_xgb,
            "encoded_data": encode.outputs.encoded_data,
            "calibrated_predictions": calibrate.outputs.calibrated_predictions,
        }

    @dsl.pipeline(
        name="finetune_with_outcomes",
        description="Finetune CoreBEHRT pipeline with provided outcomes",
    )
    def _pipeline_with_outcomes(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
    ) -> dict:
        return _common_pipeline_steps(features, tokenized, pretrain_model, outcomes)

    @dsl.pipeline(
        name="finetune_without_outcomes",
        description="Finetune CoreBEHRT pipeline with auto-created outcomes",
    )
    def _pipeline_without_outcomes(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
    ) -> dict:
        create_outcomes = component(
            "create_outcomes",
        )(
            data=data,
            features=features,
        )
        return _common_pipeline_steps(
            features, tokenized, pretrain_model, create_outcomes.outputs.outcomes
        )

    # Define a factory function that returns the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        # Check if outcomes is provided
        has_outcomes = "outcomes" in kwargs and kwargs["outcomes"] is not None

        if has_outcomes:
            # With outcomes, we don't need data
            if "data" in kwargs:
                del kwargs["data"]
            return _pipeline_with_outcomes(**kwargs)
        else:
            # Without outcomes, we need data
            if "data" not in kwargs:
                raise ValueError(
                    "'data' input is required when 'outcomes' is not provided"
                )

            # Remove outcomes from kwargs if it exists but is None
            if "outcomes" in kwargs:
                del kwargs["outcomes"]
            return _pipeline_without_outcomes(**kwargs)

    # Return the factory function instead of a direct pipeline
    return pipeline_factory
