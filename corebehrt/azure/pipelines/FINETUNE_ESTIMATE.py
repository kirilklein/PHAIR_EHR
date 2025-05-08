"""
Estimate pipeline implementation with optional outcomes and exposures.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta

FINETUNE_ESTIMATE = PipelineMeta(
    name="FINETUNE_ESTIMATE",
    help="Run the estimate pipeline.",
    inputs=[
        PipelineArg(
            name="data",
            help="Path to the raw input data. Required if outcomes or exposures is not provided.",
            required=False,
        ),
        PipelineArg(name="features", help="Path to the features data.", required=True),
        PipelineArg(
            name="tokenized", help="Path to the tokenized data.", required=True
        ),
        PipelineArg(
            name="pretrain_model", help="Path to the pretrained model.", required=True
        ),
        PipelineArg(
            name="outcomes",
            help="Path to the outcomes data. If not provided, will be created from data.",
            required=False,
        ),
        PipelineArg(
            name="exposures",
            help="Path to the exposures data. Used to train propensity model. If not provided, will be created from data.",
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
            outcomes=exposures,
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
            "encoded_data": encode.outputs.encoded_data,
            "calibrated_predictions": calibrate.outputs.calibrated_predictions,
        }

    # Define the four possible pipeline configurations
    @dsl.pipeline(
        name="finetune_with_both",
        description="Finetune CoreBEHRT pipeline with provided outcomes and exposures",
    )
    def _pipeline_with_both(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
    ) -> dict:
        return _common_pipeline_steps(
            features, tokenized, pretrain_model, exposures, outcomes
        )

    @dsl.pipeline(
        name="finetune_with_outcomes_only",
        description="Finetune CoreBEHRT pipeline with provided outcomes but auto-created exposures",
    )
    def _pipeline_with_outcomes_only(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
    ) -> dict:
        create_exposures = component("create_outcomes", name="create_exposures")(
            data=data,
            features=features,
        )
        return _common_pipeline_steps(
            features,
            tokenized,
            pretrain_model,
            create_exposures.outputs.outcomes,
            outcomes,
        )

    @dsl.pipeline(
        name="finetune_with_exposures_only",
        description="Finetune CoreBEHRT pipeline with provided exposures but auto-created outcomes",
    )
    def _pipeline_with_exposures_only(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
    ) -> dict:
        create_outcomes = component("create_outcomes", name="create_outcomes")(
            data=data,
            features=features,
        )
        return _common_pipeline_steps(
            features,
            tokenized,
            pretrain_model,
            exposures,
            create_outcomes.outputs.outcomes,
        )

    @dsl.pipeline(
        name="finetune_with_neither",
        description="Finetune CoreBEHRT pipeline with auto-created outcomes and exposures",
    )
    def _pipeline_with_neither(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
    ) -> dict:
        create_exposures = component("create_outcomes", name="create_exposures")(
            data=data,
            features=features,
        )
        create_outcomes = component("create_outcomes", name="create_outcomes")(
            data=data,
            features=features,
        )
        return _common_pipeline_steps(
            features,
            tokenized,
            pretrain_model,
            create_exposures.outputs.outcomes,
            create_outcomes.outputs.outcomes,
        )

    # Define a factory function that returns the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        has_exposures = "exposures" in kwargs and kwargs["exposures"] is not None
        has_outcomes = "outcomes" in kwargs and kwargs["outcomes"] is not None

        # Check if data is provided when needed
        if not (has_exposures and has_outcomes) and (
            "data" not in kwargs or kwargs["data"] is None
        ):
            missing = []
            if not has_exposures:
                missing.append("exposures")
            if not has_outcomes:
                missing.append("outcomes")
            raise ValueError(
                f"'data' input is required when {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not provided"
            )

        # Select the appropriate pipeline based on what's provided
        if has_exposures and has_outcomes:
            # Remove data from kwargs if it exists (not needed)
            if "data" in kwargs:
                del kwargs["data"]
            return _pipeline_with_both(**kwargs)
        elif has_outcomes:
            # With outcomes but no exposures
            if "exposures" in kwargs:
                del kwargs["exposures"]
            return _pipeline_with_outcomes_only(**kwargs)
        elif has_exposures:
            # With exposures but no outcomes
            if "outcomes" in kwargs:
                del kwargs["outcomes"]
            return _pipeline_with_exposures_only(**kwargs)
        else:
            # Neither exposures nor outcomes
            if "exposures" in kwargs:
                del kwargs["exposures"]
            if "outcomes" in kwargs:
                del kwargs["outcomes"]
            return _pipeline_with_neither(**kwargs)

    # Return the factory function instead of a direct pipeline
    return pipeline_factory
