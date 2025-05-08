"""
Estimate pipeline implementation with optional outcomes, exposures, and cohort.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta
from corebehrt.constants.causal.data import OUTCOMES, EXPOSURES

# Add COHORT constant
COHORT = "cohort"

FINETUNE_ESTIMATE = PipelineMeta(
    name="FINETUNE_ESTIMATE",
    help="Run the estimate pipeline.",
    inputs=[
        PipelineArg(
            name="data",
            help="Path to the raw input data. Required if outcomes, exposures or cohort is not provided.",
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
        PipelineArg(
            name="cohort",
            help="Path to the cohort data. If not provided, will be created from features and exposures.",
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

    def _get_cohort(features: Input, exposures: Input, cohort: Input = None):
        """Helper function to get cohort, either from input or by generating it"""
        if cohort is not None:
            return cohort

        select_cohort = component(
            "select_cohort",
        )(
            features=features,
            outcomes=exposures,
        )
        return select_cohort.outputs.cohort

    def _common_pipeline_steps(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
        cohort: Input = None,
    ) -> dict:
        # Get cohort from input or generate it
        resolved_cohort = _get_cohort(features, exposures, cohort)

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=features,
            tokenized=tokenized,
            cohort=resolved_cohort,
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
            cohort=resolved_cohort,
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
            "ps_model": calibrate.outputs.calibrated_predictions,
            "outcome_mlp": train_mlp.outputs.trained_mlp,
            "encoded_data": encode.outputs.encoded_data,
            "calibrated_predictions": calibrate.outputs.calibrated_predictions,
        }

    # Define pipeline configurations with cohort handling - without cohort as parameter
    @dsl.pipeline(
        name="finetune_with_both_with_cohort",
        description="Finetune CoreBEHRT pipeline with provided outcomes, exposures, and cohort",
    )
    def _pipeline_with_both_with_cohort(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        outcomes: Input,
        cohort: Input,
    ) -> dict:
        return _common_pipeline_steps(
            features, tokenized, pretrain_model, exposures, outcomes, cohort
        )

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
        name="finetune_with_outcomes_only_with_cohort",
        description="Finetune CoreBEHRT pipeline with provided outcomes, cohort but auto-created exposures",
    )
    def _pipeline_with_outcomes_only_with_cohort(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
        cohort: Input,
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
            cohort,
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
        name="finetune_with_exposures_only_with_cohort",
        description="Finetune CoreBEHRT pipeline with provided exposures, cohort but auto-created outcomes",
    )
    def _pipeline_with_exposures_only_with_cohort(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        exposures: Input,
        cohort: Input,
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
            cohort,
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
        name="finetune_with_neither_with_cohort",
        description="Finetune CoreBEHRT pipeline with auto-created outcomes, exposures but provided cohort",
    )
    def _pipeline_with_neither_with_cohort(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        cohort: Input,
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
            cohort,
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
        """
        Creates the appropriate pipeline based on available inputs.

        The factory handles combinations of:
        - Exposures (provided or created)
        - Outcomes (provided or created)
        - Cohort (provided or created)

        Why: Provides a unified interface that automatically handles missing data creation
        and parameter cleanup, ensuring proper pipeline configuration regardless of input state.

        Args:
            **kwargs: Pipeline inputs (data, features, tokenized, pretrain_model, exposures, outcomes, cohort)
        Returns:
            Configured pipeline instance
        Raises:
            ValueError: When data is missing but required for creating exposures/outcomes
        """
        has_exposures = EXPOSURES in kwargs and kwargs[EXPOSURES] is not None
        has_outcomes = OUTCOMES in kwargs and kwargs[OUTCOMES] is not None
        has_cohort = COHORT in kwargs and kwargs[COHORT] is not None

        # Validate data availability when needed
        if not (has_exposures and has_outcomes) and (
            "data" not in kwargs or kwargs["data"] is None
        ):
            missing = []
            if not has_exposures:
                missing.append(EXPOSURES)
            if not has_outcomes:
                missing.append(OUTCOMES)
            raise ValueError(
                f"'data' input is required when {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not provided"
            )

        # Create a copy of kwargs to avoid modifying the original
        pipeline_kwargs = kwargs.copy()

        # Select the appropriate pipeline based on what's provided
        if has_exposures and has_outcomes:
            # Remove data from kwargs if it exists (not needed)
            if "data" in pipeline_kwargs:
                del pipeline_kwargs["data"]

            if has_cohort:
                return _pipeline_with_both_with_cohort(**pipeline_kwargs)
            else:
                if COHORT in pipeline_kwargs:
                    del pipeline_kwargs[COHORT]
                return _pipeline_with_both(**pipeline_kwargs)

        elif has_outcomes:
            # With outcomes but no exposures
            if EXPOSURES in pipeline_kwargs:
                del pipeline_kwargs[EXPOSURES]

            if has_cohort:
                return _pipeline_with_outcomes_only_with_cohort(**pipeline_kwargs)
            else:
                if COHORT in pipeline_kwargs:
                    del pipeline_kwargs[COHORT]
                return _pipeline_with_outcomes_only(**pipeline_kwargs)

        elif has_exposures:
            # With exposures but no outcomes
            if OUTCOMES in pipeline_kwargs:
                del pipeline_kwargs[OUTCOMES]

            if has_cohort:
                return _pipeline_with_exposures_only_with_cohort(**pipeline_kwargs)
            else:
                if COHORT in pipeline_kwargs:
                    del pipeline_kwargs[COHORT]
                return _pipeline_with_exposures_only(**pipeline_kwargs)

        else:
            # Neither exposures nor outcomes
            if EXPOSURES in pipeline_kwargs:
                del pipeline_kwargs[EXPOSURES]
            if OUTCOMES in pipeline_kwargs:
                del pipeline_kwargs[OUTCOMES]

            if has_cohort:
                return _pipeline_with_neither_with_cohort(**pipeline_kwargs)
            else:
                if COHORT in pipeline_kwargs:
                    del pipeline_kwargs[COHORT]
                return _pipeline_with_neither(**pipeline_kwargs)

    return pipeline_factory
