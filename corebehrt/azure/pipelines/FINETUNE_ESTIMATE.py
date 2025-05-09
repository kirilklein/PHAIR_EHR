"""
Estimate pipeline implementation with optional outcomes, exposures, and cohort.
"""

from typing import Any, Dict

from corebehrt.azure.pipelines.base import PipelineArg, PipelineMeta
from corebehrt.constants.causal.data import OUTCOMES, EXPOSURES, COHORT, DATA


FINETUNE_ESTIMATE = PipelineMeta(
    name="FINETUNE_ESTIMATE",
    help="Run the estimate pipeline.",
    inputs=[
        PipelineArg(
            name=DATA,
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

    # Core implementation of pipeline steps - not a pipeline itself
    def _common_pipeline_steps(
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
                "select_cohort",
            )(
                features=features,
                outcomes=exposures,
            )
            resolved_cohort = select_cohort.outputs.cohort
        else:
            resolved_cohort = cohort

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
            "ps_model": finetune.outputs.model,
            "outcome_mlp": train_mlp.outputs.trained_mlp,
            "encoded_data": encode.outputs.encoded_data,
            "calibrated_predictions": calibrate.outputs.calibrated_predictions,
        }

    # Define all pipeline variants using a dictionary
    pipeline_configs = {}

    # 1. All inputs provided (outcomes, exposures, cohort)
    @dsl.pipeline(
        name="ft_estimate_w_out_exp_cohort",
        description="Pipeline with provided outcomes, exposures, and cohort",
    )
    def _pipeline_all(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
        exposures: Input,
        cohort: Input,
    ) -> dict:
        return _common_pipeline_steps(
            features, tokenized, pretrain_model, exposures, outcomes, cohort
        )

    pipeline_configs[(True, True, True)] = _pipeline_all

    # 2. Outcomes and exposures provided (no cohort)
    @dsl.pipeline(
        name="ft_estimate_w_out_exp",
        description="Pipeline with provided outcomes and exposures",
    )
    def _pipeline_outcomes_exposures(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
        exposures: Input,
    ) -> dict:
        return _common_pipeline_steps(
            features, tokenized, pretrain_model, exposures, outcomes
        )

    pipeline_configs[(True, True, False)] = _pipeline_outcomes_exposures

    # 3. Outcomes and cohort provided (no exposures)
    @dsl.pipeline(
        name="ft_estimate_w_out_cohort",
        description="Pipeline with provided outcomes and cohort",
    )
    def _pipeline_outcomes_cohort(
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

    pipeline_configs[(True, False, True)] = _pipeline_outcomes_cohort

    # 4. Outcomes only provided (no exposures, no cohort)
    @dsl.pipeline(
        name="ft_estimate_w_out_only",
        description="Pipeline with provided outcomes only",
    )
    def _pipeline_outcomes_only(
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

    pipeline_configs[(True, False, False)] = _pipeline_outcomes_only

    # 5. Exposures and cohort provided (no outcomes)
    @dsl.pipeline(
        name="ft_estimate_w_exp_cohort",
        description="Pipeline with provided exposures and cohort",
    )
    def _pipeline_exposures_cohort(
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

    pipeline_configs[(False, True, True)] = _pipeline_exposures_cohort

    # 6. Exposures only provided (no outcomes, no cohort)
    @dsl.pipeline(
        name="ft_estimate_w_exp_only",
        description="Pipeline with provided exposures only",
    )
    def _pipeline_exposures_only(
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

    pipeline_configs[(False, True, False)] = _pipeline_exposures_only

    # 7. Cohort only provided (no outcomes, no exposures)
    @dsl.pipeline(
        name="ft_estimate_w_cohort_only",
        description="Pipeline with provided cohort only",
    )
    def _pipeline_cohort_only(
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

    pipeline_configs[(False, False, True)] = _pipeline_cohort_only

    # 8. No inputs provided (create outcomes, exposures, cohort)
    @dsl.pipeline(
        name="ft_estimate_w_no_optional_inputs",
        description="Pipeline creating all optional inputs",
    )
    def _pipeline_basic(
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

    pipeline_configs[(False, False, False)] = _pipeline_basic

    # Factory function to select the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        """
        Creates the appropriate pipeline based on available inputs.

        This factory function:
        1. Determines which optional inputs are provided
        2. Validates that data is available when needed
        3. Selects the appropriate pipeline variant
        4. Filters kwargs to match the selected pipeline's signature

        Args:
            **kwargs: Pipeline inputs (data, features, tokenized, pretrain_model, exposures, outcomes, cohort)
        Returns:
            Configured pipeline instance
        Raises:
            ValueError: When data is missing but required for creating exposures/outcomes
        """
        has_outcomes = OUTCOMES in kwargs and kwargs[OUTCOMES] is not None
        has_exposures = EXPOSURES in kwargs and kwargs[EXPOSURES] is not None
        has_cohort = COHORT in kwargs and kwargs[COHORT] is not None

        # Validate data availability when needed
        if not (has_exposures and has_outcomes) and (
            DATA not in kwargs or kwargs[DATA] is None
        ):
            missing = []
            if not has_exposures:
                missing.append(EXPOSURES)
            if not has_outcomes:
                missing.append(OUTCOMES)
            raise ValueError(
                f"'data' input is required when {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not provided"
            )

        # Select the appropriate pipeline based on what's provided
        pipeline_key = (has_outcomes, has_exposures, has_cohort)
        selected_pipeline = pipeline_configs[pipeline_key]

        # Filter kwargs to only include parameters the selected pipeline accepts
        from inspect import signature

        pipeline_params = signature(selected_pipeline).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in pipeline_params}

        return selected_pipeline(**filtered_kwargs)

    return pipeline_factory
