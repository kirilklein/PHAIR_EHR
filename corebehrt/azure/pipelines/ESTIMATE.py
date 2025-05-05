"""
Estimate pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta

ESTIMATE = PipelineMeta(
    name="ESTIMATE",
    help="Run the estimate pipeline, given a model finetuned for propensity score estimation.",
    required_inputs={
        "data": {
            "help": "Path to the raw input data. Needed for outcome extraction, \
                 alternatively, outcome can be provided."
        },
        "finetune_model": {"help": "Path to the finetuned propensity score model."},
        "prepared_data": {"help": "Path to the prepared data."},
        "cohort": {"help": "Path to the cohort data."},
        "outcomes": {"help": "Path to the outcomes data."},
        "features": {"help": "Path to the features data."},
    },
)


def create(component: callable):
    """
    Define the Estimate pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="estimate_pipeline", description="Estimate CoreBEHRT pipeline")
    def pipeline(
        data: Input,
        cohort: Input,
        outcomes: Input,
        finetune_model: Input,
        features: Input,
        prepared_data: Input,
    ) -> dict:
        if outcomes is None:
            create_outcomes = component(
                "create_outcomes",
            )(
                data=data,
                features=features,
            )

        encode = component(
            "encode",
        )(
            prepared_data=prepared_data,
            finetune_model=finetune_model,
        )

        calibrate = component(
            "calibrate",
        )(
            finetune_model=finetune_model,
        )
        train_xgb = component(
            "train_xgb",
        )(
            encoded_data=encode.outputs.encoded_data,
            calibrated_predictions=calibrate.outputs.calibrated_predictions,
            cohort=cohort,
            outcomes=create_outcomes.outputs.outcomes if outcomes is None else outcomes,
        )

        estimate = component(
            "estimate",
        )(
            exposure_predictions=calibrate.outputs.calibrated_predictions,
            outcome_predictions=train_xgb.outputs.trained_xgb,
        )
        return {
            "estimate": estimate.outputs.estimate,
        }

    return pipeline
