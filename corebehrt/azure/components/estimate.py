from corebehrt.azure.util import job

INPUTS = {
    "calibrated_predictions": {"type": "uri_folder", "optional": True},
    "exposure_predictions": {"type": "uri_folder", "optional": True},
    "outcome_predictions": {"type": "uri_folder", "optional": True},
    "counterfactual_outcomes": {"type": "uri_folder", "optional": True},
}
OUTPUTS = {
    "estimate": {"type": "uri_folder"},
}

if __name__ == "__main__":
    from corebehrt.main_causal import estimate

    job.run_main("estimate", estimate.main_estimate, INPUTS, OUTPUTS)
