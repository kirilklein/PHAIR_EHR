from corebehrt.azure.util import job

INPUTS = {"finetune_model": {"type": "uri_folder"}}
OUTPUTS = {
    "calibrated_predictions": {"type": "uri_folder"},
}

if __name__ == "__main__":
    from corebehrt.main_causal import calibrate

    job.run_main("calibrate", calibrate.main_calibrate, INPUTS, OUTPUTS)
