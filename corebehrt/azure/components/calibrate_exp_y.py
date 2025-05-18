from corebehrt.azure.util import job

INPUTS = {"finetune_model": {"type": "uri_folder"}}
OUTPUTS = {
    "calibrated_predictions": {"type": "uri_folder"},
}

if __name__ == "__main__":
    from corebehrt.main_causal.calibrate_exp_y import main_calibrate

    job.run_main("calibrate", main_calibrate, INPUTS, OUTPUTS)
