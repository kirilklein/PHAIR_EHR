from corebehrt.azure import util
from corebehrt.main.main_causal import calibrate

INPUTS = {"finetune_model": {"type": "uri_folder"}}
OUTPUTS = {
    "calibrated_predictions": {"type": "uri_folder"},
}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "calibrate",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    # Parse args and update config
    util.prepare_config(INPUTS, OUTPUTS)
    # Run command
    calibrate.main_calibrate(util.AZURE_CONFIG_FILE)
