from corebehrt.azure import util
from corebehrt.main.main_causal import simulate

INPUTS = {
    "encoded_data": {"type": "uri_folder"},
    "calibrated_predictions": {"type": "uri_folder"},
}

OUTPUTS = {
    "simulated_outcome": {"type": "uri_folder"},
}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "simulate",
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
    simulate.main_simulate(util.AZURE_CONFIG_FILE)
