from corebehrt.azure.util import job

INPUTS = {
    "encoded_data": {"type": "uri_folder"},
    "calibrated_predictions": {"type": "uri_folder"},
}

OUTPUTS = {
    "simulated_outcome": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main_causal import simulate

    job.run_main("simulate", simulate.main_simulate, INPUTS, OUTPUTS)
