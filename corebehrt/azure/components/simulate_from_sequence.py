from corebehrt.azure.util import job

INPUTS = {
    "data": {"type": "uri_folder"},
}

OUTPUTS = {
    "outcomes": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main_causal import simulate_from_sequence

    job.run_main(
        "simulate_from_sequence", simulate_from_sequence.main_simulate, INPUTS, OUTPUTS
    )
