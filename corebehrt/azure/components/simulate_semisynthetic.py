from corebehrt.azure.util import job

INPUTS = {
    "data": {"type": "uri_folder"},
}

OUTPUTS = {
    "outcomes": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main_causal import simulate_semisynthetic

    job.run_main(
        "simulate_semisynthetic",
        simulate_semisynthetic.main_simulate,
        INPUTS,
        OUTPUTS,
    )
