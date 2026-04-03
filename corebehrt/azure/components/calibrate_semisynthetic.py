from corebehrt.azure.util import job

INPUTS = {
    "data": {"type": "uri_folder"},
}

OUTPUTS = {
    "outcomes": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main_causal import calibrate_semisynthetic

    job.run_main(
        "calibrate_semisynthetic",
        calibrate_semisynthetic.main_calibrate,
        INPUTS,
        OUTPUTS,
    )
