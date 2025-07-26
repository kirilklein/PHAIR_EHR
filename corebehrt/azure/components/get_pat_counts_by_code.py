from corebehrt.azure.util import job

INPUTS = {
    "data": {"type": "uri_folder"},
}

OUTPUTS = {
    "counts": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main_causal import get_pat_counts_by_code

    job.run_main(
        "get_pat_counts_by_code",
        get_pat_counts_by_code.main,
        INPUTS,
        OUTPUTS,
    )
