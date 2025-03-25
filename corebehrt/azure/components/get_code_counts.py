from corebehrt.azure.util import job

INPUTS = {
    "data": {"type": "uri_folder"},
    "splits": {"type": "list", "optional": True},
}
OUTPUTS = {
    "counts": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main.helper_scripts import get_code_counts

    job.run_main("get_code_counts", get_code_counts.main, INPUTS, OUTPUTS)
