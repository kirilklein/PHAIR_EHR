from corebehrt.azure.util import job

INPUTS = {
    "code_counts": {"type": "uri_folder"},
}
OUTPUTS = {
    "mapping": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main.helper_scripts import map_rare_codes

    job.run_main("map_rare_codes", map_rare_codes.main, INPUTS, OUTPUTS)
