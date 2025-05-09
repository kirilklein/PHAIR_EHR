from corebehrt.azure.util import job

INPUTS = {
    "cohort": {"type": "uri_folder"},
    "meds": {"type": "uri_folder"},
}
OUTPUTS = {"criteria": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal.helper_scripts import extract_criteria

    job.run_main(
        "extract_criteria",
        extract_criteria.main,
        INPUTS,
        OUTPUTS,
    )
