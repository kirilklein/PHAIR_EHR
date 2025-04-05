from corebehrt.azure.util import job

INPUTS = {
    "cohort": {"type": "uri_folder"},
    "meds": {"type": "uri_folder"},
}
OUTPUTS = {"cohort_advanced": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import select_cohort_advanced

    job.run_main(
        "select_cohort_advanced",
        select_cohort_advanced.main,
        INPUTS,
        OUTPUTS,
    )
