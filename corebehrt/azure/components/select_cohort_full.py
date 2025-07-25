from corebehrt.azure.util import job

INPUTS = {
    "features": {"type": "uri_folder"},
    "meds": {"type": "uri_folder"},
    "exposures": {"type": "uri_folder"},
    "index_date_matching": {"type": "uri_file", "optional": True},
}
OUTPUTS = {"cohort": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import select_cohort_full

    job.run_main(
        "select_cohort_full",
        select_cohort_full.main,
        INPUTS,
        OUTPUTS,
    )
