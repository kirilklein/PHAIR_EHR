from corebehrt.azure.util import job

INPUTS = {
    "cohort": {"type": "uri_folder"},
    "meds": {"type": "uri_folder"},
}
OUTPUTS = {"cohort_stats": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import get_stats

    job.run_main(
        "get_stats",
        get_stats.main,
        INPUTS,
        OUTPUTS,
    )
