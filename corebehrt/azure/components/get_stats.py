from corebehrt.azure.util import job

INPUTS = {
    "criteria": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "ps_calibrated_predictions": {"type": "uri_folder", "optional": True},
    "outcome_model": {"type": "uri_folder", "optional": True},
}
OUTPUTS = {"stats": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal.helper_scripts import get_stats

    job.run_main(
        "get_stats",
        get_stats.main,
        INPUTS,
        OUTPUTS,
    )
