from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "finetune_model": {"type": "uri_folder"},
    "subpopulation_pids": {"type": "uri_file", "optional": True},
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal.rerun_val_inference import main_rerun_val_inference

    job.run_main(
        "rerun_val_inference",
        main_rerun_val_inference,
        INPUTS,
        OUTPUTS,
    )
