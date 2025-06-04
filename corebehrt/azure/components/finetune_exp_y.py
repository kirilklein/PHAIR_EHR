from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "test_pids": {"type": "uri_file", "optional": True},
    "cohort": {"type": "uri_folder", "optional": True},
    "pretrain_model": {"type": "uri_folder"},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import finetune_exp_y

    job.run_main("finetune_exp_y", finetune_exp_y.main_finetune, INPUTS, OUTPUTS)
