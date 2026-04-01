from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "restart_model": {"type": "uri_folder"},
    "subpopulation_pids": {"type": "uri_file"},
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import finetune_subpop

    job.run_main("finetune_subpop", finetune_subpop.main_finetune_subpop, INPUTS, OUTPUTS)
