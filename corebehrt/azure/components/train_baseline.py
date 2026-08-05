from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import train_baseline

    job.run_main("train_baseline", train_baseline.main_baseline, INPUTS, OUTPUTS)
