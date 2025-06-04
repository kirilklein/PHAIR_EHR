from corebehrt.azure.util import job

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "outcomes": {"type": "uri_folder", "optional": True},
    "exposures": {"type": "uri_folder", "optional": True},
}
OUTPUTS = {"prepared_data": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main_causal import prepare_ft_exp_y

    job.run_main(
        "prepare_ft_exp_y",
        prepare_ft_exp_y.main,
        INPUTS,
        OUTPUTS,
    )
