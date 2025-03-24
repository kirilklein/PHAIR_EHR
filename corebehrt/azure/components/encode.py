from corebehrt.azure.util import job

INPUTS = {"finetune_model": {"type": "uri_folder"}}
OUTPUTS = {
    "encoded_data": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main_causal import encode

    job.run_main("encode", encode.main_encode, INPUTS, OUTPUTS)
