from corebehrt.azure.util import job

INPUTS = {
    "encoded_data": {"type": "uri_folder"},
    "calibrated_predictions": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder"},
    "outcomes": {"type": "uri_file"},
}
OUTPUTS = {
    "trained_xgb": {"type": "uri_folder"},
}

if __name__ == "__main__":
    from corebehrt.main_causal import train_xgb

    job.run_main("train_xgb", train_xgb.main_train, INPUTS, OUTPUTS)
