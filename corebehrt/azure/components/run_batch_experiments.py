"""Azure component for running batch resampling experiments."""

import subprocess
import sys
from corebehrt.azure.util import job

INPUTS = {
    "meds": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "pretrain_model": {"type": "uri_folder"},
}

OUTPUTS = {
    "results": {"type": "uri_folder"},  # Base experiments directory
}


def main_run_batch(config_path, unknown_args):
    """
    Run the batch experiments bash script.

    Args passed from Azure job are forwarded to the bash script.

    Args:
        config_path: Path to the generated config file (from Azure ML)
        unknown_args: Additional arguments to pass to the bash script
    """
    # Build command - the bash script is in experiments/causal_pipeline_resample/bash_scripts/
    bash_script = (
        "experiments/causal_pipeline_resample/bash_scripts/run_multiple_experiments.sh"
    )

    # Read config to get paths and pass them as arguments
    from corebehrt.modules.setup.config import load_config

    cfg = load_config(config_path)

    # Build bash command with data paths from config
    cmd = ["bash", bash_script]

    # Add data paths from config
    cmd.extend(["--meds", cfg.paths.meds])
    cmd.extend(["--features", cfg.paths.features])
    cmd.extend(["--tokenized", cfg.paths.tokenized])
    cmd.extend(["--pretrain-model", cfg.paths.pretrain_model])
    cmd.extend(["--experiment-dir", cfg.paths.results])

    # Add any additional arguments passed from Azure
    cmd.extend(unknown_args)

    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {sys.path[0]}")

    # Run the bash script
    result = subprocess.run(cmd, check=True)

    return result.returncode


if __name__ == "__main__":
    job.run_main(
        "run_batch_experiments", main_run_batch, INPUTS, OUTPUTS, allow_unknown=True
    )
