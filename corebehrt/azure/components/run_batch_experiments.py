"""Azure component for running batch resampling experiments."""

import subprocess
import shlex
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


def main_run_batch(config_path):
    """
    Run the batch experiments bash script.

    Args:
        config_path: Path to the generated config file (from Azure ML)
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

    # Add bash args from config if present
    if hasattr(cfg, "bash_args") and cfg.bash_args:
        # Parse the bash_args string and add to command
        cmd.extend(shlex.split(cfg.bash_args))

    print(f"Running command: {' '.join(cmd)}")

    # Run the bash script
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout, adjust as needed
        )
        print(f"Script output:\n{result.stdout}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print(f"stderr:\n{e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print(f"Script timed out after 3600 seconds")
        raise


if __name__ == "__main__":
    job.run_main("run_batch_experiments", main_run_batch, INPUTS, OUTPUTS)
