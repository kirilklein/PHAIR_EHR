"""Azure component for running batch resampling experiments."""

import sys
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
    Run the batch experiments using Python runner.

    Args:
        config_path: Path to the generated config file (from Azure ML)
    """
    # Read config to get paths and pass them as arguments
    from corebehrt.modules.setup.config import load_config

    cfg = load_config(config_path)

    # Get experiment names from config
    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError(
            "No experiments specified in config. Please add 'experiments' field with list of experiment names."
        )

    # Build arguments for Python runner
    args = [
        "--meds",
        cfg.paths.meds,
        "--features",
        cfg.paths.features,
        "--tokenized",
        cfg.paths.tokenized,
        "--pretrain-model",
        cfg.paths.pretrain_model,
        "--experiment-dir",
        cfg.paths.results,
    ]

    # Add bash args from config if present (parsed into individual arguments)
    if hasattr(cfg, "bash_args") and cfg.bash_args:
        # Parse the bash_args string and add to arguments
        args.extend(shlex.split(cfg.bash_args))

    # Add experiment names at the end
    args.extend(experiments)

    print(f"Running Python batch runner with arguments:")
    print(f"  Experiments: {experiments}")
    print(f"  MEDS: {cfg.paths.meds}")
    print(f"  Features: {cfg.paths.features}")
    print(f"  Tokenized: {cfg.paths.tokenized}")
    print(f"  Pretrain model: {cfg.paths.pretrain_model}")
    print(f"  Results dir: {cfg.paths.results}")
    if hasattr(cfg, "bash_args") and cfg.bash_args:
        print(f"  Additional args: {cfg.bash_args}")

    # Import and run the Python batch runner directly
    from experiments.causal_pipeline_resample.python_scripts.run_multiple_experiments import (
        parse_arguments,
        ExperimentRunner,
    )

    # Temporarily replace sys.argv to pass our arguments
    original_argv = sys.argv
    try:
        sys.argv = ["run_multiple_experiments.py"] + args
        parsed_args = parse_arguments()
        runner = ExperimentRunner(parsed_args)
        exit_code = runner.run_all_experiments()

        if exit_code != 0:
            raise RuntimeError(
                f"Batch experiments failed with exit code {exit_code}. Check logs for details."
            )

        print("Batch experiments completed successfully!")
        return 0

    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    job.run_main("run_batch_experiments", main_run_batch, INPUTS, OUTPUTS)
