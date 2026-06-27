"""Azure component for the semi-synthetic simulation study (one outer run per job)."""

import shlex

from corebehrt.azure.util import job

INPUTS = {
    "meds": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "pretrain_model": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder"},
}

OUTPUTS = {
    "results": {"type": "uri_folder"},  # Output dir for this outer run
}


def main_run_study(config_path):
    """Translate the job config into run_study CLI arguments and run it."""
    from corebehrt.modules.setup.config import load_config

    cfg = load_config(config_path)

    args = [
        "--meds",
        cfg.paths.meds,
        "--features",
        cfg.paths.features,
        "--tokenized",
        cfg.paths.tokenized,
        "--pretrain-model",
        cfg.paths.pretrain_model,
        "--cohort",
        cfg.paths.cohort,
        "--experiment-dir",
        cfg.paths.results,
    ]

    # Run-specific flags (run-id, inner-runs, sample-fraction, ...) come via --bash-args.
    if hasattr(cfg, "bash_args") and cfg.bash_args:
        args.extend(shlex.split(cfg.bash_args))

    from experiments.semisynthetic_simulation.python_scripts.run_study import main

    main(args)


if __name__ == "__main__":
    job.run_main("run_semisynthetic_study", main_run_study, INPUTS, OUTPUTS)
