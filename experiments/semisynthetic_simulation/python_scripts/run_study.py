#!/usr/bin/env python3
"""
Runner for the semi-synthetic simulation study.

Structure (mirrors the resampling study, but with the semi-synthetic simulator):

    for each OUTER run (independent simulation, own seed):
        Stage 1 (once):   simulate -> select_cohort -> prepare
        Stage 2 (K times): finetune -> calibrate -> estimate   (folds reshuffled each time)

Each Azure job runs a single outer run (pass --run-id run_NN); the outer loop is
the set of parallel jobs submitted by bash_scripts/submit_runs.sh. Run locally with
--n-runs for several outer runs in one process.
"""

import argparse
import logging
import re
from pathlib import Path

import yaml

from corebehrt.main_causal.simulate_semisynthetic import main_simulate
from corebehrt.main_causal.select_cohort_full import main as main_select_cohort
from corebehrt.main_causal.prepare_ft_exp_y import main as main_prepare
from corebehrt.main_causal.finetune_exp_y import main_finetune
from corebehrt.main_causal.calibrate_exp_y import main_calibrate
from corebehrt.main_causal.estimate import main_estimate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("semisynthetic_study")

DEFAULT_BASE_CONFIGS = Path(__file__).resolve().parent.parent / "base_configs"


def fill_config(base_path: Path, replacements: dict, out_path: Path, edit=None) -> str:
    """Substitute {{...}} placeholders into a base config, optionally edit, and save."""
    text = base_path.read_text()
    for key, value in replacements.items():
        text = text.replace(key, value)
    config = yaml.safe_load(text)
    if edit is not None:
        edit(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(config, sort_keys=False))
    return str(out_path)


def run_outer(args, run_id: str, seed: int):
    """Run one outer simulation followed by K inner reshuffle fits."""
    run_dir = Path(args.experiment_dir) / run_id
    config_dir = run_dir / "_configs"
    base = Path(args.base_configs_dir)

    shared = {
        "{{MEDS}}": args.meds,
        "{{FEATURES}}": args.features,
        "{{TOKENIZED}}": args.tokenized,
        "{{PRETRAIN_MODEL}}": args.pretrain_model,
        "{{RUN_DIR}}": str(run_dir),
    }

    logger.info("=" * 70)
    logger.info(f"OUTER RUN {run_id} (seed={seed})")
    logger.info("=" * 70)

    # ---- Stage 1: simulate -> select_cohort -> prepare (once) ----
    def set_simulation(config):
        config["seed"] = seed
        if args.sample_fraction is not None or args.sample_size is not None:
            config["sampling"] = {
                "enabled": True,
                "fraction": args.sample_fraction,
                "size": args.sample_size,
            }

    main_simulate(
        fill_config(
            base / "simulate.yaml", shared, config_dir / "simulate.yaml", set_simulation
        )
    )
    main_select_cohort(
        fill_config(
            base / "select_cohort.yaml", shared, config_dir / "select_cohort.yaml"
        )
    )
    main_prepare(
        fill_config(base / "prepare.yaml", shared, config_dir / "prepare.yaml")
    )

    # ---- Stage 2: K inner reshuffle fits ----
    for k in range(1, args.inner_runs + 1):
        inner_id = f"k_{k:02d}"
        inner_dir = run_dir / "reshuffles" / inner_id
        repl = {**shared, "{{INNER_DIR}}": str(inner_dir)}
        logger.info(f"--- {run_id} / inner fit {k}/{args.inner_runs} ({inner_id}) ---")

        def enable_reshuffle(config):
            config.setdefault("data", {})["reshuffle"] = True

        main_finetune(
            fill_config(
                base / "finetune.yaml",
                repl,
                config_dir / f"finetune_{inner_id}.yaml",
                enable_reshuffle,
            )
        )
        main_calibrate(
            fill_config(
                base / "calibrate.yaml", repl, config_dir / f"calibrate_{inner_id}.yaml"
            )
        )
        main_estimate(
            fill_config(
                base / "estimate.yaml", repl, config_dir / f"estimate_{inner_id}.yaml"
            )
        )

    logger.info(f"OUTER RUN {run_id} complete")


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the semi-synthetic simulation study"
    )
    parser.add_argument("--meds", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--tokenized", required=True)
    parser.add_argument("--pretrain-model", dest="pretrain_model", required=True)
    parser.add_argument("--experiment-dir", dest="experiment_dir", required=True)
    parser.add_argument(
        "--base-configs-dir", dest="base_configs_dir", default=str(DEFAULT_BASE_CONFIGS)
    )

    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Single outer run id, e.g. run_03 (seed = base-seed + 3). One Azure job = one run-id.",
    )
    parser.add_argument(
        "--n-runs",
        dest="n_runs",
        type=int,
        default=1,
        help="Number of outer runs in this process (local use; ignored if --run-id is given).",
    )
    parser.add_argument(
        "--inner-runs",
        "-k",
        dest="inner_runs",
        type=int,
        default=2,
        help="Inner reshuffle fits per outer run (variance estimation).",
    )
    parser.add_argument("--base-seed", dest="base_seed", type=int, default=42)

    parser.add_argument(
        "--sample-fraction",
        dest="sample_fraction",
        type=float,
        default=None,
        help="Sample this fraction of patients per run (smoke tests).",
    )
    parser.add_argument("--sample-size", dest="sample_size", type=int, default=None)

    args = parser.parse_args(argv)
    if args.sample_fraction is not None and args.sample_size is not None:
        parser.error("Specify at most one of --sample-fraction / --sample-size")
    return args


def run_number_from_id(run_id: str) -> int:
    match = re.match(r"run_(\d+)", run_id)
    return int(match.group(1)) if match else 0


def main(argv=None):
    args = parse_arguments(argv)
    if args.run_id:
        run_outer(args, args.run_id, args.base_seed + run_number_from_id(args.run_id))
    else:
        for run_number in range(1, args.n_runs + 1):
            run_id = f"run_{run_number:02d}"
            run_outer(args, run_id, args.base_seed + run_number)


if __name__ == "__main__":
    main()
