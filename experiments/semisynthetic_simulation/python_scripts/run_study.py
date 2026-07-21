#!/usr/bin/env python3
"""
Runner for the semi-synthetic simulation study.

Design (see docs/study.md):
- The cohort is FIXED and shared (a pre-existing select_cohort_full output).
  Index dates and membership come from it; treatment is the real exposure.
- Stage 1 (once per outer run): simulate outcomes -> prepare finetune data.
- Stage 2 (K inner refits): fit -> calibrate -> estimate, for the causal model
  (BERT) and/or the CatBoost baseline.

Inner refits:
- Each refit uses a distinct model seed and fold reshuffling.
- Conditional on each fitted model, estimate draws B patient-level bootstrap
  samples. The summarizer combines all K x B estimates.

Each Azure job runs one outer run (--run-id run_NN); the outer loop = the set
of parallel jobs submitted by bash_scripts/submit_runs.sh.
"""

import argparse
import logging
import re
from pathlib import Path

import yaml

from corebehrt.main_causal.simulate_semisynthetic import main_simulate
from corebehrt.main_causal.prepare_ft_exp_y import main as main_prepare
from corebehrt.main_causal.finetune_exp_y import main_finetune
from corebehrt.main_causal.train_baseline import main_baseline
from corebehrt.main_causal.calibrate_exp_y import main_calibrate
from corebehrt.main_causal.estimate import main_estimate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("semisynthetic_study")

DEFAULT_BASE_CONFIGS = Path(__file__).resolve().parent.parent / "base_configs"
DEFAULT_BOOTSTRAPS = 100


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
    """Run one outer simulation followed by K independently seeded refits."""
    run_dir = (
        Path(args.experiment_dir) if args.run_id else Path(args.experiment_dir) / run_id
    )
    config_dir = run_dir / "_configs"
    base = Path(args.base_configs_dir)

    shared = {
        "{{MEDS}}": args.meds,
        "{{FEATURES}}": args.features,
        "{{TOKENIZED}}": args.tokenized,
        "{{PRETRAIN_MODEL}}": args.pretrain_model,
        "{{COHORT}}": args.cohort,
        "{{RUN_DIR}}": str(run_dir),
    }
    logger.info("=" * 70)
    logger.info(
        f"OUTER RUN {run_id} (seed={seed}, refits={args.inner_runs}, "
        f"bootstraps={args.n_bootstrap})"
    )
    logger.info("=" * 70)

    # ---- Stage 1: simulate -> prepare (once) ----
    main_simulate(
        fill_config(
            base / "simulate.yaml",
            shared,
            config_dir / "simulate.yaml",
            lambda c: c.update(seed=seed),
        )
    )
    main_prepare(
        fill_config(base / "prepare.yaml", shared, config_dir / "prepare.yaml")
    )

    # ---- Stage 2: K inner refits ----
    for k in range(1, args.inner_runs + 1):
        inner_id = f"k_{k:02d}"
        inner_dir = run_dir / "reshuffles" / inner_id
        repl = {**shared, "{{INNER_DIR}}": str(inner_dir)}
        logger.info(f"--- {run_id} / refit {k}/{args.inner_runs} ({inner_id}) ---")

        refit_seed = seed * 1000 + k

        def configure_refit(config, _seed=refit_seed):
            """Apply the model seed and reshuffle folds without resampling patients."""
            config.setdefault("data", {})["reshuffle"] = True
            config["data"]["reshuffle_seed"] = _seed
            config["seed"] = _seed

        def configure_estimation(config, _seed=refit_seed):
            estimator = config.setdefault("estimator", {})
            estimator["n_bootstrap"] = args.n_bootstrap
            estimator["bootstrap_seed"] = _seed
            estimator["save_bootstrap_samples"] = True
            estimator["use_observed_point_estimate"] = True

        if not args.baseline_only:
            _run_model(
                "bert",
                base,
                repl,
                config_dir,
                inner_id,
                configure_refit,
                configure_estimation,
                main_finetune,
                "finetune.yaml",
                "calibrate.yaml",
                "estimate.yaml",
            )
        if not args.bert_only:
            _run_model(
                "baseline",
                base,
                repl,
                config_dir,
                inner_id,
                configure_refit,
                configure_estimation,
                main_baseline,
                "train_baseline.yaml",
                "calibrate_baseline.yaml",
                "estimate_baseline.yaml",
            )

    logger.info(f"OUTER RUN {run_id} complete")


def _run_model(
    name,
    base,
    repl,
    config_dir,
    inner_id,
    fit_edit,
    est_edit,
    fit_main,
    fit_cfg,
    cal_cfg,
    est_cfg,
):
    """Run fit -> calibrate -> estimate for one model family on one refit."""
    fit_main(
        fill_config(
            base / fit_cfg, repl, config_dir / f"{name}_fit_{inner_id}.yaml", fit_edit
        )
    )
    main_calibrate(
        fill_config(base / cal_cfg, repl, config_dir / f"{name}_cal_{inner_id}.yaml")
    )
    main_estimate(
        fill_config(
            base / est_cfg, repl, config_dir / f"{name}_est_{inner_id}.yaml", est_edit
        )
    )


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the semi-synthetic study")
    parser.add_argument("--meds", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--tokenized", required=True)
    parser.add_argument("--pretrain-model", dest="pretrain_model", required=True)
    parser.add_argument(
        "--cohort", required=True, help="Pre-existing cohort dir (fixed across runs)"
    )
    parser.add_argument("--experiment-dir", dest="experiment_dir", required=True)
    parser.add_argument(
        "--base-configs-dir", dest="base_configs_dir", default=str(DEFAULT_BASE_CONFIGS)
    )

    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Single outer run id, e.g. run_03 (seed = base-seed + 3).",
    )
    parser.add_argument("--n-runs", dest="n_runs", type=int, default=1)
    parser.add_argument(
        "--inner-runs",
        "-k",
        dest="inner_runs",
        type=int,
        default=1,
        help="Independently seeded model refits per outer run.",
    )
    parser.add_argument(
        "--n-bootstrap",
        "-b",
        dest="n_bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help="Patient-level bootstrap samples per fitted propensity model.",
    )
    parser.add_argument("--base-seed", dest="base_seed", type=int, default=42)

    parser.add_argument("--bert-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")

    args = parser.parse_args(argv)
    if args.bert_only and args.baseline_only:
        parser.error("Cannot specify both --bert-only and --baseline-only")
    if args.inner_runs < 1:
        parser.error("--inner-runs must be at least 1")
    if args.n_bootstrap < 2:
        parser.error("--n-bootstrap must be at least 2")
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
