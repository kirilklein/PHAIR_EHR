#!/usr/bin/env python3
"""
Python-based batch experiment runner for Azure jobs.

This script replicates the functionality of run_multiple_experiments.sh but runs
natively in Python, avoiding bash subprocess issues in Azure environments.
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Import the config generation function
from experiments.causal_pipeline_resample.python_scripts.generate_configs import (
    generate_experiment_configs,
)

# Import main functions from corebehrt modules
from corebehrt.main_causal.simulate_with_sampling import main_simulate
from corebehrt.main_causal.select_cohort_full import main as main_select_cohort
from corebehrt.main_causal.prepare_ft_exp_y import main as main_prepare_finetune
from corebehrt.main_causal.train_baseline import main_baseline
from corebehrt.main_causal.calibrate_exp_y import main_calibrate
from corebehrt.main_causal.estimate import main_estimate
from corebehrt.main_causal.finetune_exp_y import main_finetune

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for batch causal pipeline experiments with resampling."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.experiments_dir = Path(args.experiment_dir)
        self.failed_experiments = []
        self.timeout_experiments = []
        self.success_count = 0
        self.current_count = 0
        self.total_count = len(args.experiments) * args.n_runs

        # Initialize logging
        self.log_dir = self.experiments_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.setup_batch_logging()

    def setup_batch_logging(self):
        """Setup batch-level logging to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"batch_run_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        logger.info(f"Batch log file: {log_file}")
        logger.info("=" * 80)
        logger.info("Running Multiple Causal Pipeline Experiments")
        logger.info("=" * 80)

    def run_all_experiments(self) -> int:
        """Run all experiments across all runs."""
        logger.info(f"Number of runs: {self.args.n_runs}")
        logger.info(f"Experiments: {', '.join(self.args.experiments)}")
        logger.info(f"Total: {self.total_count} experiments to run")

        if self.args.baseline_only:
            logger.info("Pipeline: BASELINE ONLY")
        elif self.args.bert_only:
            logger.info("Pipeline: BERT ONLY")
        else:
            logger.info("Pipeline: BASELINE + BERT")

        logger.info(
            f"Overwrite mode: {'ENABLED' if self.args.overwrite else 'DISABLED'}"
        )
        logger.info(f"Failfast mode: {'ENABLED' if self.args.failfast else 'DISABLED'}")
        logger.info("=" * 80)

        for run_number in range(1, self.args.n_runs + 1):
            if self.args.run_id:
                run_id = self.args.run_id
            else:
                run_id = f"run_{run_number:02d}"

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"STARTING RUN {run_number} of {self.args.n_runs}: {run_id}")
            logger.info("=" * 80)

            for experiment in self.args.experiments:
                self.current_count += 1
                success = self.run_single_experiment(experiment, run_id)

                if not success and self.args.failfast:
                    logger.error("")
                    logger.error("=" * 80)
                    logger.error("FAILFAST MODE: Stopping due to failure")
                    logger.error("=" * 80)
                    self.print_summary()
                    return 1

        self.print_summary()
        return 0 if not self.failed_experiments else 1

    def run_single_experiment(self, experiment_name: str, run_id: str) -> bool:
        """Run a single outer experiment with K inner reshuffles."""
        logger.info("")
        logger.info("-" * 80)
        logger.info(
            f"Running experiment {self.current_count} of {self.total_count}: {run_id}/{experiment_name}"
        )
        logger.info(f"  Outer run: {run_id}")
        logger.info(f"  Inner runs: {self.args.inner_runs}")
        logger.info("-" * 80)

        start_time = datetime.now()

        try:
            # STAGE 1: Data Preparation (ONCE per outer run)
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"STAGE 1: Data Preparation for {run_id}/{experiment_name}")
            logger.info("=" * 80)

            # Generate base configs
            self.generate_configs(experiment_name, run_id)

            target_dir = self.experiments_dir / run_id / experiment_name

            # Run data preparation steps (sample, simulate, prepare)
            self.run_data_preparation(experiment_name, run_id, target_dir)

            # STAGE 2: K Inner Runs (reshuffles for variance estimation)
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"STAGE 2: Running {self.args.inner_runs} Inner Reshuffles")
            logger.info("=" * 80)

            for k in range(1, self.args.inner_runs + 1):
                inner_id = f"k_{k:02d}"
                logger.info("")
                logger.info("-" * 80)
                logger.info(f"Inner run {k}/{self.args.inner_runs}: {inner_id}")
                logger.info("-" * 80)

                # Run baseline pipeline
                if not self.args.bert_only:
                    self.run_baseline_inner(
                        experiment_name, run_id, target_dir, inner_id, k
                    )

                # Run BERT pipeline
                if not self.args.baseline_only:
                    self.run_bert_inner(
                        experiment_name, run_id, target_dir, inner_id, k
                    )

            # Success
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info("")
            logger.info("=" * 80)
            logger.info(
                f"✓ SUCCESS: {run_id}/{experiment_name} (duration: {duration:.1f}s)"
            )
            logger.info(f"  Completed {self.args.inner_runs} inner runs")
            logger.info("=" * 80)
            self.success_count += 1
            return True

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(
                f"✗ FAILED: {run_id}/{experiment_name} (duration: {duration:.1f}s)"
            )
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())

            self.failed_experiments.append(f"{run_id}/{experiment_name}")
            return False

    def generate_configs(self, experiment_name: str, run_id: str):
        """Generate experiment configs."""
        logger.info("Step: Generating experiment configs...")

        script_dir = Path("experiments/causal_pipeline_resample")

        generate_experiment_configs(
            experiment_name=experiment_name,
            script_dir=script_dir,
            run_id=run_id,
            experiments_dir=str(self.experiments_dir),
            meds_data=self.args.meds,
            features_data=self.args.features,
            tokenized_data=self.args.tokenized,
            pretrain_model=self.args.pretrain_model,
            base_seed=self.args.base_seed,
            sample_fraction=self.args.sample_fraction,
            sample_size=self.args.sample_size,
            base_configs_dir=self.args.base_configs_dir,
        )

        logger.info("✓ Configs generated successfully")

    def run_data_preparation(self, experiment_name: str, run_id: str, target_dir: Path):
        """Run data preparation steps (Stage 1 - once per outer run)."""
        logger.info("Stage 1: Sampling, Simulation, and Data Preparation")

        # Simulate outcomes with sampling
        self.run_step(
            step_name="simulate_outcomes_with_sampling",
            main_func=main_simulate,
            config_name="simulation",
            check_file=target_dir / "simulated_outcomes" / "counterfactuals.csv",
            experiment_name=experiment_name,
        )

        # Select cohort
        self.run_step(
            step_name="select_cohort",
            main_func=main_select_cohort,
            config_name="select_cohort",
            check_file=target_dir / "cohort" / "pids.pt",
            experiment_name=experiment_name,
        )

        # Prepare finetune data (creates base folds)
        self.run_step(
            step_name="prepare_finetune_data",
            main_func=main_prepare_finetune,
            config_name="prepare_finetune",
            check_file=target_dir
            / "prepared_data"
            / "folds.pt",  # Check for folds.pt not just patients.pt
            experiment_name=experiment_name,
        )

        logger.info("✓ Stage 1 complete: Data ready for inner runs")

    def run_baseline_inner(
        self, experiment_name: str, run_id: str, target_dir: Path, inner_id: str, k: int
    ):
        """Run baseline (CatBoost) pipeline for one inner reshuffle."""
        logger.info("Baseline Pipeline (with reshuffle)")

        # Inner directory for this reshuffle
        inner_dir = target_dir / "reshuffles" / inner_id

        # Generate config with reshuffle enabled
        config_name_suffix = f"_{inner_id}"
        self.generate_inner_config(
            experiment_name, run_id, "train_baseline", inner_dir, k, config_name_suffix
        )
        self.generate_inner_config(
            experiment_name, run_id, "calibrate", inner_dir, k, config_name_suffix
        )
        self.generate_inner_config(
            experiment_name, run_id, "estimate", inner_dir, k, config_name_suffix
        )

        # Train baseline
        self.run_step(
            step_name=f"train_baseline ({inner_id})",
            main_func=main_baseline,
            config_name=f"train_baseline{config_name_suffix}",
            check_file=inner_dir / "models" / "baseline" / "combined_predictions.csv",
            experiment_name=experiment_name,
        )

        # Calibrate (baseline)
        self.run_step(
            step_name=f"calibrate_baseline ({inner_id})",
            main_func=main_calibrate,
            config_name=f"calibrate{config_name_suffix}",
            check_file=inner_dir
            / "models"
            / "baseline"
            / "calibrated"
            / "combined_calibrated_predictions.csv",
            experiment_name=experiment_name,
        )

        # Estimate (baseline)
        self.run_step(
            step_name=f"estimate_baseline ({inner_id})",
            main_func=main_estimate,
            config_name=f"estimate{config_name_suffix}",
            check_file=inner_dir / "estimate" / "baseline" / "estimate_results.csv",
            experiment_name=experiment_name,
        )

    def run_bert_inner(
        self, experiment_name: str, run_id: str, target_dir: Path, inner_id: str, k: int
    ):
        """Run BERT pipeline for one inner reshuffle."""
        logger.info("BERT Pipeline (with reshuffle)")

        # Inner directory for this reshuffle
        inner_dir = target_dir / "reshuffles" / inner_id

        # Generate config with reshuffle enabled
        config_name_suffix = f"_{inner_id}"
        self.generate_inner_config(
            experiment_name, run_id, "finetune_bert", inner_dir, k, config_name_suffix
        )
        self.generate_inner_config(
            experiment_name, run_id, "calibrate_bert", inner_dir, k, config_name_suffix
        )
        self.generate_inner_config(
            experiment_name, run_id, "estimate_bert", inner_dir, k, config_name_suffix
        )

        # Finetune BERT
        self.run_step(
            step_name=f"finetune_bert ({inner_id})",
            main_func=main_finetune,
            config_name=f"finetune_bert{config_name_suffix}",
            check_file=inner_dir / "models" / "bert" / "combined_predictions.csv",
            experiment_name=experiment_name,
        )

        # Calibrate (BERT)
        self.run_step(
            step_name=f"calibrate_bert ({inner_id})",
            main_func=main_calibrate,
            config_name=f"calibrate_bert{config_name_suffix}",
            check_file=inner_dir
            / "models"
            / "bert"
            / "calibrated"
            / "combined_calibrated_predictions.csv",
            experiment_name=experiment_name,
        )

        # Estimate (BERT)
        self.run_step(
            step_name=f"estimate_bert ({inner_id})",
            main_func=main_estimate,
            config_name=f"estimate_bert{config_name_suffix}",
            check_file=inner_dir / "estimate" / "bert" / "estimate_results.csv",
            experiment_name=experiment_name,
        )

    def generate_inner_config(
        self,
        experiment_name: str,
        run_id: str,
        base_config_name: str,
        inner_dir: Path,
        k: int,
        config_name_suffix: str,
    ):
        """Generate config for an inner run with updated paths and reshuffle settings."""
        import yaml

        # Load the base config
        base_config_path = (
            Path("experiments/causal_pipeline_resample/generated_configs")
            / experiment_name
            / f"{base_config_name}.yaml"
        )

        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Update paths to point to inner directory
        if "paths" in config:
            # For finetune/train: update model output path
            if "model" in config["paths"]:
                # Replace the main output directory with reshuffles/k_XX
                orig_model_path = Path(config["paths"]["model"])
                # Get relative path after experiment directory
                parts = list(orig_model_path.parts)
                # Find "models" index and replace everything before it with inner_dir
                if "models" in parts:
                    models_idx = parts.index("models")
                    new_path = inner_dir / Path(*parts[models_idx:])
                    config["paths"]["model"] = str(new_path)

            # For calibrate: update input (finetune_model) and output paths
            if "finetune_model" in config["paths"]:
                orig_path = Path(config["paths"]["finetune_model"])
                parts = list(orig_path.parts)
                if "models" in parts:
                    models_idx = parts.index("models")
                    new_path = inner_dir / Path(*parts[models_idx:])
                    config["paths"]["finetune_model"] = str(new_path)

            if "calibrated_predictions" in config["paths"]:
                orig_path = Path(config["paths"]["calibrated_predictions"])
                parts = list(orig_path.parts)
                if "models" in parts:
                    models_idx = parts.index("models")
                    new_path = inner_dir / Path(*parts[models_idx:])
                    config["paths"]["calibrated_predictions"] = str(new_path)

            # For estimate: update input (calibrated) and output paths
            if "estimate" in config["paths"]:
                orig_path = Path(config["paths"]["estimate"])
                parts = list(orig_path.parts)
                if "estimate" in parts:
                    est_idx = parts.index("estimate")
                    new_path = inner_dir / Path(*parts[est_idx:])
                    config["paths"]["estimate"] = str(new_path)

        # Enable reshuffling for finetune/train configs
        if "train_baseline" in base_config_name or "finetune" in base_config_name:
            if "data" not in config:
                config["data"] = {}
            config["data"]["reshuffle"] = True
            # Don't set reshuffle_seed - let it auto-generate from time for randomness

        # Update logging paths to include inner_id (prevents overwriting outputs)
        if "logging" in config and isinstance(config["logging"], dict):
            if "path" in config["logging"]:
                orig_log_path = config["logging"]["path"]
                # Append inner_id to create separate log directories per inner run
                # Extract inner_id from config_name_suffix (e.g., "_k_01" -> "k_01")
                inner_id_str = config_name_suffix.lstrip("_")
                # e.g., "./outputs/logs" -> "./outputs/logs/k_01"
                config["logging"]["path"] = f"{orig_log_path}/{inner_id_str}"

        # Save the modified config
        output_config_path = (
            Path("experiments/causal_pipeline_resample/generated_configs")
            / experiment_name
            / f"{base_config_name}{config_name_suffix}.yaml"
        )
        output_config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.debug(f"Generated inner config: {output_config_path}")

    def run_step(
        self,
        step_name: str,
        main_func: callable,
        config_name: str,
        check_file: Path,
        experiment_name: str,
    ):
        """Run a single pipeline step."""
        # Check if already completed
        if not self.args.overwrite and check_file.exists():
            logger.info(f"==== Skipping {step_name} (output already exists) ====")
            return

        logger.info(f"==== Running {step_name}... ====")
        config_path = (
            Path("experiments/causal_pipeline_resample/generated_configs")
            / experiment_name
            / f"{config_name}.yaml"
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Call the main function
        main_func(str(config_path))

        logger.info(f"==== Success: {step_name} completed ====")

    def print_summary(self):
        """Print final summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY: Multiple Experiments Completed")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {self.total_count}")
        logger.info(f"Successful: {self.success_count}")
        logger.info(f"Failed: {len(self.failed_experiments)}")

        if self.failed_experiments:
            logger.info("")
            logger.info("Failed experiments:")
            for exp in self.failed_experiments:
                logger.info(f"  - {exp}")
        else:
            logger.info("")
            logger.info("All experiments completed successfully!")

        logger.info("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multiple causal pipeline experiments with resampling"
    )

    # Positional arguments
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Names of experiments to run",
    )

    # Run configuration
    parser.add_argument(
        "--n_runs",
        "-n",
        type=int,
        default=1,
        help="Number of outer runs to execute (default: 1)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Specific run ID to use (overrides --n_runs)",
    )
    parser.add_argument(
        "--inner_runs",
        "-k",
        dest="inner_runs",
        type=int,
        default=1,
        help="Number of inner reshuffles per outer run for variance estimation (default: 1)",
    )

    # Pipeline mode
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline (CatBoost) pipeline",
    )
    parser.add_argument(
        "--bert-only",
        action="store_true",
        help="Run only BERT pipeline (requires baseline data)",
    )

    # Data paths
    parser.add_argument(
        "--meds",
        required=True,
        help="Path to MEDS data directory",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to features directory",
    )
    parser.add_argument(
        "--tokenized",
        required=True,
        help="Path to tokenized data directory",
    )
    parser.add_argument(
        "--pretrain-model",
        dest="pretrain_model",
        required=True,
        help="Path to pretrained BERT model",
    )

    # Output configuration
    parser.add_argument(
        "--experiment-dir",
        "-e",
        dest="experiment_dir",
        default="./outputs/causal/sim_study_sampling/runs",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--base-configs-dir",
        dest="base_configs_dir",
        default=None,
        help="Custom base configs directory",
    )

    # Sampling configuration
    parser.add_argument(
        "--base-seed",
        dest="base_seed",
        type=int,
        default=42,
        help="Base seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--sample-fraction",
        dest="sample_fraction",
        type=float,
        default=None,
        help="Fraction of patients to sample (0 < F <= 1)",
    )
    parser.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=None,
        help="Absolute number of patients to sample",
    )

    # Execution options
    parser.add_argument(
        "--timeout-factor",
        dest="timeout_factor",
        type=float,
        default=1.0,
        help="Timeout scaling factor (not used in Python version, kept for compatibility)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run all steps",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop immediately if any experiment fails",
    )

    args = parser.parse_args()

    # Validation
    if args.baseline_only and args.bert_only:
        parser.error("Cannot specify both --baseline-only and --bert-only")

    if args.sample_fraction is None and args.sample_size is None:
        parser.error("Either --sample-fraction or --sample-size must be provided")

    if args.sample_fraction is not None and args.sample_size is not None:
        parser.error("Cannot specify both --sample-fraction and --sample-size")

    if args.run_id:
        args.n_runs = 1

    return args


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        runner = ExperimentRunner(args)
        exit_code = runner.run_all_experiments()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
