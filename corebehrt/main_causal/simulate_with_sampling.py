"""
Simulation with per-run cohort sampling for resampling experiments.

This module extends the standard simulation workflow by first sampling a subset
of patients from a pre-built base cohort, then simulating outcomes only for
those sampled patients.
"""

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.simulation.realistic_simulator import (
    RealisticCausalSimulator as CausalSimulator,
)
from corebehrt.modules.simulation.config_realistic import create_simulation_config
from corebehrt.functional.causal.cohort_sampler import sample_cohort
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import PID_FILE
from collections import defaultdict
import pandas as pd
from os.path import join
import logging
from tqdm import tqdm
import torch

logger = logging.getLogger("simulate_with_sampling")

CONFIG_PATH = "./corebehrt/configs/causal/simulate_realistic.yaml"


def main_simulate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_simulate_from_sequence()

    # Load base cohort PIDs
    sampling_cfg = cfg.get("sampling", {})
    if not sampling_cfg.get("enabled", False):
        raise ValueError(
            "Sampling must be enabled in config for simulate_with_sampling. "
            "Set sampling.enabled: true in config."
        )

    base_cohort_path = sampling_cfg.get("base_cohort_path")
    if not base_cohort_path:
        raise ValueError("sampling.base_cohort_path must be specified in config")

    # Load full cohort PIDs
    full_pids_file = join(base_cohort_path, PID_FILE)
    logger.info(f"Loading base cohort from {full_pids_file}")
    full_pids = torch.load(full_pids_file)
    logger.info(f"Loaded {len(full_pids)} patients from base cohort")

    # Sample subset of PIDs
    sample_fraction = sampling_cfg.get("fraction", 0.5)
    seed = cfg.get("seed", 42)
    logger.info(f"Sampling {sample_fraction * 100:.1f}% of cohort with seed {seed}")
    sampled_pids = sample_cohort(full_pids, sample_fraction, seed)
    logger.info(f"Sampled {len(sampled_pids)} patients")

    # Save sampled PIDs to cohort directory (will be used by downstream steps)
    cohort_dir = cfg.paths.get("cohort")
    if cohort_dir:
        import os

        os.makedirs(cohort_dir, exist_ok=True)
        sampled_pids_file = join(cohort_dir, PID_FILE)
        torch.save(sampled_pids, sampled_pids_file)
        logger.info(f"Saved sampled PIDs to {sampled_pids_file}")

    # Convert to set for fast lookup
    sampled_pids_set = set(sampled_pids.tolist())

    # Initialize simulation components
    shard_loader = ShardLoader(cfg.paths.data, cfg.paths.splits)
    simulation_config = create_simulation_config(cfg)
    simulator = CausalSimulator(simulation_config)

    # Run simulation with filtering
    simulate_with_filtering(
        shard_loader, simulator, cfg.paths.outcomes, sampled_pids_set
    )


def simulate_with_filtering(
    shard_loader: ShardLoader,
    simulator: CausalSimulator,
    outcomes_dir: str,
    sampled_pids_set: set,
):
    """
    Simulates outcomes by processing data shards, filtering to sampled patients.

    Args:
        shard_loader: Loader for data shards
        simulator: Configured causal simulator
        outcomes_dir: Directory to save simulated outcomes
        sampled_pids_set: Set of sampled patient IDs to include
    """
    logger.info("--- Starting simulation with sampling ---")
    simulated_outcomes = defaultdict(list)

    for shard, _ in tqdm(shard_loader(), desc="Simulating from shards"):
        # Filter shard to only include sampled patients
        mask = shard[PID_COL].isin(sampled_pids_set)
        filtered_shard = shard[mask]

        if len(filtered_shard) == 0:
            continue

        # Simulate on filtered shard
        simulated_temp = simulator.simulate_dataset(filtered_shard)
        for k, df in simulated_temp.items():
            if not df.empty:
                simulated_outcomes[k].append(df)

    logger.info("--- Simulation complete, saving results ---")

    for k, df_list in simulated_outcomes.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            df.to_csv(join(outcomes_dir, f"{k}.csv"), index=False)
            logger.info(f"Saved {len(df)} rows to {k}.csv")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
