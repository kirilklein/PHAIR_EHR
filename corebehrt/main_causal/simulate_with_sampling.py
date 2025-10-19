"""
Simulation with per-run cohort sampling for resampling experiments.

This module extends the standard simulation workflow by first sampling a subset
of patients from MEDS data, then simulating outcomes only for those sampled patients.
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

    # Check sampling configuration
    sampling_cfg = cfg.get("sampling", {})
    if not sampling_cfg.get("enabled", False):
        raise ValueError(
            "Sampling must be enabled in config for simulate_with_sampling. "
            "Set sampling.enabled: true in config."
        )

    # Initialize shard loader to get all available PIDs from MEDS data
    shard_loader = ShardLoader(cfg.paths.data, cfg.paths.splits)

    # Extract all unique PIDs from MEDS data
    logger.info("Extracting all patient IDs from MEDS data...")
    all_pids = set()
    for shard, _ in tqdm(shard_loader(), desc="Scanning MEDS shards"):
        all_pids.update(shard[PID_COL].unique())

    full_pids = torch.tensor(sorted(all_pids))
    logger.info(f"Found {len(full_pids)} unique patients in MEDS data")

    # Sample subset of PIDs (either by size or fraction)
    sample_size = sampling_cfg.get("size", None)
    sample_fraction = sampling_cfg.get("fraction", None)
    seed = cfg.get("seed", 42)

    if sample_size is not None:
        logger.info(f"Sampling {sample_size} patients with seed {seed}")
        sampled_pids = sample_cohort(full_pids, sample_size=sample_size, seed=seed)
    elif sample_fraction is not None:
        logger.info(
            f"Sampling {sample_fraction * 100:.1f}% of patients with seed {seed}"
        )
        sampled_pids = sample_cohort(
            full_pids, sample_fraction=sample_fraction, seed=seed
        )
    else:
        raise ValueError(
            "Either 'sampling.size' or 'sampling.fraction' must be specified in config"
        )

    logger.info(f"Sampled {len(sampled_pids)} patients from {len(full_pids)} total ({len(sampled_pids)/len(full_pids)*100:.1f}%)")

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

    # Re-initialize shard loader for simulation (need fresh iterator)
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
    logger.info(f"Total patients to simulate: {len(sampled_pids_set)}")
    simulated_outcomes = defaultdict(list)
    total_shards_processed = 0
    total_patients_in_shards = 0

    for shard, _ in tqdm(shard_loader(), desc="Simulating from shards"):
        total_shards_processed += 1
        # Filter shard to only include sampled patients
        mask = shard[PID_COL].isin(sampled_pids_set)
        filtered_shard = shard[mask]

        if len(filtered_shard) == 0:
            continue

        sampled_patients_in_shard = filtered_shard[PID_COL].nunique()
        total_patients_in_shards += sampled_patients_in_shard
        logger.debug(f"Shard {total_shards_processed}: {sampled_patients_in_shard} sampled patients present")

        # Simulate on filtered shard
        simulated_temp = simulator.simulate_dataset(filtered_shard)
        for k, df in simulated_temp.items():
            if not df.empty:
                simulated_outcomes[k].append(df)

    logger.info("--- Simulation complete, saving results ---")
    logger.info("=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Shards processed: {total_shards_processed}")
    logger.info(f"Patient appearances across shards: {total_patients_in_shards}")
    logger.info(f"Note: Patients may appear in multiple shards")

    # Track unique patients across all outputs
    total_simulated = 0
    
    for k, df_list in simulated_outcomes.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            unique_pids = df[PID_COL].nunique()
            df.to_csv(join(outcomes_dir, f"{k}.csv"), index=False)
            logger.info(f"Saved {len(df)} rows to {k}.csv (unique patients: {unique_pids})")
            
            # Track total from a patient-level output
            if k in ['counterfactuals', 'ite'] and total_simulated == 0:
                total_simulated = unique_pids

    if total_simulated > 0:
        logger.info("=" * 60)
        logger.info(f"Final simulated patients: {total_simulated}")
        logger.info(f"Patient retention: {total_simulated}/{len(sampled_pids_set)} ({total_simulated/len(sampled_pids_set)*100:.1f}%)")
        logger.info("=" * 60)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
