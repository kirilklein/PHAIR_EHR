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

CONFIG_PATH = "./corebehrt/configs/causal/simulate_realistic_with_sampling.yaml"


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

    logger.info(
        f"Sampled {len(sampled_pids)} patients from {len(full_pids)} total ({len(sampled_pids) / len(full_pids) * 100:.1f}%)"
    )

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
    logger.info(
        f"Target: {len(sampled_pids_set)} sampled patients to find across all shards"
    )
    simulated_outcomes = defaultdict(list)
    total_shards_processed = 0
    cumulative_patients_found = set()  # Track unique patients found across all shards
    cumulative_patients_simulated = (
        set()
    )  # Track unique patients successfully simulated

    for shard, _ in tqdm(shard_loader(), desc="Simulating from shards"):
        total_shards_processed += 1
        # Filter shard to only include sampled patients
        mask = shard[PID_COL].isin(sampled_pids_set)
        filtered_shard = shard[mask]

        if len(filtered_shard) == 0:
            continue

        # Track how many target patients are in this shard
        sampled_patients_in_shard = set(filtered_shard[PID_COL].unique())
        cumulative_patients_found.update(sampled_patients_in_shard)
        found_pct = (len(cumulative_patients_found) / len(sampled_pids_set)) * 100

        logger.debug(
            f"Shard {total_shards_processed}: Found {len(sampled_patients_in_shard)} of target patients | Cumulative: {len(cumulative_patients_found)}/{len(sampled_pids_set)} ({found_pct:.1f}%)"
        )

        # Simulate on filtered shard
        simulated_temp = simulator.simulate_dataset(filtered_shard)

        # Track simulated patients
        if simulated_temp:
            for k, df in simulated_temp.items():
                if not df.empty:
                    simulated_outcomes[k].append(df)
                    # Track unique simulated patients from a patient-level output
                    if k in ["counterfactuals", "ite"]:
                        cumulative_patients_simulated.update(df[PID_COL].unique())

    # Final summary after all shards processed
    logger.info(f"Processed {total_shards_processed} shards")
    logger.info(
        f"Found {len(cumulative_patients_found)}/{len(sampled_pids_set)} target patients across all shards ({len(cumulative_patients_found) / len(sampled_pids_set) * 100:.1f}%)"
    )
    if len(cumulative_patients_simulated) > 0:
        logger.info(
            f"Successfully simulated {len(cumulative_patients_simulated)} patients after filtering ({len(cumulative_patients_simulated) / len(cumulative_patients_found) * 100:.1f}% of found patients)"
        )

    logger.info("--- Simulation complete, saving results ---")
    logger.info("=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Sampled patients (target): {len(sampled_pids_set)}")

    found_total = len(cumulative_patients_found)
    found_pct = (found_total / len(sampled_pids_set) * 100) if found_total else 0.0
    logger.info(f"Patients found in shards: {found_total} ({found_pct:.1f}%)")
    simulated_total = len(cumulative_patients_simulated)
    simulated_pct_of_found = (
        (simulated_total / found_total) * 100 if found_total else 0.0
    )
    logger.info(
        f"Patients after filtering: {simulated_total} ({simulated_pct_of_found:.1f}% of found)"
    )

    logger.info(
        f"Overall retention: {simulated_total}/{len(sampled_pids_set)} ({simulated_pct_of_found:.1f}%)"
    )
    logger.info("-" * 60)

    for k, df_list in simulated_outcomes.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            df.to_csv(join(outcomes_dir, f"{k}.csv"), index=False)
            # Only log unique patients if the dataframe has a PID_COL
            if PID_COL in df.columns:
                unique_pids = df[PID_COL].nunique()
                logger.info(
                    f"Saved {len(df)} rows to {k}.csv (unique patients: {unique_pids})"
                )
            else:
                logger.info(f"Saved {len(df)} rows to {k}.csv")

    logger.info("=" * 60)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
