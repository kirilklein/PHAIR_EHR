from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.simulation.semisynthetic_simulator import (
    SemiSyntheticCausalSimulator as CausalSimulator,
)
from corebehrt.modules.simulation.config_semisynthetic import (
    create_semisynthetic_config,
)
from corebehrt.functional.causal.cohort_sampler import sample_cohort
from corebehrt.constants.data import PID_COL
from corebehrt.constants.paths import PID_FILE
from collections import defaultdict
import os
import pandas as pd
from os.path import join
import logging
from tqdm import tqdm
import torch

logger = logging.getLogger("simulate")


CONFIG_PATH = "./corebehrt/configs/causal/simulate_semisynthetic.yaml"


def main_simulate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_simulate_from_sequence()

    shard_loader = ShardLoader(cfg.paths.data, cfg.paths.splits)
    simulation_config = create_semisynthetic_config(cfg)
    simulator = CausalSimulator(simulation_config)

    # Optionally restrict to a sampled subset of patients (e.g. for smoke tests
    # or per-run resampling). Disabled by default, so default behaviour is unchanged.
    sampling_cfg = cfg.get("sampling", {})
    if sampling_cfg.get("enabled", False):
        shard_loader = _sample_patients(
            shard_loader, sampling_cfg, cfg.get("seed", 42), cfg.paths.get("cohort")
        )

    # Pass 1: compute global feature means/stds for standardization
    logger.info("--- Pass 1: computing global feature statistics ---")
    simulator.compute_global_feature_stats(shard_loader)

    # Pass 2: simulate outcomes using globally standardized features
    simulate(shard_loader, simulator, cfg.paths.outcomes)


class SampledShardLoader:
    """Wraps a ShardLoader and filters every shard to a fixed set of patient IDs."""

    def __init__(self, shard_loader: ShardLoader, pids_set: set):
        self.shard_loader = shard_loader
        self.pids_set = pids_set

    def __call__(self):
        for shard, meta in self.shard_loader():
            yield shard[shard[PID_COL].isin(self.pids_set)], meta


def _sample_patients(
    shard_loader: ShardLoader, sampling_cfg: dict, seed: int, cohort_dir: str
) -> SampledShardLoader:
    """Sample a subset of patient IDs and return a shard loader filtered to them."""
    all_pids = set()
    for shard, _ in tqdm(shard_loader(), desc="Scanning shards for sampling"):
        all_pids.update(shard[PID_COL].unique())
    full_pids = torch.tensor(sorted(all_pids))

    sampled_pids = sample_cohort(
        full_pids,
        sample_fraction=sampling_cfg.get("fraction"),
        sample_size=sampling_cfg.get("size"),
        seed=seed,
    )
    logger.info(
        f"Sampled {len(sampled_pids)} of {len(full_pids)} patients (seed={seed})"
    )

    if cohort_dir:
        os.makedirs(cohort_dir, exist_ok=True)
        torch.save(sampled_pids, join(cohort_dir, PID_FILE))

    return SampledShardLoader(shard_loader, set(sampled_pids.tolist()))


def simulate(shard_loader: ShardLoader, simulator: CausalSimulator, outcomes_dir: str):
    """
    Simulates outcomes by processing data shards in a single pass.

    Iterates through each data shard, calls simulate_dataset,
    aggregates the results, and saves each outcome type to a separate CSV file.
    Then computes stats and plots from the aggregated results.
    """
    logger.info("--- Starting semi-synthetic simulation ---")
    simulated_outcomes = defaultdict(list)
    for shard, _ in tqdm(shard_loader(), desc="Simulating from shards"):
        simulated_temp = simulator.simulate_dataset(shard)
        for k, df in simulated_temp.items():
            if not df.empty:
                simulated_outcomes[k].append(df)

    logger.info("--- Simulation complete, saving results ---")

    aggregated = {}
    for k, df_list in simulated_outcomes.items():
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            df.to_csv(join(outcomes_dir, f"{k}.csv"), index=False)
            aggregated[k] = df

    if "counterfactuals" in aggregated and "ite" in aggregated:
        logger.info("--- Computing stats and plots from aggregated results ---")
        simulator.finalize(aggregated["counterfactuals"], aggregated["ite"])


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
