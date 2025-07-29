from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.causal.directory import CausalDirectoryPreparer
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.simulation.simulator_all import CausalSimulator
from corebehrt.modules.simulation.config_all import SimulationConfig
from collections import defaultdict
import pandas as pd
from os.path import join
import logging
from tqdm import tqdm

logger = logging.getLogger("simulate")


CONFIG_PATH = "./corebehrt/configs/causal/simulate.yaml"


def main_simulate(config_path):
    cfg = load_config(config_path)

    # Setup directories
    CausalDirectoryPreparer(cfg).setup_simulate_from_sequence()

    shard_loader = ShardLoader(cfg.paths.data, cfg.paths.splits)
    simulation_config = SimulationConfig(cfg)
    simulator = CausalSimulator(simulation_config)
    simulate(shard_loader, simulator, cfg.paths.outcomes)


def simulate(shard_loader: ShardLoader, simulator: CausalSimulator, outcomes_dir: str):
    """
    Simulates outcomes by processing data shards.

    Iterates through each data shard, applies the simulator, aggregates
    the results, and saves each outcome type to a separate CSV file.
    """
    simulated_outcomes = defaultdict(list)
    for shard, _ in tqdm(shard_loader(), desc="loop shards"):
        simulated_temp = simulator.simulate_dataset(shard)
        for k, df in simulated_temp.items():
            simulated_outcomes[k].append(df)
    for k, df_list in simulated_outcomes.items():
        df = pd.concat(df_list)
        df.to_csv(join(outcomes_dir, f"{k}.csv"), index=False)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_simulate(args.config_path)
