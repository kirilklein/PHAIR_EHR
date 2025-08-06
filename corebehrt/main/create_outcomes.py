"""Create tokenized features from formatted data. config template: data.yaml"""

import logging
from os.path import join

from tqdm import tqdm

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.cohort_handling.outcomes import OutcomeMaker
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/outcomes_test.yaml"


def main_data(config_path):
    cfg = load_config(config_path)

    prepper = DirectoryPreparer(cfg)
    prepper.setup_create_outcomes()

    logger = logging.getLogger("create_outcomes")
    logger.info("Starting outcomes creation")
    create_outcomes(
        ShardLoader(
            data_dir=cfg.paths.data,
            splits=cfg.paths.get("splits", None),
            patient_info_path=join(cfg.paths.features, "patient_info.parquet"),
        ),
        cfg,
        logger,
    )

    logger.info("Finish outcomes creation")
    logger.info("Done")


def create_outcomes(loader, cfg, logger) -> None:
    """Process batches of concept and patient data to create outcome tables.
    The outcomes are written to CSV files in an append mode to handle large datasets.

    Args:
        loader: A callable that yields tuples of (concept_batch, patient_batch) DataFrames.
        cfg: Configuration object containing outcome settings and paths.
        logger: Logger object for tracking progress.

    Note:
        The function processes data in batches to handle large datasets efficiently.
        It uses the OutcomeMaker class to generate outcome tables for each batch,
        then appends the results to CSV files. This avoids holding all outcomes
        in memory.
    """
    outcomes_path = cfg.paths.outcomes
    outcome_maker = OutcomeMaker(cfg.outcomes)

    for concept_batch, _ in tqdm(
        loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
    ):
        outcome_maker(concept_batch, outcomes_path)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
