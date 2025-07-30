from collections import defaultdict
from os.path import join

import pandas as pd
from tqdm import tqdm

from corebehrt.constants.data import PID_COL
from corebehrt.modules.cohort_handling.outcomes import OutcomeMaker
from corebehrt.modules.monitoring.logger import TqdmToLogger


def process_data(loader, cfg, logger) -> None:
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
    header_written = defaultdict(bool)
    outcomes_path = cfg.paths.outcomes
    outcome_maker = OutcomeMaker(cfg.outcomes)

    for concept_batch, patient_batch in tqdm(
        loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
    ):
        pids = concept_batch[PID_COL].unique()
        outcome_maker(concept_batch, patient_batch, pids, outcomes_path, header_written)

    for key in cfg.outcomes:
        if not header_written[key]:
            logger.warning(f"Outcomes table for {key} is empty. No file was created.")
