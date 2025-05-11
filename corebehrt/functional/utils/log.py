import logging

import pandas as pd


def log_table(table: pd.DataFrame, logger: logging.Logger):
    """Log a table. To be used for logging formatted tables."""
    logger.info("\n" + table.to_string())
