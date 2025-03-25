"""
This script is used to map rare codes to more common parent codes or group labels.
"""

import json
import logging
import os
from os.path import join

import pandas as pd
import yaml

from corebehrt.constants.helper import (
    CODE_COUNTS_FILE_NAME,
    RARE_CODE_MAPPING_FILE_NAME,
)
from corebehrt.functional.helpers.rare_code_mapping import group_rare_codes
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/helper/map_rare_codes.yaml"

logger = logging.getLogger("map_rare_codes")


def main(config_path):
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_logging("map_rare_codes")
    with open(
        join(cfg.paths.code_counts, cfg.get("file", CODE_COUNTS_FILE_NAME)), "r"
    ) as f:
        code_counts = json.load(f)

    code_counts = pd.Series(code_counts)
    logger.info(f"Before mapping: {len(code_counts)} unique codes")
    rare_code_counts = code_counts[code_counts < cfg.threshold]
    rare_code_counts = rare_code_counts.to_dict()
    logger.info(f"Found {len(rare_code_counts)} rare codes")

    mapping = group_rare_codes(
        rare_code_counts, cfg.threshold, cfg.hierarchical_pattern, cfg.separator
    )

    # Apply mapping to get mapped codes
    mapped_codes = set()
    for code in code_counts.index:
        mapped_code = mapping.get(code, code)  # Use original if not in mapping
        mapped_codes.add(mapped_code)

    logger.info(f"After mapping: {len(mapped_codes)} unique codes")

    os.makedirs(cfg.paths.mapping, exist_ok=True)
    # Save config for reproducibility
    with open(join(cfg.paths.mapping, "config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)
    with open(join(cfg.paths.mapping, RARE_CODE_MAPPING_FILE_NAME), "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
