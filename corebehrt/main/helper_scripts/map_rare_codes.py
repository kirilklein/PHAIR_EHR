"""
This script is used to map rare codes to more common parent codes or group labels.
"""

import json
import logging
import os
from os.path import join

import pandas as pd
import torch
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


def main(config_path):
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_logging("map_rare_codes")
    logger = logging.getLogger("map_rare_codes")
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
    # Print a few example mappings
    logger.info("Example mappings:")
    for orig_code, mapped_code in list(mapping.items())[:5]:
        if orig_code != mapped_code:
            logger.info(f"  {orig_code} -> {mapped_code}")
    # Save config for reproducibility
    with open(join(cfg.paths.mapping, "config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)
    torch.save(mapping, join(cfg.paths.mapping, RARE_CODE_MAPPING_FILE_NAME))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
