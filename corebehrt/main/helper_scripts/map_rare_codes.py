"""
This script is used to map rare codes to more common parent codes or group labels.
"""

import json
import logging
import os
from os.path import join

import pandas as pd

from corebehrt.constants.helper import RARE_CODE_MAPPING_FILE_NAME
from corebehrt.functional.helpers.rare_code_mapping import group_rare_codes
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/helper/map_rare_codes.yaml"

logger = logging.getLogger("map_rare_codes")


def main(config_path):
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_logging("map_rare_codes")
    with open(cfg.paths.code_counts, "r") as f:
        code_counts = json.load(f)

    code_counts = pd.Series(code_counts)
    code_counts = code_counts[code_counts < cfg.threshold]
    code_counts = code_counts.to_dict()
    logger.info(f"Found {len(code_counts)} rare codes")

    mapping = group_rare_codes(
        code_counts, cfg.threshold, cfg.hierarchical_pattern, cfg.separator
    )
    os.makedirs(cfg.paths.mapping, exist_ok=True)
    with open(join(cfg.paths.mapping, RARE_CODE_MAPPING_FILE_NAME), "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main(args.config_path)
