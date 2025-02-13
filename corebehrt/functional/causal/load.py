from os.path import join
from typing import Tuple

import numpy as np
import torch

from corebehrt.constants.paths import ENCODINGS_FILE, PID_FILE


def load_encodings_and_pids_from_encoded_dir(
    encoded_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load encodings and patient IDs from the encoded directory.

    Args:
        encoded_dir: Path to directory containing encoded data files

    Returns:
        Tuple of (encodings array, patient IDs array)
    """
    encodings = torch.load(join(encoded_dir, ENCODINGS_FILE)).numpy()
    pids = torch.load(join(encoded_dir, PID_FILE))
    return encodings, pids
