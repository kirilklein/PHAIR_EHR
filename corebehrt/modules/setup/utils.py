import numpy as np
import torch


def add_safe_globals():
    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Int64DType]
    )
