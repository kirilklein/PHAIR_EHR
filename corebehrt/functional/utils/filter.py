from typing import List, Dict, Iterable
import pandas as pd


def filter_folds_by_pids(folds: List[Dict], available_pids: List[str]) -> List[Dict]:
    """Filter folds to only include available PIDs.

    Args:
        folds: List of dictionaries containing train/val splits
        available_pids: Set of PIDs that are available in the dataset

    Returns:
        List of dictionaries with filtered train/val splits
    """
    available_pids = set(available_pids)
    filtered_folds = []
    for fold in folds:
        filtered_fold = {
            key: [pid for pid in pids if pid in available_pids]
            for key, pids in fold.items()
        }
        filtered_folds.append(filtered_fold)
    return filtered_folds


def safe_control_pids(
    all_pids: Iterable,
    exposed_pids: Iterable,
    drop_duplicates: bool = True,
    preserve_order: bool = True,
) -> List:
    """
    Return control PIDs = all_pids \ exposed_pids, while:
      - preserving the dtype and exact values of all_pids
      - aligning exposed to all_pids' dtype for comparison
      - preserving order (default) and optionally removing duplicates
    """
    all_idx = pd.Index(all_pids)
    exp_idx = pd.Index(exposed_pids)

    # Align datatypes for comparison while preserving original all_idx
    cmp_all, cmp_exp = _align_indices_for_comparison(all_idx, exp_idx)

    # Build mask and filter
    if preserve_order:
        mask = ~cmp_all.isin(cmp_exp)
        result = all_idx[mask]
        if drop_duplicates:
            result = result.drop_duplicates()
        return result.tolist()

    # Non-order-preserving variant
    mask = ~cmp_all.isin(cmp_exp)
    result = all_idx[mask]
    if drop_duplicates:
        result = result.drop_duplicates()
    return result.tolist()


def _align_indices_for_comparison(
    all_idx: pd.Index, exp_idx: pd.Index
) -> tuple[pd.Index, pd.Index]:
    """
    Align two indices for comparison. Strategy:
    1. Try to convert both to int
    2. If that fails, convert both to string
    3. If that fails, fall back to object
    """
    # Strategy 1: Try to convert both to int
    try:
        cmp_all = _convert_to_int(all_idx)
        cmp_exp = _convert_to_int(exp_idx)
        return cmp_all, cmp_exp
    except (TypeError, ValueError):
        pass

    # Strategy 2: Try to convert both to string
    try:
        cmp_all = _convert_to_string(all_idx)
        cmp_exp = _convert_to_string(exp_idx)
        return cmp_all, cmp_exp
    except (TypeError, ValueError):
        pass

    # Strategy 3: Fallback to object dtype
    return all_idx.astype("object"), exp_idx.astype("object")


def _convert_to_int(idx: pd.Index) -> pd.Index:
    """Convert index to int, raising exception if not possible."""
    return pd.Index([int(x) for x in idx])


def _convert_to_string(idx: pd.Index) -> pd.Index:
    """Convert index to string."""
    return pd.Index([str(x) for x in idx])
