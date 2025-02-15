from typing import List, Dict


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
