# corebehrt/trainer/freezing_utils.py
from typing import Tuple


def check_task_plateau(
    current_metric: float,
    best_metric: float,
    threshold: float,
    higher_is_better: bool = True,
) -> Tuple[bool, float]:
    """
    Checks if a task's performance metric has plateaued and updates the best metric.

    This is a pure function that checks if the relative improvement between
    the current and best metric is below a given threshold.

    Args:
        current_metric: The metric value from the current epoch.
        best_metric: The best metric value seen so far. Can be None if initializing.
        threshold: The relative improvement threshold to determine a plateau.
        higher_is_better: Set to True if a higher metric value is better (e.g., AUC),
                          and False if a lower value is better (e.g., loss).

    Returns:
        A tuple containing:
        - bool: True if the performance has plateaued, otherwise False.
        - float: The updated best metric value.
    """
    if best_metric is None:
        # Initialize the best metric on the first run
        return False, current_metric

    # Avoid division by zero if the best metric is 0
    if best_metric == 0:
        improvement = current_metric if higher_is_better else -current_metric
    else:
        improvement = (current_metric - best_metric) / abs(best_metric)

    # Determine if a plateau has been reached
    if higher_is_better:
        plateau = improvement < threshold
        new_best = max(current_metric, best_metric)
    else:  # Lower is better
        plateau = improvement > -threshold
        new_best = min(current_metric, best_metric)

    return plateau, new_best
