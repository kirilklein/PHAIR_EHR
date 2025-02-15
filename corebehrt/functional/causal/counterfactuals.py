import torch


def combine_counterfactuals(exposure, exposed_values, control_values):
    """Combines counterfactual values based on exposure status.

    For each individual, returns the opposite of their actual exposure:
    - If exposed (exposure=1), returns their control value
    - If not exposed (exposure=0), returns their exposed value

    Args:
        exposure (numpy.ndarray): Binary array indicating exposure status (1=exposed, 0=control)
        exposed_values (numpy.ndarray): Values under exposure condition
        control_values (numpy.ndarray): Values under control condition

    Returns:
        numpy.ndarray: Combined array where each element is the counterfactual value
            based on the opposite of the actual exposure status
    """
    return torch.where(exposure == 1, control_values, exposed_values)
