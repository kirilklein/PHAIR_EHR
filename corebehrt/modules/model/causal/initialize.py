import torch
from typing import Dict, List
import logging


def initialize_sigmoid_bias(
    model: torch.nn.Module,
    outcomes: Dict[str, List[int]],
    exposures: List[int],
    logger: logging.Logger,
) -> torch.nn.Module:
    """
    Initialize the final-layer biases of outcome and exposure heads
    based on empirical prevalence.

    This method sets the bias terms of each classification head to
    the log-odds of the observed prevalence in the training data.
    Initializing in this way helps the model start with predictions
    closer to the base rates, which can improve stability and
    convergence early in training.

    Parameters
    ----------
    model : torch.nn.Module
        A model with `outcome_heads` (dict of heads keyed by outcome name)
        and an `exposure_head`. Each head must expose a final classifier
        layer (`.classifier[-1].bias`).
    outcomes : dict[str, array-like]
        Mapping from outcome name to a sequence/array of binary labels
        (0/1). Only outcomes present in this dict will have their biases
        initialized.
    exposures : array-like
        Sequence/array of binary exposure labels (0/1) used to initialize
        the exposure head bias.
    logger : logging.Logger
        Logger used to record initialization progress and prevalence
        statistics.

    Returns
    -------
    model : torch.nn.Module
        The input model with updated bias values for all specified heads.

    Notes
    -----
    - Prevalence values are clipped to the range [1e-7, 1 - 1e-7] to
      avoid numerical issues with log(0) or infinite log-odds.
    - The initialized bias is computed as::

          bias = log(p / (1 - p))

      where `p` is the empirical prevalence of the positive class.
    """
    logger.info("Initializing outcome head biases based on prevalence")
    for outcome_name in model.outcome_names:
        if outcome_name in outcomes:
            outcome_labels = torch.tensor(outcomes[outcome_name], dtype=torch.float32)
            p_mean = outcome_labels.mean().item()

            # Avoid log(0) or log(inf) by clipping prevalence
            p_mean = torch.clamp(torch.tensor(p_mean), min=1e-7, max=1 - 1e-7).item()

            initial_bias = torch.log(torch.tensor(p_mean / (1 - p_mean)))
            model.outcome_heads[outcome_name].classifier[-1].bias.data.fill_(
                initial_bias
            )

            logger.info(
                f"Initialized {outcome_name} bias to {initial_bias:.4f} (prevalence: {p_mean:.4f})"
            )

    # Also initialize exposure head bias if desired
    exposure_labels = torch.tensor(exposures, dtype=torch.float32)
    p_mean_exposure = exposure_labels.mean().item()
    p_mean_exposure = torch.clamp(
        torch.tensor(p_mean_exposure), min=1e-7, max=1 - 1e-7
    ).item()
    initial_bias_exposure = torch.log(
        torch.tensor(p_mean_exposure / (1 - p_mean_exposure))
    )
    model.exposure_head.classifier[-1].bias.data.fill_(initial_bias_exposure)
    logger.info(
        f"Initialized exposure bias to {initial_bias_exposure:.4f} (prevalence: {p_mean_exposure:.4f})"
    )
    return model
