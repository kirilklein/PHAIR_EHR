"""
Initializer for causal inference models.

This module provides initialization functionality for causal inference models,
extending the base initializer to handle both exposure and outcome predictions.
"""

import logging

from corebehrt.modules.model.causal.model import CorebehrtForCausalFineTuning
from corebehrt.modules.setup.initializer import Initializer
from corebehrt.modules.trainer.utils import get_loss_weight

logger = logging.getLogger(__name__)


class CausalInitializer(Initializer):
    """Initializer for causal inference models.

    Extends the base Initializer to handle initialization of models that predict
    both exposure and outcome. Currently only supports loading a pre-trained model from checkpoint.
    """

    def initialize_finetune_model(self, outcomes, exposures):
        if self.checkpoint:
            logger.info("Loading model from checkpoint")
            loss_weight_outcomes = get_loss_weight(self.cfg, outcomes)
            loss_weight_exposures = get_loss_weight(self.cfg, exposures)
            add_config = {
                **self.cfg.model,
                "pos_weight_outcomes": loss_weight_outcomes,
                "pos_weight_exposures": loss_weight_exposures,
            }
            model = self.loader.load_model(
                CorebehrtForCausalFineTuning,
                checkpoint=self.checkpoint,
                add_config=add_config,
            )
            model.to(self.device)
            return model
        else:
            raise NotImplementedError("Fine-tuning from scratch is not supported.")
