import logging

from corebehrt.modules.setup.causal.initializer import CausalInitializer
from corebehrt.modules.setup.manager import ModelManager

logger = logging.getLogger(__name__)


class CausalModelManager(ModelManager):
    """Manager for initializing model, optimizer and scheduler."""

    def initialize_finetune_model(self, checkpoint, outcomes):
        logger.info("Initializing model")
        self.initializer = CausalInitializer(
            self.cfg, checkpoint=checkpoint, model_path=self.checkpoint_model_path
        )
        model = self.initializer.initialize_finetune_model(outcomes)
        return model
