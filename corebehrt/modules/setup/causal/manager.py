import logging
from typing import Dict, List

from corebehrt.modules.setup.causal.initializer import CausalInitializer
from corebehrt.modules.setup.manager import ModelManager

logger = logging.getLogger(__name__)


def _print_finetune_label_diagnostics(
    outcomes: Dict[str, List[int]], exposures: List[int]
) -> None:
    """Stdout diagnostics for Azure/remote jobs where log config may hide tracebacks."""
    print("[CausalModelManager] finetune label diagnostics:", flush=True)
    for name, vals in outcomes.items():
        uniq = sorted(set(vals))
        print(
            f"  outcome {name!r}: n={len(vals)} unique_values={uniq}",
            flush=True,
        )
    exp_uniq = sorted(set(exposures))
    print(
        f"  exposures: n={len(exposures)} unique_values={exp_uniq}",
        flush=True,
    )


class CausalModelManager(ModelManager):
    """Manager for initializing model, optimizer and scheduler."""

    def initialize_finetune_model(
        self, checkpoint, outcomes: Dict[str, List[int]], exposures: List[int]
    ):
        logger.info("Initializing model")
        self.initializer = CausalInitializer(
            self.cfg, checkpoint=checkpoint, model_path=self.checkpoint_model_path
        )
        _print_finetune_label_diagnostics(outcomes, exposures)
        try:
            model = self.initializer.initialize_finetune_model(outcomes, exposures)
        except Exception as e:
            print(
                f"[CausalModelManager] initialize_finetune_model FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            logger.exception("initialize_finetune_model failed")
            raise
        return model
