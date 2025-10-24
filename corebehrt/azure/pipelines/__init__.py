from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from corebehrt.azure.pipelines.FINETUNE_CALIBRATE_ESTIMATE import (
    FINETUNE_CALIBRATE_ESTIMATE,
)
from corebehrt.azure.pipelines.FINETUNE_ESTIMATE import FINETUNE_ESTIMATE
from corebehrt.azure.pipelines.FINETUNE_ESTIMATE_SIMULATED import (
    FINETUNE_ESTIMATE_SIMULATED,
)

PIPELINE_REGISTRY = [
    E2E,
    FINETUNE,
    FINETUNE_CALIBRATE_ESTIMATE,
    FINETUNE_ESTIMATE,
    FINETUNE_ESTIMATE_SIMULATED,
]
