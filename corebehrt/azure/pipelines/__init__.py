from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from corebehrt.azure.pipelines.ESTIMATE import FINETUNE_ESTIMATE

PIPELINE_REGISTRY = [E2E, FINETUNE, FINETUNE_ESTIMATE]
