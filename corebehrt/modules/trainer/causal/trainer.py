from collections import namedtuple
from typing import List

import torch
import yaml

from corebehrt.constants.causal.data import EXPOSURE_TARGET
from corebehrt.constants.data import TARGET
from corebehrt.modules.monitoring.logger import get_tqdm
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_avg_metrics,
    save_curves,
    save_metrics_to_csv,
    save_predictions,
)
from corebehrt.modules.setup.config import Config
from corebehrt.modules.trainer.trainer import EHRTrainer

yaml.add_representer(Config, lambda dumper, data: data.yaml_repr(dumper))

BEST_MODEL_ID = 999  # For backwards compatibility
DEFAULT_CHECKPOINT_FREQUENCY = 100


class CausalEHRTrainer(EHRTrainer):
    def _evaluate(self, epoch: int, mode="val") -> tuple:
        """Returns the validation/test loss and metrics for both exposure and outcome."""
        if mode == "val":
            if self.val_dataset is None:
                self.log("No validation dataset provided")
                return None, None
            dataloader = self.get_dataloader(self.val_dataset, mode="val")
        elif mode == "test":
            if self.test_dataset is None:
                self.log("No test dataset provided")
                return None, None
            dataloader = self.get_dataloader(self.test_dataset, mode="test")
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'val' or 'test'")

        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(mode)
        loss = 0

        # Separate metric tracking for exposure and outcome
        exposure_metric_values = {f"exposure_{name}": [] for name in self.metrics}
        outcome_metric_values = {f"outcome_{name}": [] for name in self.metrics}

        # Lists to store predictions if accumulating
        exposure_logits_list = [] if self.accumulate_logits else None
        exposure_targets_list = [] if self.accumulate_logits else None
        outcome_logits_list = [] if self.accumulate_logits else None
        outcome_targets_list = [] if self.accumulate_logits else None
        cf_outcome_logits_list = [] if self.accumulate_logits else None

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)
                    cf_outputs = self.model(batch, cf=True)
                # Add to total loss if available
                if hasattr(outputs, "loss"):
                    loss += outputs.loss.item()

                if self.accumulate_logits:
                    # Store exposure predictions
                    exposure_logits_list.append(outputs.exposure_logits.float().cpu())
                    exposure_targets_list.append(batch[EXPOSURE_TARGET].cpu())

                    # Store outcome predictions
                    outcome_logits_list.append(outputs.outcome_logits.float().cpu())
                    outcome_targets_list.append(batch[TARGET].cpu())

                    # Store counterfactual outcome predictions
                    cf_outcome_logits_list.append(
                        cf_outputs.outcome_logits.float().cpu()
                    )
                else:
                    # Calculate metrics on the fly
                    for name, func in self.metrics.items():
                        # Exposure metrics
                        exposure_outputs = namedtuple("Outputs", ["logits"])(
                            outputs.exposure_logits
                        )
                        exposure_batch = {"target": batch[EXPOSURE_TARGET]}
                        exposure_metric_values[f"exposure_{name}"].append(
                            func(exposure_outputs, exposure_batch)
                        )

                        # Outcome metrics
                        outcome_outputs = namedtuple("Outputs", ["logits"])(
                            outputs.outcome_logits
                        )
                        outcome_batch = {"target": batch[TARGET]}
                        outcome_metric_values[f"outcome_{name}"].append(
                            func(outcome_outputs, outcome_batch)
                        )

        # Process all accumulated predictions
        if self.accumulate_logits:
            metrics = self.process_causal_classification_results(
                exposure_logits_list,
                exposure_targets_list,
                outcome_logits_list,
                outcome_targets_list,
                epoch,
                mode=mode,
                cf_outcome_logits=cf_outcome_logits_list,
            )
        else:
            # Average metrics calculated on the fly
            exposure_metrics = compute_avg_metrics(exposure_metric_values)
            outcome_metrics = compute_avg_metrics(outcome_metric_values)
            metrics = {**exposure_metrics, **outcome_metrics}

        self.model.train()

        # Return average loss and all metrics
        return loss / len(loop), metrics

    def process_causal_classification_results(
        self,
        exposure_logits: List[torch.Tensor],
        exposure_targets: List[torch.Tensor],
        outcome_logits: List[torch.Tensor],
        outcome_targets: List[torch.Tensor],
        epoch: int,
        cf_outcome_logits: List[torch.Tensor] = None,
        mode="val",
    ) -> dict:
        """Process results for both exposure and outcome predictions."""
        # Process exposure results
        exposure_targets = torch.cat(exposure_targets)
        exposure_logits = torch.cat(exposure_logits)
        exposure_batch = {"target": exposure_targets}
        exposure_outputs = namedtuple("Outputs", ["logits"])(exposure_logits)

        # Process outcome results
        outcome_targets = torch.cat(outcome_targets)
        outcome_logits = torch.cat(outcome_logits)
        outcome_batch = {"target": outcome_targets}
        outcome_outputs = namedtuple("Outputs", ["logits"])(outcome_logits)

        # Process counterfactual outcome results
        cf_outcome_logits = torch.cat(cf_outcome_logits)

        # Compute metrics for both outcomes
        metrics = {}

        # Exposure metrics
        for name, func in self.metrics.items():
            exposure_value = func(exposure_outputs, exposure_batch)
            exposure_key = f"exposure_{name}"
            self.log(f"{exposure_key}: {exposure_value}")
            metrics[exposure_key] = exposure_value

        # Outcome metrics
        for name, func in self.metrics.items():
            outcome_value = func(outcome_outputs, outcome_batch)
            outcome_key = f"outcome_{name}"
            self.log(f"{outcome_key}: {outcome_value}")
            metrics[outcome_key] = outcome_value

        # Save curves and metrics for both targets
        # Exposure
        save_curves(
            self.run_folder,
            exposure_logits,
            exposure_targets,
            epoch,
            f"{mode}_exposure",
        )
        save_metrics_to_csv(
            self.run_folder,
            {k: v for k, v in metrics.items() if k.startswith("exposure_")},
            epoch,
            f"{mode}_exposure",
        )
        save_curves(
            self.run_folder,
            exposure_logits,
            exposure_targets,
            BEST_MODEL_ID,
            f"{mode}_exposure",
        )
        save_metrics_to_csv(
            self.run_folder,
            {k: v for k, v in metrics.items() if k.startswith("exposure_")},
            BEST_MODEL_ID,
            f"{mode}_exposure",
        )
        save_predictions(
            self.run_folder,
            exposure_logits,
            exposure_targets,
            BEST_MODEL_ID,
            f"{mode}_exposure",
        )

        # Outcome
        save_curves(
            self.run_folder, outcome_logits, outcome_targets, epoch, f"{mode}_outcome"
        )
        save_metrics_to_csv(
            self.run_folder,
            {k: v for k, v in metrics.items() if k.startswith("outcome_")},
            epoch,
            f"{mode}_outcome",
        )
        save_curves(
            self.run_folder,
            outcome_logits,
            outcome_targets,
            BEST_MODEL_ID,
            f"{mode}_outcome",
        )
        save_metrics_to_csv(
            self.run_folder,
            {k: v for k, v in metrics.items() if k.startswith("outcome_")},
            BEST_MODEL_ID,
            f"{mode}_outcome",
        )
        save_predictions(
            self.run_folder,
            outcome_logits,
            outcome_targets,
            BEST_MODEL_ID,
            f"{mode}_outcome",
        )

        save_predictions(
            self.run_folder,
            cf_outcome_logits,
            None,
            BEST_MODEL_ID,
            f"{mode}_outcome_cf",
            save_targets=False,
        )

        return metrics
