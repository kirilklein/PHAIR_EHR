from collections import namedtuple
from typing import Dict

import torch
import yaml

from corebehrt.constants.causal.data import (
    EXPOSURE_TARGET,
    OUTCOME,
    EXPOSURE,
    CF_OUTCOME,
)
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
from dataclasses import dataclass

BEST_MODEL_ID = 999  # For backwards compatibility
DEFAULT_CHECKPOINT_FREQUENCY = 100


@dataclass
class CausalPredictionData:
    logits_list: list
    metric_values: dict = None
    targets_list: list = None
    target_key: str = None


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

        # Metric tracking for both target types
        prediction_data = {
            EXPOSURE: CausalPredictionData(
                metric_values={f"{EXPOSURE}_{name}": [] for name in self.metrics},
                logits_list=[] if self.accumulate_logits else None,
                targets_list=[] if self.accumulate_logits else None,
                target_key=EXPOSURE_TARGET,
            ),
            OUTCOME: CausalPredictionData(
                metric_values={f"{OUTCOME}_{name}": [] for name in self.metrics},
                logits_list=[] if self.accumulate_logits else None,
                targets_list=[] if self.accumulate_logits else None,
                target_key=TARGET,
            ),
            CF_OUTCOME: CausalPredictionData(
                logits_list=[] if self.accumulate_logits else None,
            ),
        }

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
                    # Store predictions for later processing
                    self._accumulate_predictions(
                        batch, outputs, cf_outputs, prediction_data
                    )
                else:
                    # Calculate metrics on the fly
                    self._calculate_batch_metrics(batch, outputs, prediction_data)

        # Process all accumulated predictions
        if self.accumulate_logits:
            metrics = self.process_causal_classification_results(
                prediction_data, epoch, mode
            )
        else:
            # Average metrics calculated on the fly
            exposure_metrics = compute_avg_metrics(
                prediction_data[EXPOSURE].metric_values
            )
            outcome_metrics = compute_avg_metrics(
                prediction_data[OUTCOME].metric_values
            )
            
            # Compute simple average metrics
            simple_metrics = {}
            for name in self.metrics.keys():
                if f"{EXPOSURE}_{name}" in exposure_metrics and f"{OUTCOME}_{name}" in outcome_metrics:
                    simple_metrics[name] = (exposure_metrics[f"{EXPOSURE}_{name}"] + outcome_metrics[f"{OUTCOME}_{name}"]) / 2
            
            metrics = {**exposure_metrics, **outcome_metrics, **simple_metrics}

        self.model.train()

        # Return average loss and all metrics
        return loss / len(loop), metrics

    def process_causal_classification_results(
        self,
        prediction_data: Dict[str, CausalPredictionData],
        epoch: int,
        mode="val",
    ) -> dict:
        """Process results for both exposure and outcome predictions."""

        metrics = {}

        # Process exposure and outcome predictions
        for target_type in [EXPOSURE, OUTCOME]:
            # Concatenate tensors
            targets = torch.cat(prediction_data[target_type].targets_list)
            logits = torch.cat(prediction_data[target_type].logits_list)

            # Calculate metrics
            batch = {TARGET: targets}
            outputs = namedtuple("Outputs", ["logits"])(logits)

            # Compute metrics
            for name, func in self.metrics.items():
                value = func(outputs, batch)
                key = f"{target_type}_{name}"
                self.log(f"{key}: {value}")
                metrics[key] = value

            # Save results
            self._save_target_results(
                target_type, logits, targets, metrics, epoch, mode
            )

        # Compute average metrics (simple metric names)
        for name in self.metrics.keys():
            if f"{EXPOSURE}_{name}" in metrics and f"{OUTCOME}_{name}" in metrics:
                avg_value = (metrics[f"{EXPOSURE}_{name}"] + metrics[f"{OUTCOME}_{name}"]) / 2
                metrics[name] = avg_value
                self.log(f"{name} (avg): {avg_value}")

        # Process counterfactual outcome
        cf_logits = torch.cat(prediction_data[CF_OUTCOME].logits_list)
        save_predictions(
            self.run_folder,
            cf_logits,
            None,
            BEST_MODEL_ID,
            f"{mode}_{CF_OUTCOME}",
            save_targets=False,
        )

        return metrics

    def _accumulate_predictions(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: namedtuple,
        cf_outputs: namedtuple,
        prediction_data: Dict[str, CausalPredictionData],
    ):
        """Helper method to accumulate predictions for later processing"""
        # Store exposure predictions
        prediction_data[EXPOSURE].logits_list.append(
            outputs.exposure_logits.float().cpu()
        )
        prediction_data[EXPOSURE].targets_list.append(batch[EXPOSURE_TARGET].cpu())

        # Store outcome predictions
        prediction_data[OUTCOME].logits_list.append(
            outputs.outcome_logits.float().cpu()
        )
        prediction_data[OUTCOME].targets_list.append(batch[TARGET].cpu())

        # Store counterfactual outcome predictions
        prediction_data[CF_OUTCOME].logits_list.append(
            cf_outputs.outcome_logits.float().cpu()
        )

    def _calculate_batch_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: namedtuple,
        prediction_data: Dict[str, CausalPredictionData],
    ):
        """Helper method to calculate metrics on a per-batch basis"""
        for name, func in self.metrics.items():
            # Exposure metrics
            exposure_outputs = namedtuple("Outputs", ["logits"])(
                outputs.exposure_logits
            )
            exposure_batch = {TARGET: batch[EXPOSURE_TARGET]}
            exposure_value = func(exposure_outputs, exposure_batch)
            prediction_data[EXPOSURE].metric_values[f"{EXPOSURE}_{name}"].append(
                exposure_value
            )

            # Outcome metrics
            outcome_outputs = namedtuple("Outputs", ["logits"])(outputs.outcome_logits)
            outcome_batch = {TARGET: batch[TARGET]}
            outcome_value = func(outcome_outputs, outcome_batch)
            prediction_data[OUTCOME].metric_values[f"{OUTCOME}_{name}"].append(
                outcome_value
            )

    def _save_target_results(
        self,
        target_type: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        metrics: dict,
        epoch: int,
        mode: str,
    ):
        """Helper method to save curves, metrics and predictions for a target type"""
        # Filter metrics for this target type
        target_metrics = {
            k: v for k, v in metrics.items() if k.startswith(f"{target_type}_")
        }

        # Save for current epoch
        save_curves(
            self.run_folder,
            logits,
            targets,
            epoch,
            f"{mode}_{target_type}",
        )
        save_metrics_to_csv(
            self.run_folder,
            target_metrics,
            epoch,
            f"{mode}_{target_type}",
        )

        # Save for best model ID (for backwards compatibility)
        save_curves(
            self.run_folder,
            logits,
            targets,
            BEST_MODEL_ID,
            f"{mode}_{target_type}",
        )
        save_metrics_to_csv(
            self.run_folder,
            target_metrics,
            BEST_MODEL_ID,
            f"{mode}_{target_type}",
        )
        save_predictions(
            self.run_folder,
            logits,
            targets,
            BEST_MODEL_ID,
            f"{mode}_{target_type}",
        )
