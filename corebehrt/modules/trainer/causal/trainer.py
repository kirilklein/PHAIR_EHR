import os
from collections import defaultdict, namedtuple
from typing import Dict

import matplotlib.pyplot as plt
import torch
import yaml

from corebehrt.constants.causal.data import (
    CF_OUTCOME,
    EXPOSURE,
    EXPOSURE_TARGET,
    OUTCOME,
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
    def __init__(self, *args, plateau_threshold=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metric tracking for plotting
        self.metric_history = defaultdict(list)
        self.epoch_history = []
        self.plateau_threshold = plateau_threshold
        self.encoder_frozen = False
        self.best_exposure_auc = None
        self.best_outcome_auc = None

    def _evaluate(self, epoch: int, mode="val") -> tuple:
        """Returns the validation/test loss and metrics for both exposure and outcome."""
        if mode == "val":
            if self.val_dataset is None:
                self.log("No validation dataset provided")
                return None, None, None, None
            dataloader = self.get_dataloader(self.val_dataset, mode="val")
        elif mode == "test":
            if self.test_dataset is None:
                self.log("No test dataset provided")
                return None, None, None, None
            dataloader = self.get_dataloader(self.test_dataset, mode="test")
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'val' or 'test'")

        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(mode)
        loss = 0
        exposure_loss_total = 0
        outcome_loss_total = 0

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
                if hasattr(outputs, "exposure_loss"):
                    exposure_loss_total += outputs.exposure_loss.item()
                if hasattr(outputs, "outcome_loss"):
                    outcome_loss_total += outputs.outcome_loss.item()

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
                if (
                    f"{EXPOSURE}_{name}" in exposure_metrics
                    and f"{OUTCOME}_{name}" in outcome_metrics
                ):
                    simple_metrics[name] = (
                        exposure_metrics[f"{EXPOSURE}_{name}"]
                        + outcome_metrics[f"{OUTCOME}_{name}"]
                    ) / 2

            metrics = {**exposure_metrics, **outcome_metrics, **simple_metrics}
        if mode == "val" and metrics:
            self._update_model_task_weights(metrics)
        self.model.train()

        # Return average losses and all metrics
        avg_loss = loss / len(loop)
        avg_exposure_loss = exposure_loss_total / len(loop)
        avg_outcome_loss = outcome_loss_total / len(loop)

        return avg_loss, metrics, avg_exposure_loss, avg_outcome_loss

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
                avg_value = (
                    metrics[f"{EXPOSURE}_{name}"] + metrics[f"{OUTCOME}_{name}"]
                ) / 2
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

    def validate_and_log(self, epoch: int, epoch_loss: float, train_loop) -> None:
        val_loss, val_metrics, val_exposure_loss, val_outcome_loss = self._evaluate(
            epoch, mode="val"
        )
        _, test_metrics, test_exposure_loss, test_outcome_loss = self._evaluate(
            epoch, mode="test"
        )

        # Calculate average train loss for this epoch
        avg_train_loss = sum(epoch_loss) / (len(train_loop) / self.accumulation_steps)

        # Store metrics for plotting
        self._update_metric_history(
            epoch,
            avg_train_loss,
            val_loss,
            val_metrics,
            test_metrics,
            val_exposure_loss,
            val_outcome_loss,
            test_exposure_loss,
            test_outcome_loss,
        )

        # Plot metrics
        self._plot_metrics()

        if epoch == 1:  # for testing purposes/if first epoch is best
            self._save_checkpoint(
                epoch,
                train_loss=epoch_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                final_step_loss=epoch_loss[-1],
                best_model=True,
            )

        self._self_log_results(
            epoch, val_loss, val_metrics, epoch_loss, len(train_loop)
        )

        current_metric_value = val_metrics.get(
            self.stopping_metric, val_loss
        )  # get the metric we monitor. Same as early stopping

        if self._should_unfreeze_on_plateau(current_metric_value):
            self._unfreeze_model("Performance plateau detected!")

        if self._should_stop_early(
            epoch, current_metric_value, val_loss, epoch_loss, val_metrics, test_metrics
        ):
            return
        self._save_checkpoint_conditionally(
            epoch, epoch_loss, val_loss, val_metrics, test_metrics
        )
        self._check_and_freeze_encoder(val_metrics)

    def _update_metric_history(
        self,
        epoch,
        train_loss,
        val_loss,
        val_metrics,
        test_metrics,
        val_exposure_loss=None,
        val_outcome_loss=None,
        test_exposure_loss=None,
        test_outcome_loss=None,
    ):
        """Update the metric history for plotting"""
        self.epoch_history.append(epoch)

        # Store losses - use val_loss as default for first epoch if train_loss is 0
        if train_loss == 0 and val_loss is not None:
            self.metric_history["train_loss"].append(val_loss)
        else:
            self.metric_history["train_loss"].append(train_loss)

        if val_loss is not None:
            self.metric_history["val_loss"].append(val_loss)

        # Store individual losses
        if val_exposure_loss is not None:
            self.metric_history["val_exposure_loss"].append(val_exposure_loss)
        if val_outcome_loss is not None:
            self.metric_history["val_outcome_loss"].append(val_outcome_loss)
        if test_exposure_loss is not None:
            self.metric_history["test_exposure_loss"].append(test_exposure_loss)
        if test_outcome_loss is not None:
            self.metric_history["test_outcome_loss"].append(test_outcome_loss)

        # Store validation metrics
        if val_metrics:
            for metric_name, value in val_metrics.items():
                self.metric_history[f"val_{metric_name}"].append(float(value))

        # Store test metrics
        if test_metrics:
            for metric_name, value in test_metrics.items():
                self.metric_history[f"test_{metric_name}"].append(float(value))

    def _plot_metrics(self):
        """Plot all metrics and save to output_dir/figs"""
        if len(self.epoch_history) < 2:  # Need at least 2 points to plot
            return

        figs_dir = os.path.join(self.run_folder, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        # Group metrics by base name for better visualization
        metric_groups = self._group_metrics()

        for group_name, metrics in metric_groups.items():
            self._plot_metric_group(group_name, metrics, figs_dir)

    def _group_metrics(self):
        """Group metrics by their base name (e.g., roc_auc, pr_auc, loss)"""
        groups = defaultdict(dict)

        for metric_name, values in self.metric_history.items():
            if len(values) != len(self.epoch_history):
                continue  # Skip if lengths don't match

            # Determine base metric name and prefix
            if metric_name in ["train_loss", "val_loss"]:
                base_name = "loss"
                prefix = metric_name.replace("_loss", "")
            elif metric_name.startswith("val_"):
                base_name = metric_name[4:]  # Remove 'val_' prefix
                prefix = "val"
            elif metric_name.startswith("test_"):
                base_name = metric_name[5:]  # Remove 'test_' prefix
                prefix = "test"
            else:
                base_name = metric_name
                prefix = "train"

            groups[base_name][prefix] = values

        return groups

    def _plot_metric_group(self, metric_name, metric_data, figs_dir):
        """Plot a group of related metrics (e.g., train/val/test for same metric)"""
        plt.figure(figsize=(10, 6))

        # Define colors for different metric types
        colors = {
            "train": "blue",
            "val": "orange",
            "test": "green",
            "exposure": "red",
            "outcome": "purple",
        }

        for prefix, values in metric_data.items():
            color = colors.get(prefix, "black")
            plt.plot(
                self.epoch_history,
                values,
                label=f"{prefix}_{metric_name}",
                color=color,
                marker="o",
                markersize=3,
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"{metric_name.replace('_', ' ').title()} Over Training")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.savefig(
            os.path.join(figs_dir, f"{metric_name}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()  # Close to prevent memory issues

    def _update_model_task_weights(self, metrics: dict):
        exposure_auc = metrics.get(f"{EXPOSURE}_roc_auc", 0.5)
        outcome_auc = metrics.get(f"{OUTCOME}_roc_auc", 0.5)

        exposure_weight = max(1 - ((exposure_auc - 0.5) * 2), 0.1)
        outcome_weight = max(1 - ((outcome_auc - 0.5) * 2), 0.1)

        self.model.update_task_weights(exposure_weight, outcome_weight)

        self.log(
            f"Updated model task weights - Exposure: {exposure_weight}, Outcome: {outcome_weight}"
        )

    def _check_and_freeze_encoder(self, metrics: dict):
        """
        Freeze the encoder if either exposure or outcome AUC plateaus.
        """
        if self.encoder_frozen:
            return

        exposure_auc = metrics.get(f"{EXPOSURE}_roc_auc", 0.5)
        outcome_auc = metrics.get(f"{OUTCOME}_roc_auc", 0.5)

        exp_plateau, self.best_exposure_auc = self._task_plateau(
            exposure_auc, self.best_exposure_auc, True
        )
        out_plateau, self.best_outcome_auc = self._task_plateau(
            outcome_auc, self.best_outcome_auc, True
        )

        if exp_plateau or out_plateau:
            self._freeze_encoder()
            self.log(
                f"Encoder frozen due to plateau (exp_plateau={exp_plateau}, out_plateau={out_plateau})"
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.1
                self.log(f"Learning rate updated to {param_group['lr']}")

    def _task_plateau(
        self, current_auc: float, best_auc: float, higher_is_better: bool
    ):
        """
        Check if a task's AUC has plateaued and update best AUC.

        Returns:
            plateau (bool): True if improvement < threshold
            new_best_auc (float): updated best AUC value
        """
        if best_auc is None:
            # Initialize best AUC
            return False, current_auc

        improvement = (current_auc - best_auc) / best_auc
        plateau = (
            improvement < self.plateau_threshold
            if higher_is_better
            else improvement > -self.plateau_threshold
        )
        improvement = improvement if higher_is_better else -improvement
        new_best = current_auc if improvement > 0 else best_auc
        return plateau, new_best

    def _freeze_encoder(self):
        """
        Freeze all encoder parameters to stop updating them.
        """
        for name, param in self.model.named_parameters():
            if "cls" not in name:
                param.requires_grad = False
        self.encoder_frozen = True
