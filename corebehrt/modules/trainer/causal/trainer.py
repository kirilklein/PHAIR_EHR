import os
from collections import defaultdict, namedtuple
from typing import Dict

import torch
import yaml
from corebehrt.functional.trainer.plotting import plot_training_curves
from corebehrt.constants.causal.data import (
    CF_OUTCOME,
    EXPOSURE,
    EXPOSURE_TARGET,
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
        self.best_outcome_aucs = {}  # Track best AUC for each outcome
        self.outcome_names = self.model.config.outcome_names

    def _evaluate(self, epoch: int, mode="val") -> tuple:
        """Returns the validation/test loss and metrics for exposure and all outcomes."""
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
        outcome_losses_total = defaultdict(float)

        prediction_data = {
            EXPOSURE: CausalPredictionData(
                metric_values={f"{EXPOSURE}_{name}": [] for name in self.metrics},
                logits_list=[] if self.accumulate_logits else None,
                targets_list=[] if self.accumulate_logits else None,
                target_key=EXPOSURE_TARGET,
            ),
        }

        for outcome_name in self.outcome_names:
            prediction_data[outcome_name] = CausalPredictionData(
                metric_values={f"{outcome_name}_{name}": [] for name in self.metrics},
                logits_list=[] if self.accumulate_logits else None,
                targets_list=[] if self.accumulate_logits else None,
                target_key=outcome_name,
            )
            prediction_data[f"{CF_OUTCOME}_{outcome_name}"] = CausalPredictionData(
                logits_list=[] if self.accumulate_logits else None,
            )

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)
                    cf_outputs = self.model(batch, cf=True)

                if hasattr(outputs, "loss"):
                    loss += outputs.loss.item()
                if hasattr(outputs, "exposure_loss"):
                    exposure_loss_total += outputs.exposure_loss.item()
                if hasattr(outputs, "outcome_losses"):
                    for outcome_name, outcome_loss in outputs.outcome_losses.items():
                        outcome_losses_total[outcome_name] += outcome_loss.item()

                if self.accumulate_logits:
                    self._accumulate_predictions(
                        batch, outputs, cf_outputs, prediction_data
                    )
                else:
                    self._calculate_batch_metrics(batch, outputs, prediction_data)

        if self.accumulate_logits:
            metrics = self.process_causal_classification_results(
                prediction_data, epoch, mode
            )
        else:
            exposure_metrics = compute_avg_metrics(
                prediction_data[EXPOSURE].metric_values
            )

            all_outcome_metrics = {}
            for outcome_name in self.outcome_names:
                outcome_metrics = compute_avg_metrics(
                    prediction_data[outcome_name].metric_values
                )
                all_outcome_metrics.update(outcome_metrics)

            simple_metrics = {}
            for name in self.metrics.keys():
                outcome_values = [
                    all_outcome_metrics[f"{outcome_name}_{name}"]
                    for outcome_name in self.outcome_names
                    if f"{outcome_name}_{name}" in all_outcome_metrics
                ]
                if outcome_values and f"{EXPOSURE}_{name}" in exposure_metrics:
                    all_values = [
                        exposure_metrics[f"{EXPOSURE}_{name}"]
                    ] + outcome_values
                    simple_metrics[name] = sum(all_values) / len(all_values)

            metrics = {**exposure_metrics, **all_outcome_metrics, **simple_metrics}

        self.model.train()

        avg_loss = loss / len(loop)
        avg_exposure_loss = exposure_loss_total / len(loop)
        avg_outcome_losses = {
            outcome_name: total_loss / len(loop)
            for outcome_name, total_loss in outcome_losses_total.items()
        }

        return avg_loss, metrics, avg_exposure_loss, avg_outcome_losses

    def process_causal_classification_results(
        self,
        prediction_data: Dict[str, CausalPredictionData],
        epoch: int,
        mode="val",
    ) -> dict:
        """Process results for exposure and all outcome predictions."""

        metrics = {}

        # Process exposure predictions
        targets = torch.cat(prediction_data[EXPOSURE].targets_list)
        logits = torch.cat(prediction_data[EXPOSURE].logits_list)

        # Calculate metrics for exposure
        batch = {TARGET: targets}
        outputs = namedtuple("Outputs", ["logits"])(logits)

        for name, func in self.metrics.items():
            value = func(outputs, batch)
            key = f"{EXPOSURE}_{name}"
            self.log(f"{key}: {value}")
            metrics[key] = value

        # Save exposure results
        self._save_target_results(
            EXPOSURE,
            logits,
            targets,
            {k: v for k, v in metrics.items() if k.startswith(f"{EXPOSURE}_")},
            epoch,
            mode,
        )

        # Process each outcome prediction
        for outcome_name in self.outcome_names:
            targets = torch.cat(prediction_data[outcome_name].targets_list)
            logits = torch.cat(prediction_data[outcome_name].logits_list)

            # Calculate metrics for this outcome
            batch = {TARGET: targets}
            outputs = namedtuple("Outputs", ["logits"])(logits)

            for name, func in self.metrics.items():
                value = func(outputs, batch)
                key = f"{outcome_name}_{name}"
                self.log(f"{key}: {value}")
                metrics[key] = value

            # Save results for this outcome
            self._save_target_results(
                outcome_name,
                logits,
                targets,
                {k: v for k, v in metrics.items() if k.startswith(f"{outcome_name}_")},
                epoch,
                mode,
            )

            # Save counterfactual predictions for this outcome
            cf_logits = torch.cat(
                prediction_data[f"{CF_OUTCOME}_{outcome_name}"].logits_list
            )
            save_predictions(
                self.run_folder,
                cf_logits,
                None,
                BEST_MODEL_ID,
                f"{mode}_{CF_OUTCOME}_{outcome_name}",
                save_targets=False,
            )

        # Compute average metrics across exposure + all outcomes
        for name in self.metrics.keys():
            outcome_values = [
                metrics[f"{outcome_name}_{name}"]
                for outcome_name in self.outcome_names
                if f"{outcome_name}_{name}" in metrics
            ]
            if outcome_values and f"{EXPOSURE}_{name}" in metrics:
                all_values = [metrics[f"{EXPOSURE}_{name}"]] + outcome_values
                avg_value = sum(all_values) / len(all_values)
                metrics[name] = avg_value
                self.log(f"{name} (avg): {avg_value}")

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

        # Store outcome predictions for each outcome
        for outcome_name in self.outcome_names:
            prediction_data[outcome_name].logits_list.append(
                outputs.outcome_logits[outcome_name].float().cpu()
            )
            prediction_data[outcome_name].targets_list.append(
                batch[f"outcome_{outcome_name}"].cpu()
            )

            # Store counterfactual outcome predictions
            prediction_data[f"{CF_OUTCOME}_{outcome_name}"].logits_list.append(
                cf_outputs.outcome_logits[outcome_name].float().cpu()
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

            # Outcome metrics for each outcome
            for outcome_name in self.outcome_names:
                outcome_outputs = namedtuple("Outputs", ["logits"])(
                    outputs.outcome_logits[outcome_name]
                )
                outcome_batch = {TARGET: batch[outcome_name]}
                outcome_value = func(outcome_outputs, outcome_batch)
                prediction_data[outcome_name].metric_values[
                    f"{outcome_name}_{name}"
                ].append(outcome_value)

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
        plot_training_curves(
            self.metric_history,
            self.epoch_history,
            os.path.join(self.run_folder, "figs"),
            self.log,
        )

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
        if test_exposure_loss is not None:
            self.metric_history["test_exposure_loss"].append(test_exposure_loss)

        # UPDATED: Handle outcome losses as dictionaries
        if val_outcome_loss is not None:
            for outcome_name, loss_value in val_outcome_loss.items():
                self.metric_history[f"val_{outcome_name}_loss"].append(loss_value)

        if test_outcome_loss is not None:
            for outcome_name, loss_value in test_outcome_loss.items():
                self.metric_history[f"test_{outcome_name}_loss"].append(loss_value)

        # Store validation metrics
        if val_metrics:
            for metric_name, value in val_metrics.items():
                self.metric_history[f"val_{metric_name}"].append(float(value))

        # Store test metrics
        if test_metrics:
            for metric_name, value in test_metrics.items():
                self.metric_history[f"test_{metric_name}"].append(float(value))

    def _check_and_freeze_encoder(self, metrics: dict):
        """
        Freeze the encoder if exposure or any outcome AUC plateaus.
        """
        if self.encoder_frozen:
            return

        exposure_auc = metrics.get(f"{EXPOSURE}_roc_auc", 0.5)

        # Check exposure plateau
        exp_plateau, self.best_exposure_auc = self._task_plateau(
            exposure_auc, self.best_exposure_auc, True
        )

        # Check each outcome plateau
        outcome_plateaus = []
        for outcome_name in self.best_outcome_aucs.keys():
            outcome_auc = metrics.get(f"{outcome_name}_roc_auc", 0.5)
            out_plateau, self.best_outcome_aucs[outcome_name] = self._task_plateau(
                outcome_auc, self.best_outcome_aucs.get(outcome_name), True
            )
            outcome_plateaus.append(out_plateau)

        # Initialize outcome AUCs for new outcomes
        for outcome_name in [
            k
            for k in metrics.keys()
            if k.endswith("_roc_auc") and k != f"{EXPOSURE}_roc_auc"
        ]:
            outcome_name_clean = outcome_name.replace("_roc_auc", "")
            if outcome_name_clean not in self.best_outcome_aucs:
                self.best_outcome_aucs[outcome_name_clean] = metrics[outcome_name]

        if exp_plateau or any(outcome_plateaus):
            self._freeze_encoder()
            self.log(
                f"Encoder frozen due to plateau (exp_plateau={exp_plateau}, outcome_plateaus={outcome_plateaus})"
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
            if not any(substring in name for substring in ["pooler", "cls", "head"]):
                param.requires_grad = False
        self.encoder_frozen = True
