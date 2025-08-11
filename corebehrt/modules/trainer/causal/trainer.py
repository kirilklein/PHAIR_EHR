import os
from collections import defaultdict, namedtuple
from typing import Dict

import torch
import random
import yaml

from corebehrt import azure
from corebehrt.constants.causal.data import (
    CF_OUTCOME,
    EXPOSURE,
    EXPOSURE_TARGET,
)
from corebehrt.constants.data import TARGET
from corebehrt.functional.trainer.freezing_utils import check_task_plateau
from corebehrt.functional.trainer.plotting import (
    plot_prediction_histograms,
    plot_training_curves,
)
from corebehrt.modules.monitoring.logger import get_tqdm
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_avg_metrics,
    save_curves,
    save_metrics_to_csv,
    save_predictions,
)
from corebehrt.modules.setup.config import Config
from corebehrt.modules.trainer.causal.utils import CausalPredictionData, EpochMetrics
from corebehrt.modules.trainer.pcgrad import PCGrad
from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.trainer.utils import limit_dict_for_logging

yaml.add_representer(Config, lambda dumper, data: data.yaml_repr(dumper))

BEST_MODEL_ID = 999  # For backwards compatibility
DEFAULT_CHECKPOINT_FREQUENCY = 100


class CausalEHRTrainer(EHRTrainer):
    def __init__(self, *args, plateau_threshold=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metric tracking for plotting
        self.metric_history = defaultdict(list)
        self.epoch_history = []
        self.plateau_threshold = plateau_threshold
        self.encoder_frozen = False
        self.outcome_names = self.model.config.outcome_names
        self.best_outcome_aucs = {name: None for name in self.outcome_names}
        self.best_exposure_auc = None
        self.use_pcgrad = self.args.get("use_pcgrad", False)
        self.plot_histograms = self.args.get("plot_histograms", False)
        self.save_encodings = self.args.get("save_encodings", False)
        self._set_plateau_parameters()
        self._set_logging_parameters()

        if self.use_pcgrad:
            self.optimizer = PCGrad(self.optimizer)

    def _set_plateau_parameters(self):
        self.freeze_encoder_on_plateau = self.args.get(
            "freeze_encoder_on_plateau", False
        )
        self.freeze_encoder_on_plateau_threshold = self.args.get(
            "freeze_encoder_on_plateau_threshold", 0.01
        )
        self.freeze_encoder_on_plateau_patience = self.args.get(
            "freeze_encoder_on_plateau_patience", 3
        )
        self.plateau_counter = 0

    def _set_logging_parameters(self):
        self.log_all_targets = self.args.get("log_all_targets", False)
        self.save_curves = self.args.get("save_curves", False)
        self.num_targets_to_log = self.args.get("num_targets_to_log", 10)
        self.outcome_names_to_log = self.outcome_names
        if (
            not self.log_all_targets
            and len(self.outcome_names) > self.num_targets_to_log
        ):
            self.outcome_names_to_log = random.sample(
                self.outcome_names, self.num_targets_to_log
            )
            self.log(
                f"Logging metrics for a subset of {self.num_targets_to_log} \n outcomes: {self.outcome_names_to_log}"
            )

    def _train_step(self, batch: dict):
        self.optimizer.zero_grad()
        self.batch_to_device(batch)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = self.model(batch)

        if self.use_pcgrad:
            # Collect individual task losses for PCGrad
            task_losses = []

            # Add exposure loss if available
            if hasattr(outputs, "exposure_loss") and outputs.exposure_loss is not None:
                task_losses.append(outputs.exposure_loss)

            # Add outcome losses if available
            if hasattr(outputs, "outcome_losses") and outputs.outcome_losses:
                for outcome_loss in outputs.outcome_losses.values():
                    if outcome_loss is not None:
                        task_losses.append(outcome_loss)

            # If we don't have individual losses, fall back to total loss
            if not task_losses and hasattr(outputs, "loss"):
                task_losses = [outputs.loss]

            if not task_losses:
                return outputs.loss if hasattr(outputs, "loss") else None

            # Scale each loss individually for PCGrad
            scaled_losses = [self.scaler.scale(loss) for loss in task_losses]

            # Use PCGrad backward instead of regular backward
            if len(scaled_losses) > 1:
                # Multiple tasks - use PCGrad
                self.optimizer.pc_backward(scaled_losses)
            else:
                # Single task - use regular backward
                scaled_losses[0].backward()

            # Return the total loss for logging (unscaled)
            return outputs.loss if hasattr(outputs, "loss") else sum(task_losses)
        else:
            loss = outputs.loss
            if loss is not None:
                self.scaler.scale(loss).backward()
            return loss

    def _log_batch(self, metrics: list):
        metrics_for_log = [
            m for i, m in enumerate(metrics) if i < self.num_targets_to_log
        ]
        if azure.is_mlflow_available():
            azure.log_batch(metrics=metrics_for_log)
        else:
            self.log(metrics_for_log)

    def _update(self):
        """Updates the model (optimizer and scheduler) with PCGrad support"""
        # PCGrad optimizer exposes param_groups, so scaler can work with it directly
        self._clip_gradients()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

    def _self_log_results(
        self,
        epoch: int,
        val_loss: float,
        val_metrics: dict,
        epoch_loss: float,
        len_train_loop: int,
    ) -> None:
        self.log(f"Epoch {epoch} val loss: {val_loss}")
        self.log(
            f"Epoch {epoch} metrics: {limit_dict_for_logging(val_metrics, self.num_targets_to_log)}\n"
        )

    def _evaluate(self, mode="val", save_encodings: bool = False) -> tuple:
        """Returns the validation/test loss and metrics for exposure and all outcomes."""
        if mode == "val":
            if self.val_dataset is None:
                self.log("No validation dataset provided")
                return None, None, None, None, None
            dataloader = self.get_dataloader(self.val_dataset, mode="val")
        elif mode == "test":
            if self.test_dataset is None:
                self.log("No test dataset provided")
                return None, None, None, None, None
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

        if save_encodings:
            prediction_data["pids"] = []
            prediction_data["patient_encodings"] = []
            prediction_data["token_ids"] = []
            prediction_data["token_encodings"] = []

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
                    outputs = self.model(batch, return_encodings=save_encodings)
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
                    if save_encodings:
                        self._accumulate_encodings(outputs, prediction_data)
                else:
                    self._calculate_batch_metrics(batch, outputs, prediction_data)

        if self.accumulate_logits:
            metrics = self.process_causal_classification_results(
                prediction_data, mode, save_results=False
            )
            if save_encodings:
                self._save_encodings(prediction_data, mode)
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

        return avg_loss, metrics, avg_exposure_loss, avg_outcome_losses, prediction_data

    def _clip_gradients(self):
        """Clip gradients with PCGrad support"""
        if self.args.get("gradient_clip", False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.args.get("gradient_clip", {}).get("clip_value", 1.0),
            )

    def process_causal_classification_results(
        self,
        prediction_data: Dict[str, CausalPredictionData],
        mode="val",
        save_results=False,
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
            self.log(f"{key}: {round(value, 3)}")
            metrics[key] = value

        if save_results:
            self._save_target_results(
                EXPOSURE,
                logits,
                targets,
                {k: v for k, v in metrics.items() if k.startswith(f"{EXPOSURE}_")},
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
                metrics[key] = value

            if save_results:
                self._save_target_results(
                    outcome_name,
                    logits,
                    targets,
                    {
                        k: v
                        for k, v in metrics.items()
                        if k.startswith(f"{outcome_name}_")
                    },
                    mode,
                )
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
                self.log(f"{name} (avg): {round(avg_value, 3)}")

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
            prediction_data[outcome_name].targets_list.append(batch[outcome_name].cpu())

            # Store counterfactual outcome predictions
            prediction_data[f"{CF_OUTCOME}_{outcome_name}"].logits_list.append(
                cf_outputs.outcome_logits[outcome_name].float().cpu()
            )

    def _accumulate_encodings(self, outputs, prediction_data):
        prediction_data["pids"].extend(outputs.pids.tolist())
        prediction_data["patient_encodings"].append(outputs.patient_encodings.cpu())
        prediction_data["token_ids"].append(outputs.token_ids.cpu())
        prediction_data["token_encodings"].append(outputs.token_encodings.cpu())

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

    def _save_encodings(self, prediction_data: dict, mode: str):
        """Saves token and patient encodings."""
        self.log(f"Saving encodings for mode {mode}...")
        encodings_dir = os.path.join(self.run_folder, "encodings")
        os.makedirs(encodings_dir, exist_ok=True)
        # This can consume a lot of memory, but it's the most straightforward way
        # to save the encodings per patient.
        pids = prediction_data["pids"]
        patient_encodings = torch.cat(prediction_data["patient_encodings"], dim=0)

        # This will be very large.
        token_ids = torch.cat(prediction_data["token_ids"], dim=0)
        token_encodings = torch.cat(prediction_data["token_encodings"], dim=0)

        patient_enc_dict = {pid: enc for pid, enc in zip(pids, patient_encodings)}
        token_enc_dict = {
            pid: {"token_ids": t_ids, "encodings": t_enc}
            for pid, t_ids, t_enc in zip(pids, token_ids, token_encodings)
        }

        torch.save(
            patient_enc_dict,
            os.path.join(encodings_dir, f"{mode}_patient_encodings.pt"),
        )
        torch.save(
            token_enc_dict, os.path.join(encodings_dir, f"{mode}_token_encodings.pt")
        )

        self.log("Encodings saved.")

    def _save_target_results(
        self,
        target_type: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        metrics: dict,
        mode: str,
    ):
        """Helper method to save curves, metrics and predictions for a target type"""
        # Filter metrics for this target type
        target_metrics = {
            k: v for k, v in metrics.items() if k.startswith(f"{target_type}_")
        }

        # Save for best model ID (for backwards compatibility)
        if self.save_curves:
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
        (
            val_loss,
            val_metrics,
            val_exposure_loss,
            val_outcome_loss,
            val_prediction_data,
        ) = self._evaluate(mode="val")
        _, test_metrics, test_exposure_loss, test_outcome_loss, test_prediction_data = (
            self._evaluate(mode="test")
        )

        current_metric_value = val_metrics.get(
            self.stopping_metric, val_loss
        )  # get the metric we monitor. Same as early stopping
        is_best = self._is_improvement(current_metric_value)
        if is_best and epoch > 0:
            self.log(
                f"New best model found at epoch {epoch} with {self.stopping_metric}: \
                    {round(current_metric_value, 3)}. Saving results."
            )
            # If it's the best, save all the detailed artifacts
            if self.accumulate_logits:
                self.process_causal_classification_results(
                    val_prediction_data, mode="val", save_results=False
                )
                if test_prediction_data:
                    self.process_causal_classification_results(
                        test_prediction_data, mode="test", save_results=False
                    )
        # # Add debugging to see what metrics are being returned
        # self.log(
        #     f"Validation metrics keys: {list(val_metrics.keys()) if val_metrics else 'None'}"
        # )
        # self.log(
        #     f"Test metrics keys: {list(test_metrics.keys()) if test_metrics else 'None'}"
        # )

        # Plot prediction histograms
        if self.plot_histograms and val_prediction_data:
            plot_prediction_histograms(
                val_prediction_data,
                self.run_folder,
                self.outcome_names_to_log,
                self.accumulate_logits,
            )

        # Check outcome-specific metrics
        if val_metrics:
            all_outcome_metrics = {
                k: v
                for k, v in val_metrics.items()
                if any(outcome in k for outcome in self.outcome_names)
            }
            if self.log_all_targets:
                self.log(
                    f"Outcome metrics in validation: \
                    \n{limit_dict_for_logging(all_outcome_metrics, self.num_targets_to_log)}"
                )
            else:
                outcome_metrics_to_log = {
                    k: v
                    for k, v in all_outcome_metrics.items()
                    if any(
                        outcome_log in k for outcome_log in self.outcome_names_to_log
                    )
                }
                if outcome_metrics_to_log:
                    self.log(
                        f"Outcome metrics in validation (subset): \
                            \n{limit_dict_for_logging(outcome_metrics_to_log, self.num_targets_to_log)}"
                    )

        # Calculate average train loss for this epoch
        avg_train_loss = sum(epoch_loss) / (len(train_loop) / self.accumulation_steps)  # type: ignore

        # Store metrics for plotting, but not for epoch 0
        if epoch > 0:
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                val_exposure_loss=val_exposure_loss,
                val_outcome_losses=val_outcome_loss,
                test_exposure_loss=test_exposure_loss,
                test_outcome_losses=test_outcome_loss,
            )
            self._update_metric_history(epoch_metrics)

            # Plot metrics
            plot_training_curves(
                self.metric_history,
                self.epoch_history,
                os.path.join(self.run_folder, "figs"),
                self.outcome_names,
                self.log,
                max_legend_items=self.num_targets_to_log,
            )

        if (is_best and epoch > 0) or (
            epoch == 1
        ):  # for testing purposes/if first epoch is best
            self._save_checkpoint(
                BEST_MODEL_ID,
                best_model=True,
            )

        self._self_log_results(
            epoch, val_loss, val_metrics, epoch_loss, len(train_loop)
        )

        if self._should_stop_early(
            epoch, current_metric_value, val_loss, epoch_loss, val_metrics, test_metrics
        ):
            return

        self._check_and_freeze_encoder(val_metrics)

    def _update_metric_history(self, metrics: EpochMetrics):
        """Update the metric history for plotting"""
        self.epoch_history.append(metrics.epoch)

        # Store main losses
        self._store_scalar_loss(
            "train_loss", metrics.train_loss, fallback=metrics.val_loss
        )
        self._store_scalar_loss("val_loss", metrics.val_loss)

        # Store exposure losses
        self._store_scalar_loss("val_exposure_loss", metrics.val_exposure_loss)
        self._store_scalar_loss("test_exposure_loss", metrics.test_exposure_loss)

        # Store outcome losses and metrics
        self._store_outcome_losses("val", metrics.val_outcome_losses)
        self._store_outcome_losses("test", metrics.test_outcome_losses)
        self._store_metrics("val", metrics.val_metrics)
        self._store_metrics("test", metrics.test_metrics)

    def _store_scalar_loss(self, key: str, value, fallback=None):
        """Store a scalar loss value with optional fallback"""
        if value is not None:
            self.metric_history[key].append(value)
        elif value == 0 and fallback is not None:  # Handle train_loss special case
            self.metric_history[key].append(fallback)

    def _store_outcome_losses(self, prefix: str, outcome_losses: dict):
        """Store outcome losses from a dictionary"""
        if outcome_losses is not None:
            for outcome_name, loss_value in outcome_losses.items():
                key = f"{prefix}_{outcome_name}_loss"
                self.metric_history[key].append(loss_value)

    def _store_metrics(self, prefix: str, metrics: dict):
        """Store metrics with a given prefix"""
        if metrics:
            self.log(
                f"Storing metrics: {limit_dict_for_logging(metrics, self.num_targets_to_log)}"
            )
            for metric_name, value in metrics.items():
                is_outcome_metric = any(
                    outcome_name in metric_name for outcome_name in self.outcome_names
                )
                # For outcome metrics, only store if they are in the log subset or if all are logged
                should_log = not is_outcome_metric or (
                    is_outcome_metric
                    and any(
                        outcome_log in metric_name
                        for outcome_log in self.outcome_names_to_log
                    )
                )

                if should_log:
                    key = f"{prefix}_{metric_name}"
                    self.metric_history[key].append(float(value))

    def _check_and_freeze_encoder(self, metrics: dict):
        """
        Freeze the encoder if exposure or any outcome AUC plateaus for a number of epochs.
        This now uses the standalone `check_task_plateau` utility.
        """
        if not self.freeze_encoder_on_plateau or self.encoder_frozen:
            return

        exposure_auc = metrics.get(f"{EXPOSURE}_roc_auc", 0.5)

        exp_plateau, self.best_exposure_auc = check_task_plateau(
            current_metric=exposure_auc,
            best_metric=self.best_exposure_auc,
            threshold=self.freeze_encoder_on_plateau_threshold,
            higher_is_better=True,
        )

        # Check each outcome for a plateau
        outcome_plateaus = []
        for outcome_name in self.best_outcome_aucs.keys():
            outcome_auc = metrics.get(f"{outcome_name}_roc_auc", 0.5)
            out_plateau, self.best_outcome_aucs[outcome_name] = check_task_plateau(
                current_metric=outcome_auc,
                best_metric=self.best_outcome_aucs[outcome_name],
                threshold=self.freeze_encoder_on_plateau_threshold,
                higher_is_better=True,
            )
            outcome_plateaus.append(out_plateau)

        if exp_plateau or any(outcome_plateaus):
            self.plateau_counter += 1
            self.log(
                f"Plateau detected. Counter: \
                    {self.plateau_counter}/{self.freeze_encoder_on_plateau_patience}"
            )
        else:
            if self.plateau_counter > 0:
                self.log(f"No plateau. Counter reset to 0.")
            self.plateau_counter = 0

        if self.plateau_counter >= self.freeze_encoder_on_plateau_patience:
            self._freeze_encoder()
            self.log(
                f"Encoder frozen due to plateau for {self.freeze_encoder_on_plateau_patience} \
                    epochs (exp_plateau={exp_plateau}, \
                        outcome_plateaus={outcome_plateaus})"
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.5
                self.log(f"Learning rate updated to {param_group['lr']}")

    def _freeze_encoder(self):
        """
        Freeze all encoder parameters to stop updating them.
        """
        for name, param in self.model.named_parameters():
            if not any(substring in name for substring in ["pooler", "cls", "head"]):
                param.requires_grad = False
        self.encoder_frozen = True

    def _save_checkpoint(self, epoch: int, best_model=False, **kwargs) -> None:
        """Saves a checkpoint. Model with optimizer and scheduler if available."""
        # Model/training specific
        id = epoch if not best_model else BEST_MODEL_ID
        os.makedirs(os.path.join(self.run_folder, "checkpoints"), exist_ok=True)
        checkpoint_name = os.path.join(
            self.run_folder, "checkpoints", f"checkpoint_epoch{id}_end.pt"
        )

        if self.use_pcgrad:
            # PCGrad exposes the underlying optimizer via the optimizer property
            optimizer_state_dict = self.optimizer.optimizer.state_dict()
        else:
            optimizer_state_dict = self.optimizer.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer_state_dict,
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                **kwargs,
            },
            checkpoint_name,
        )
