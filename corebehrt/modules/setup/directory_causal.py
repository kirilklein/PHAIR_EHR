import logging

from corebehrt.constants.causal.paths import (
    CALIBRATE_CFG,
    ENCODE_CFG,
    ESTIMATE_CFG,
    SIMULATE_CFG,
    TRAIN_MLP_CFG,
    TRAIN_XGB_CFG,
)
from corebehrt.constants.paths import COHORT_CFG, FINETUNE_CFG, PRETRAIN_CFG
from corebehrt.modules.setup.config import Config
from corebehrt.modules.setup.directory import DirectoryPreparer

logger = logging.getLogger(__name__)  # Get the logger for this module


class CausalDirectoryPreparer(DirectoryPreparer):
    """Prepares directories for training and evaluation."""

    def __init__(self, cfg: Config) -> None:
        """Sets up DirectoryPreparer and adds defaul configuration to cfg."""
        super().__init__(cfg)

    def setup_encode(self) -> None:
        """
        Validates path config and sets up directories for encode.
        """
        # Setup logging
        self.setup_logging("encode")

        # Validate and create directories
        self.check_directory("finetune_model")
        self.create_directory("encoded_data")

        # Write config in output directory.
        self.write_config("encoded_data", source="finetune_model", name=PRETRAIN_CFG)
        self.write_config("encoded_data", source="finetune_model", name=FINETUNE_CFG)
        self.write_config("encoded_data", name=ENCODE_CFG)

    def setup_simulate(self) -> None:
        """
        Validates path config and sets up directories for simulate.
        """
        # Setup logging
        self.setup_logging("simulate")

        # Validate and create directories
        self.check_directory("calibrated_predictions")
        self.check_directory("encoded_data")
        self.create_directory("simulated_outcome")

        # Write config in output directory.
        self.write_config(
            "simulated_outcome", source="calibrated_predictions", name=CALIBRATE_CFG
        )
        self.write_config(
            "simulated_outcome", source="calibrated_predictions", name=FINETUNE_CFG
        )
        self.write_config("simulated_outcome", source="encoded_data", name=ENCODE_CFG)
        self.write_config("simulated_outcome", name=SIMULATE_CFG)

    def setup_calibrate(self) -> None:
        """
        Validates path config and sets up directories for calibrate.
        """
        # Setup logging
        self.setup_logging("calibrate")

        # Validate and create directories
        self.check_directory("finetune_model")
        self.create_directory("calibrated_predictions")

        # Write config in output directory.
        self.write_config(
            "calibrated_predictions", source="finetune_model", name=FINETUNE_CFG
        )
        self.write_config("calibrated_predictions", name=CALIBRATE_CFG)

    def setup_train_mlp(self) -> None:
        """
        Validates path config and sets up directories for train_mlp.
        """
        # Setup logging
        self.setup_logging("train_mlp")

        # Validate and create directories
        self.check_directory("encoded_data")
        self.check_directory("calibrated_predictions")
        self.check_directory("cohort")
        self.create_run_directory("trained_mlp", base="runs")

        # Write config in output directory.
        self.write_config("trained_mlp", source="encoded_data", name=ENCODE_CFG)
        self.write_config(
            "trained_mlp", source="calibrated_predictions", name=CALIBRATE_CFG
        )
        self.write_config("trained_mlp", source="cohort", name=COHORT_CFG)
        self.write_config("trained_mlp", name=TRAIN_MLP_CFG)

    def setup_train_xgb(self) -> None:
        """
        Validates path config and sets up directories for train_xgb.
        """
        out = "trained_xgb"
        # Setup logging
        self.setup_logging("train_xgb")

        # Validate and create directories
        self.check_directory("encoded_data")
        self.check_directory("calibrated_predictions")
        self.check_directory("cohort")
        self.create_run_directory(out, base="runs")

        # Write config in output directory.
        self.write_config(out, source="encoded_data", name=ENCODE_CFG)
        self.write_config(out, source="calibrated_predictions", name=CALIBRATE_CFG)
        self.write_config(out, source="cohort", name=COHORT_CFG)
        self.write_config(out, name=TRAIN_XGB_CFG)

    def setup_estimate(self) -> None:
        """
        Validates path config and sets up directories for estimate.
        """
        # Setup logging
        self.setup_logging("estimate")

        # Validate and create directories
        self.check_directory("exposure_predictions")
        self.check_directory("outcome_predictions")

        # Optional counterfactual outcomes check
        if self.cfg.paths.get("counterfactual_outcomes", False):
            self.check_directory("counterfactual_outcomes")

        # Create estimate directory
        self.create_run_directory("estimate", base="runs")

        # Write config in output directory
        self.write_config("estimate", name=ESTIMATE_CFG)
