import logging
from os.path import join

from corebehrt.constants.causal.paths import (
    CALIBRATE_CFG,
    ENCODE_CFG,
    ESTIMATE_CFG,
    EXTRACT_CRITERIA_CFG,
    GET_STATS_CFG,
    SIMULATE_CFG,
)
from corebehrt.constants.paths import (
    COHORT_CFG,
    DATA_CFG,
    FINETUNE_CFG,
    PREPARE_FINETUNE_CFG,
    PRETRAIN_CFG,
)
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

    def setup_estimate(self) -> None:
        """
        Validates path config and sets up directories for estimate.
        """
        # Setup logging
        self.setup_logging("estimate")

        # Validate and create directories
        if self.cfg.paths.get("calibrated_predictions", False):
            self.check_directory("calibrated_predictions")
        else:
            self.check_directory("exposure_predictions")
            self.check_directory("outcome_predictions")

        # Optional counterfactual outcomes check
        if self.cfg.paths.get("counterfactual_outcomes", False):
            self.check_directory("counterfactual_outcomes")

        # Create estimate directory
        self.create_run_directory("estimate", base="runs")

        # Write config in output directory
        self.write_config("estimate", name=ESTIMATE_CFG)

    def setup_select_cohort_advanced(self) -> None:
        """
        Validates path config and sets up directories for select_cohort_advanced.
        """
        # Setup logging
        self.setup_logging("select_cohort_advanced")
        # Check input directories
        self.check_directory("cohort")
        self.check_directory("meds")
        # Create output directories
        self.create_directory("cohort_advanced")
        self.write_config("cohort_advanced", name=COHORT_CFG)

    def setup_select_cohort_full(self) -> None:
        """
        Validates path config and sets up directories for select_cohort_full.
        """
        # Setup logging
        self.setup_logging("select_cohort_full")
        # Check input directories
        self.check_directory("meds")
        self.check_directory("exposures")
        self.check_directory("features")
        # Create output directories
        self.create_directory("cohort")
        self.write_config("cohort", name=COHORT_CFG)

    def setup_extract_criteria(self) -> None:
        """
        Validates path config and sets up directories for extract_criteria.
        """
        # Setup logging
        self.setup_logging("extract_criteria")
        # Check input directories
        self.check_directory("cohort")
        self.check_directory("meds")
        # Create output directories
        self.create_directory("criteria")
        self.write_config("criteria", name=EXTRACT_CRITERIA_CFG)

    def setup_get_stats(self) -> None:
        """
        Validates path config and sets up directories for get_stats.
        """
        self.setup_logging("get_stats")
        self.create_directory("stats")
        self.check_directory("criteria")
        # Optional input directories
        if self.cfg.paths.get("cohort", None) is not None:
            self.check_directory("cohort")
        if self.cfg.paths.get("ps_calibrated_predictions", None) is not None:
            self.check_directory("ps_calibrated_predictions")
        if self.cfg.paths.get("outcome_model", None) is not None:
            self.check_directory("outcome_model")

        self.write_config("stats", name=GET_STATS_CFG)

    def setup_prepare_finetune_exposure_outcome(self, name=None) -> None:
        """
        Validates path config and sets up directories for preparing finetune data.
        """
        self.setup_logging("prepare finetune data")
        self.check_directory("features")
        self.check_directory("tokenized")
        self.check_directory("cohort")

        # If "outcome" is set, check that it exists.
        if outcome := self.cfg.paths.get("outcome", False):
            # If "outcomes" is also set, use as prefix
            if outcomes := self.cfg.paths.get("outcomes", False):
                self.cfg.paths.outcome = join(outcomes, outcome)

            self.check_file("outcome")

        if exposure := self.cfg.paths.get("exposure", False):
            if exposures := self.cfg.paths.get("exposures", False):
                self.cfg.paths.exposure = join(exposures, exposure)

            self.check_file("exposure")

        self.create_directory("prepared_data", clear=True)
        if name is None:
            self.write_config("prepared_data", name=PREPARE_FINETUNE_CFG)
        else:
            # If name is given, use it as config name
            self.write_config("prepared_data", name=name)
        self.write_config("prepared_data", source="features", name=DATA_CFG)
