import logging
import os
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
        try:
            self.check_directory("calibrated_predictions")
        except:
            logger.warning(
                "No calibrated predictions found, checking for exposure and outcome predictions"
            )
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
        self.setup_prepare_finetune_exposure_outcome_shared(name)

        # If "outcome" is set, check that it exists.
        if outcome := self.cfg.paths.get("outcome", False):
            # If "outcomes" is also set, use as prefix
            if outcomes := self.cfg.paths.get("outcomes", False):
                self.cfg.paths.outcome = join(outcomes, outcome)

            self.check_file("outcome")

    def setup_prepare_finetune_exposure_multitarget(self, name=None) -> None:
        """
        Validates path config and sets up directories for preparing finetune data with multiple outcomes.
        """
        self.setup_prepare_finetune_exposure_outcome_shared(name)

        # Outcomes directory must be set for multi-outcome handling
        if not (outcomes_dir := self.cfg.paths.get("outcomes", False)):
            raise ValueError(
                "'outcomes' directory must be set when using multi-outcome setup"
            )

        # Get outcome file paths as dictionary based on configuration
        outcome_file_dict = self._get_outcome_file_dict(outcomes_dir)

        # Set the final outcome paths dictionary
        self.cfg.paths.outcome_files = outcome_file_dict

    def setup_prepare_finetune_exposure_outcome_shared(self, name=None):
        self.setup_logging("prepare finetune data")
        self.check_directory("features")
        self.check_directory("tokenized")
        self.check_directory("cohort")

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

    def _get_outcome_file_dict(self, outcomes_dir: str) -> dict[str, str]:
        """
        Get dictionary of outcome file paths based on configuration.

        Args:
            outcomes_dir: Directory containing outcome files

        Returns:
            Dictionary mapping outcome names to full file paths
        """
        outcome_names = self.cfg.paths.get("outcome_names", None)

        if outcome_names is None:
            logger.warning(
                "No outcome names found, discovering all CSV files in outcomes directory"
            )
            return self._discover_csv_files_dict(outcomes_dir)
        elif isinstance(outcome_names, dict):
            logger.info(f"Creating outcome dictionary for {outcome_names}")
            outcome_paths = self._create_outcome_dict(outcomes_dir, outcome_names)
            for file_path in outcome_paths.values():
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Outcome file not found: {file_path}")
            return outcome_paths
        else:
            raise ValueError(
                f"Invalid outcome configuration: {outcome_names}. Should be dict or None"
            )

    @staticmethod
    def _create_outcome_dict(
        outcomes_dir: str,
        outcome_names: dict[str, str],
    ) -> dict[str, str]:
        """
        Create dictionary mapping outcome names to full file paths.
        Args:
            outcomes_dir: Directory containing outcome files
            outcome_names: Dictionary mapping outcome names to file paths
        """
        return {key: join(outcomes_dir, value) for key, value in outcome_names.items()}

    @staticmethod
    def _discover_csv_files_dict(outcomes_dir: str) -> dict[str, str]:
        """
        Auto-discover all CSV files in outcomes directory and return as dictionary.
        """
        try:
            all_files = os.listdir(outcomes_dir)
            csv_files = [f for f in all_files if f.endswith(".csv")]

            if not csv_files:
                raise ValueError(
                    f"No CSV files found in outcomes directory: {outcomes_dir}"
                )

            result = {}
            for csv_file in csv_files:
                name = os.path.splitext(csv_file)[0]  # Remove .csv extension
                path = join(outcomes_dir, csv_file)
                result[name] = path

            return result

        except OSError as e:
            raise ValueError(f"Could not read outcomes directory {outcomes_dir}: {e}")
