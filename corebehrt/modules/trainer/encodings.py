import os
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from corebehrt.constants.data import (
    AGE_FEAT,
    MASK_TOKEN,
    PAD_TOKEN,
    PID_COL,
    SEP_TOKEN,
    CONCEPT_FEAT,
)
from corebehrt.modules.preparation.causal.dataset import CausalPatientDataset


class EncodingSaver:
    """
    A class to generate and save patient and token-level encodings
    from a trained model and a dataset. It saves patient encodings to a single
    parquet file and token encodings to parquet shards in a subdirectory.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: CausalPatientDataset,
        vocab: Dict[str, int],
        save_dir: str,
        batch_size: int = 64,
    ):
        """
        Initializes the EncodingSaver.
        Args:
            model: The trained PyTorch model.
            dataset: The dataset (e.g., val_dataset, test_dataset) to process.
            vocab (Dict[str, int]): The vocabulary mapping tokens to integer IDs.
            save_dir (str): The directory where the encoding files will be saved.
            batch_size (int): The batch size for processing the data.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.vocab = vocab

        # Initialize ignore token IDs
        self.ignore_token_ids = self._get_ignore_token_ids()

        # Initialize storage
        self.patient_batches: List[pd.DataFrame] = []
        self.token_shard_counter = 0

        # Setup directories
        self._setup_directories()

        self._print_initialization_info()

    def _get_ignore_token_ids(self) -> set:
        """Get token IDs to ignore during processing."""
        ignore_codes = [MASK_TOKEN, PAD_TOKEN, SEP_TOKEN]
        return {self.vocab[code] for code in ignore_codes if code in self.vocab}

    def _setup_directories(self):
        """Create necessary directories for saving encodings."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.token_save_dir = os.path.join(self.save_dir, "token_encodings")
        os.makedirs(self.token_save_dir, exist_ok=True)

    def _print_initialization_info(self):
        """Print initialization information."""
        print(f"EncodingSaver initialized. Encodings will be saved to: {self.save_dir}")
        print(f"Using device: {self.device}")
        print(f"Ignoring token IDs: {self.ignore_token_ids}")

    def _get_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the dataset."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
        )

    def _create_token_mask(self, concepts: torch.Tensor) -> torch.Tensor:
        """Create a mask for valid tokens (excluding ignored tokens)."""
        mask = torch.ones_like(concepts, dtype=torch.bool, device=self.device)
        for token_id in self.ignore_token_ids:
            mask &= concepts != token_id
        return mask

    def _get_last_valid_ages(self, batch: Dict, mask: torch.Tensor) -> List[float]:
        """Get the last valid age for each patient after masking."""
        concepts = batch[CONCEPT_FEAT]
        last_valid_ages = []

        for i in range(concepts.shape[0]):  # For each patient in the batch
            valid_positions = torch.where(mask[i])[0]
            if len(valid_positions) > 0:
                last_valid_pos = valid_positions[-1].item()
                age_value = batch[AGE_FEAT][i, last_valid_pos].cpu().numpy().item()
            else:
                # Fallback to last position if no valid tokens
                age_value = batch[AGE_FEAT][i, -1].cpu().numpy().item()

            last_valid_ages.append(float(age_value))

        return last_valid_ages

    def _process_token_encodings(self, batch: Dict, outputs, mask: torch.Tensor):
        """Process and save token-level encodings for the current batch."""
        if not mask.any():  # Skip if no valid tokens in the batch
            return

        concepts = batch[CONCEPT_FEAT]
        pids = batch[PID_COL]
        expanded_pids = pids.unsqueeze(1).expand(-1, concepts.shape[1])

        # Create metadata DataFrame
        token_metadata = pd.DataFrame(
            {
                PID_COL: expanded_pids[mask].cpu().numpy(),
                AGE_FEAT: batch[AGE_FEAT][mask].cpu().numpy(),
                CONCEPT_FEAT: concepts[mask].cpu().numpy(),
            }
        )

        # Create encodings DataFrame
        token_encodings = outputs.token_encodings[mask].cpu().numpy()
        encoding_columns = [f"x{i}" for i in range(token_encodings.shape[1])]
        token_encodings_df = pd.DataFrame(token_encodings, columns=encoding_columns)

        # Combine and save
        token_batch_df = pd.concat([token_metadata, token_encodings_df], axis=1)
        self._save_token_shard(token_batch_df)

    def _save_token_shard(self, token_batch_df: pd.DataFrame):
        """Save a token encoding batch as a parquet shard."""
        shard_path = os.path.join(
            self.token_save_dir, f"{self.token_shard_counter}.parquet"
        )
        token_batch_df.to_parquet(shard_path, index=False)
        self.token_shard_counter += 1

    def _process_patient_encodings(self, batch: Dict, outputs, mask: torch.Tensor):
        """Process and accumulate patient-level encodings for the current batch."""
        patient_encodings = outputs.patient_encodings.cpu().numpy()
        last_valid_ages = self._get_last_valid_ages(batch, mask)

        # Create metadata DataFrame
        patient_metadata = pd.DataFrame(
            {
                PID_COL: batch[PID_COL].cpu().numpy(),
                AGE_FEAT: last_valid_ages,
            }
        )

        # Create encodings DataFrame
        encoding_columns = [f"x{i}" for i in range(patient_encodings.shape[1])]
        patient_encodings_df = pd.DataFrame(patient_encodings, columns=encoding_columns)

        # Combine and accumulate
        patient_batch_df = pd.concat([patient_metadata, patient_encodings_df], axis=1)
        self.patient_batches.append(patient_batch_df)

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to the appropriate device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _process_single_batch(self, batch: Dict):
        """Process a single batch to generate and save encodings."""
        batch = self._move_batch_to_device(batch)

        with torch.no_grad():
            outputs = self.model(batch, return_encodings=True)

        # Create mask for valid tokens
        concepts = batch[CONCEPT_FEAT]
        mask = self._create_token_mask(concepts)

        # Process both token and patient encodings
        self._process_token_encodings(batch, outputs, mask)
        self._process_patient_encodings(batch, outputs, mask)

    def _process_and_save_batches(self, dataloader: DataLoader):
        """
        Iterates through the dataloader, processes each batch, and saves
        token encodings as parquet shards while accumulating patient encodings.
        """
        for batch in tqdm(dataloader, desc="Generating and Saving Encodings"):
            self._process_single_batch(batch)

    def _cleanup_existing_files(self):
        """Clean up existing encoding files before saving new ones."""
        # Clean up patient encodings file
        patient_filepath = os.path.join(self.save_dir, "patient_encodings.parquet")
        if os.path.exists(patient_filepath):
            os.remove(patient_filepath)

        # Clean up token encodings directory
        if os.path.exists(self.token_save_dir):
            for file in os.listdir(self.token_save_dir):
                file_path = os.path.join(self.token_save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _save_patient_encodings(self):
        """Save accumulated patient encodings to a single parquet file."""
        patient_filepath = os.path.join(self.save_dir, "patient_encodings.parquet")

        if self.patient_batches:
            all_patients_df = pd.concat(self.patient_batches, ignore_index=True)
            all_patients_df.to_parquet(patient_filepath, index=False)
            print(f"\nPatient encodings saved to: {patient_filepath}")
            print(f"Total patients: {len(all_patients_df)}")
        else:
            print("\nNo patient encodings to save.")

    def _print_completion_summary(self):
        """Print summary information about the encoding process."""
        print(
            f"Token encodings saved as {self.token_shard_counter} parquet shards in: {self.token_save_dir}"
        )
        print("\n--- Encoding saving process complete. ---")

    def save(self):
        """
        The main public method to run the entire encoding and saving process.
        """
        dataloader = self._get_dataloader()

        self._cleanup_existing_files()
        self._process_and_save_batches(dataloader)
        self._save_patient_encodings()
        self._print_completion_summary()
