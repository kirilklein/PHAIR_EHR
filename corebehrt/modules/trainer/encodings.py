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
        batch_size: int = 32,
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

        ignore_codes = [MASK_TOKEN, PAD_TOKEN, SEP_TOKEN]
        self.ignore_token_ids = {
            self.vocab[code] for code in ignore_codes if code in self.vocab
        }

        # Accumulator for patient encodings
        self.patient_batches: List[pd.DataFrame] = []

        # Token encoding shard counter
        self.token_shard_counter = 0

        os.makedirs(self.save_dir, exist_ok=True)

        # Create token_encodings subdirectory
        self.token_save_dir = os.path.join(self.save_dir, "token_encodings")
        os.makedirs(self.token_save_dir, exist_ok=True)

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

    def _process_and_save_batches(self, dataloader: DataLoader):
        """
        Iterates through the dataloader, processes each batch, and saves
        token encodings as parquet shards while accumulating patient encodings.
        """
        for batch in tqdm(dataloader, desc="Generating and Saving Encodings"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            with torch.no_grad():
                outputs = self.model(batch, return_encodings=True)

            # --- Process and Save Token-Level Encodings ---
            concepts = batch[CONCEPT_FEAT]
            mask = torch.ones_like(concepts, dtype=torch.bool, device=self.device)
            for token_id in self.ignore_token_ids:
                mask &= concepts != token_id

            if mask.any():  # Proceed only if there are valid tokens in the batch
                pids = batch[PID_COL]
                expanded_pids = pids.unsqueeze(1).expand(-1, concepts.shape[1])

                # Create DataFrame for the current batch's valid tokens
                df_token_meta = pd.DataFrame(
                    {
                        PID_COL: expanded_pids[mask].cpu().numpy(),
                        AGE_FEAT: batch[AGE_FEAT][mask].cpu().numpy(),
                        CONCEPT_FEAT: concepts[mask].cpu().numpy(),
                    }
                )
                token_encodings = outputs.token_encodings[mask].cpu().numpy()
                df_token_encodings = pd.DataFrame(
                    token_encodings,
                    columns=[f"x{i}" for i in range(token_encodings.shape[1])],
                )
                df_token_batch = pd.concat([df_token_meta, df_token_encodings], axis=1)

                # Save token batch as parquet shard
                token_shard_path = os.path.join(
                    self.token_save_dir, f"{self.token_shard_counter}.parquet"
                )
                df_token_batch.to_parquet(token_shard_path, index=False)
                self.token_shard_counter += 1

            # --- Process and Accumulate Patient-Level Encodings ---
            patient_encodings = outputs.patient_encodings.cpu().numpy()

            # Get the last valid (non-ignored) age for each patient
            concepts = batch[CONCEPT_FEAT]
            mask = torch.ones_like(concepts, dtype=torch.bool, device=self.device)
            for token_id in self.ignore_token_ids:
                mask &= concepts != token_id

            # Find the last valid token position for each patient
            last_valid_ages = []
            for i in range(concepts.shape[0]):  # For each patient in the batch
                valid_positions = torch.where(mask[i])[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1].item()
                    last_valid_ages.append(
                        float(batch[AGE_FEAT][i, last_valid_pos].cpu().numpy().item())
                    )
                else:
                    # Fallback to last position if no valid tokens (shouldn't happen normally)
                    last_valid_ages.append(
                        float(batch[AGE_FEAT][i, -1].cpu().numpy().item())
                    )

            df_patient_meta = pd.DataFrame(
                {
                    PID_COL: batch[PID_COL].cpu().numpy(),
                    AGE_FEAT: last_valid_ages,  # Use last valid age after masking
                }
            )
            df_patient_encodings = pd.DataFrame(
                patient_encodings,
                columns=[f"x{i}" for i in range(patient_encodings.shape[1])],
            )
            df_patient_batch = pd.concat(
                [df_patient_meta, df_patient_encodings], axis=1
            )

            # Accumulate patient batch
            self.patient_batches.append(df_patient_batch)

    def save(self):
        """
        The main public method to run the entire encoding and saving process.
        """
        dataloader = self._get_dataloader()

        # Clean up existing files/directories
        patient_filepath = os.path.join(self.save_dir, "patient_encodings.parquet")
        if os.path.exists(patient_filepath):
            os.remove(patient_filepath)

        # Clean up token encodings directory
        if os.path.exists(self.token_save_dir):
            for file in os.listdir(self.token_save_dir):
                file_path = os.path.join(self.token_save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        self._process_and_save_batches(dataloader)

        # Save accumulated patient encodings to single parquet file
        if self.patient_batches:
            df_all_patients = pd.concat(self.patient_batches, ignore_index=True)
            df_all_patients.to_parquet(patient_filepath, index=False)
            print(f"\nPatient encodings saved to: {patient_filepath}")
            print(f"Total patients: {len(df_all_patients)}")
        else:
            print("\nNo patient encodings to save.")

        print(
            f"Token encodings saved as {self.token_shard_counter} parquet shards in: {self.token_save_dir}"
        )
        print("\n--- Encoding saving process complete. ---")
