import os
from typing import Dict

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
    from a trained model and a dataset. It writes data in chunks to CSV files
    to handle very large datasets without high memory usage.
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
            save_dir (str): The directory where the encoding CSV files will be saved.
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

        os.makedirs(self.save_dir, exist_ok=True)
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

    def _process_and_save_batches(
        self, dataloader: DataLoader, patient_filepath: str, token_filepath: str
    ):
        """
        Iterates through the dataloader, processes each batch, and appends
        the results directly to CSV files to conserve memory.
        """
        is_first_batch = True

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
                    columns=[f"encoding_{i}" for i in range(token_encodings.shape[1])],
                )
                df_token_batch = pd.concat([df_token_meta, df_token_encodings], axis=1)

                # Append to token CSV
                df_token_batch.to_csv(
                    token_filepath,
                    mode="a" if not is_first_batch else "w",
                    header=is_first_batch,
                    index=False,
                )

            # --- Process and Save Patient-Level Encodings ---
            patient_encodings = outputs.patient_encodings.cpu().numpy()
            df_patient_meta = pd.DataFrame(
                {
                    PID_COL: batch[PID_COL].cpu().numpy(),
                    AGE_FEAT: batch[AGE_FEAT][:, 0]
                    .cpu()
                    .numpy(),  # First age as representative
                }
            )
            df_patient_encodings = pd.DataFrame(
                patient_encodings,
                columns=[f"encoding_{i}" for i in range(patient_encodings.shape[1])],
            )
            df_patient_batch = pd.concat(
                [df_patient_meta, df_patient_encodings], axis=1
            )

            # Append to patient CSV
            df_patient_batch.to_csv(
                patient_filepath,
                mode="a" if not is_first_batch else "w",
                header=is_first_batch,
                index=False,
            )

            is_first_batch = False  # After the first batch, we always append

    def save(self):
        """
        The main public method to run the entire encoding and saving process.
        """
        dataloader = self._get_dataloader()

        patient_filepath = os.path.join(self.save_dir, "patient_encodings.csv")
        token_filepath = os.path.join(self.save_dir, "token_encodings.csv")

        # Ensure files are clean before starting if they exist
        if os.path.exists(patient_filepath):
            os.remove(patient_filepath)
        if os.path.exists(token_filepath):
            os.remove(token_filepath)

        self._process_and_save_batches(dataloader, patient_filepath, token_filepath)

        print("\n--- Encoding saving process complete. ---")
        print(f"Patient encodings saved to: {patient_filepath}")
        print(f"Token encodings saved to: {token_filepath}")
