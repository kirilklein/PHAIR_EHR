"""
Causal model heads for EHR-based inference.

This module provides heads for causal inference tasks, including
It allows to train on exposure and outcome at the same time.
"""

import torch
import torch.nn as nn


class CausalFineTuneHead(nn.Module):
    """
    Classification head for causal inference tasks.

    This component applies a BiGRU for sequence pooling followed by a linear classifier.
    It can optionally incorporate exposure status information into the outcome prediction.

    Attributes:
        pool (CausalBiGRU): Pooling layer with BiGRU
        classifier (CausalMLP): MLP classification layer
    """

    def __init__(self, hidden_size: int, with_exposure: bool = False):
        super().__init__()
        self.pool = CausalBiGRU(hidden_size, with_exposure)
        self.classifier = CausalMLP(self.pool.pooled_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        exposure_status: torch.Tensor = None,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the causal fine-tuning head.

        Args:
            hidden_states: Token-level hidden states from the encoder
            attention_mask: Mask indicating which tokens are valid (1) or padding (0)
            exposure_status: Treatment/exposure indicator
            return_embedding: If True, return the pooled embedding instead of logits

        Returns:
            torch.Tensor: Classification logits or pooled embeddings
        """
        pooled = self.pool(hidden_states, attention_mask, exposure_status)

        if return_embedding:
            return pooled

        return self.classifier(pooled)


class CausalBiGRU(nn.Module):
    """
    Bidirectional GRU for sequence pooling with optional exposure status incorporation.

    This component extracts a fixed-size vector representation from variable-length sequences
    using a bidirectional GRU. It returns the pooled representation without classification.

    Attributes:
        hidden_size (int): Size of the input hidden states
        rnn_hidden_size (int): Size of each GRU direction (half of hidden_size)
        rnn (nn.GRU): Bidirectional GRU layer
        with_exposure (bool): Whether to incorporate exposure information
        pooled_size (int): Size of the pooled output
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_hidden_size = hidden_size // 2

        # Bidirectional GRU
        self.rnn = nn.GRU(
            hidden_size, self.rnn_hidden_size, batch_first=True, bidirectional=True
        )

        # Calculate pooled output size
        self.pooled_size = hidden_size
        self.norm = torch.nn.LayerNorm(self.pooled_size)

        # Store last pooled output for analysis/debugging
        self.last_pooled_output = None

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [batch_size, seq_len, hidden_size]
        attention_mask: torch.Tensor,  # Shape: [batch_size, seq_len]
    ) -> torch.Tensor:
        """
        Forward pass for the causal BiGRU.

        Args:
            hidden_states: Token-level hidden states from the encoder
            attention_mask: Mask indicating which tokens are valid (1) or padding (0)
            exposure_status: Treatment/exposure indicator (0/1 or -1/1)

        Returns:
            torch.Tensor: Pooled embeddings [batch_size, pooled_size]
        """
        # Get sequence lengths for pack_padded_sequence
        lengths = attention_mask.sum(dim=1).cpu()

        # Pack the padded sequence for efficient RNN processing
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )

        # Pass through the RNN
        output, _ = self.rnn(packed)

        # Unpack back to a padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get indices of last valid token for each sequence
        last_sequence_idx = lengths - 1
        batch_indices = torch.arange(output.shape[0], device=output.device)

        # Extract forward GRU's output at the last valid token
        forward_output = output[
            batch_indices, last_sequence_idx, : self.rnn_hidden_size
        ]

        # Extract backward GRU's output at the first token
        backward_output = output[:, 0, self.rnn_hidden_size :]

        # Concatenate both directions
        x = torch.cat((forward_output, backward_output), dim=-1)

        x = self.norm(x)
        # Store the pooled output for inspection/debugging
        self.last_pooled_output = x

        return x


class CausalMLP(nn.Module):
    """
    MLP classifier for causal inference tasks.

    Simple MLP that takes pooled embeddings and outputs classification logits.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            nn.Linear(input_size, input_size // 2, bias=True),
            nn.LayerNorm(input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size // 2, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP classifier.

        Args:
            x: Input embeddings [batch_size, input_size]

        Returns:
            torch.Tensor: Classification logits [batch_size, 1]
        """
        return self.classifier(x)
