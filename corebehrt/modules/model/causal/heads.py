"""
Causal model heads for EHR-based inference.

This module provides a modular architecture with separate components for
sequence pooling and classification.
"""

import torch
import torch.nn as nn


class PatientRepresentationPooler(nn.Module):
    """
    Pools token-level hidden states into a single patient-level representation.
    This uses a bidirectional GRU to capture sequential information.
    """

    def __init__(self, input_size: int, output_size: int, bidirectional: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.rnn_hidden_size = output_size // 2 if bidirectional else output_size

        self.rnn = nn.GRU(
            input_size,
            self.rnn_hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Token embeddings from the base model [batch_size, seq_len, hidden_size].
            attention_mask (torch.Tensor): The attention mask [batch_size, seq_len].

        Returns:
            torch.Tensor: The patient representation vector [batch_size, output_size].
        """
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )

        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get the last valid token index for each sequence
        last_sequence_idx = lengths - 1
        batch_indices = torch.arange(output.shape[0], device=output.device)

        if self.bidirectional:
            # Concatenate the last forward output and the first backward output
            forward_output = output[
                batch_indices, last_sequence_idx, : self.rnn_hidden_size
            ]
            backward_output = output[:, 0, self.rnn_hidden_size :]
            patient_repr = torch.cat((forward_output, backward_output), dim=-1)
        else:
            # Just take the last valid token's output
            patient_repr = output[batch_indices, last_sequence_idx]

        return patient_repr


class SharedRepresentationPooler(nn.Module):
    """
    A wrapper pooler that uses a disentangling pooler internally but returns
    only the shared component (z_c) of the representation.

    This provides a clean interface for downstream tasks that need a single
    patient vector for encoding or inference.
    """

    def __init__(
        self, disentangling_pooler: PatientRepresentationPooler, split_sizes: list[int]
    ):
        super().__init__()
        self.disentangling_pooler = disentangling_pooler
        self.split_sizes = split_sizes

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Get the full, combined representation from the internal pooler
        full_repr = self.disentangling_pooler(hidden_states, attention_mask)

        # Split and return only the first part (the shared representation)
        z_c, _, _ = torch.split(full_repr, self.split_sizes, dim=-1)
        return z_c


class MLPHead(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) head for classification.

    It consists of a normalization layer followed by a two-layer feed-forward network.
    """

    def __init__(
        self, input_size: int, hidden_size_ratio: int = 2, dropout_prob: float = 0.1
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size // hidden_size_ratio, bias=True),
            nn.LayerNorm(input_size // hidden_size_ratio),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_size // hidden_size_ratio, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input representation tensor of shape [batch_size, input_size].

        Returns:
            torch.Tensor: The output logits of shape [batch_size, 1].
        """
        x = self.norm(x)
        return self.classifier(x)
