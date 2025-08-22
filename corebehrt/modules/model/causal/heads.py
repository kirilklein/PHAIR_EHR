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

    def __init__(self, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn_hidden_size = hidden_size // 2 if bidirectional else hidden_size

        self.rnn = nn.GRU(
            hidden_size,
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
            torch.Tensor: The patient representation vector [batch_size, hidden_size].
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


class MLPHead(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) head for classification.

    It consists of a normalization layer followed by a two-layer feed-forward network.
    """

    def __init__(
        self, input_size: int, hidden_size_ratio: int = 2, dropout_prob: float = 0.1
    ):
        super().__init__()
        intermediate_size = input_size // hidden_size_ratio
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, intermediate_size),
            nn.GELU(),
            # nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input representation tensor of shape [batch_size, input_size].

        Returns:
            torch.Tensor: The output logits of shape [batch_size, 1].
        """
        return self.classifier(x)


class CLSPooler(nn.Module):
    """
    Pools the sequence output by simply selecting the hidden state corresponding
    to the last valid token in the sequence.

    This is designed to work with models where a special token (like [CLS])
    is appended at the end to act as an aggregate sequence representation.
    """

    def __init__(self, **kwargs):
        # The __init__ is simple as there are no layers to create.
        # **kwargs is included to accept unused arguments if the config
        # sends them, making it a flexible drop-in replacement.
        super().__init__()

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Token embeddings from the base model
                                          [batch_size, seq_len, hidden_size].
            attention_mask (torch.Tensor): The attention mask [batch_size, seq_len].

        Returns:
            torch.Tensor: The representation of the last token for each sequence
                          [batch_size, hidden_size].
        """
        # Calculate the length of each sequence by summing the attention mask
        lengths = attention_mask.sum(dim=1).long()

        # Get the index of the last token for each sequence (length - 1)
        last_token_indices = lengths - 1

        # Create a tensor of batch indices [0, 1, 2, ...]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        # Use advanced indexing to select the hidden state of the last token
        # for each item in the batch.
        cls_representation = hidden_states[batch_indices, last_token_indices]

        return cls_representation
