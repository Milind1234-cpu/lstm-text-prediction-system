"""PyTorch LSTM model architecture for text prediction with GPU support.

This module defines the bidirectional LSTM neural network architecture
for next-word prediction using PyTorch with CUDA GPU acceleration.
"""

import torch
import torch.nn as nn
from rich.table import Table

from ..utils.config import (
    BIDIRECTIONAL_LSTM_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    SEQUENCE_LENGTH,
    UNIDIRECTIONAL_LSTM_UNITS,
    VOCABULARY_SIZE,
)
from ..utils.logger import console, print_panel, print_success, setup_logger

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# PyTorch LSTM Model Class
# ============================================================================


class LSTMModelPyTorch(nn.Module):
    """Bidirectional LSTM model for text prediction using PyTorch.

    This class implements a neural network architecture with:
    - Embedding layer (256 dimensions, 10,000 vocabulary)
    - Bidirectional LSTM layer (512 units)
    - Dropout layer (0.3 rate)
    - Unidirectional LSTM layer (256 units)
    - Dropout layer (0.3 rate)
    - Dense output layer (10,000 units, softmax activation)

    Attributes:
        embedding: Embedding layer
        bidirectional_lstm: Bidirectional LSTM layer
        dropout1: First dropout layer
        unidirectional_lstm: Unidirectional LSTM layer
        dropout2: Second dropout layer
        output: Output dense layer
        device: Device (cuda or cpu)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize PyTorch LSTM model.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__()

        self.vocabulary_size = VOCABULARY_SIZE
        self.sequence_length = SEQUENCE_LENGTH
        self.embedding_dim = EMBEDDING_DIM
        self.bidirectional_units = BIDIRECTIONAL_LSTM_UNITS
        self.unidirectional_units = UNIDIRECTIONAL_LSTM_UNITS
        self.dropout_rate = DROPOUT_RATE
        self.device = device

        logger.info("Initializing PyTorch LSTM model architecture")

        # Build the model layers
        self.embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        self.bidirectional_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.bidirectional_units // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.unidirectional_lstm = nn.LSTM(
            input_size=self.bidirectional_units,
            hidden_size=self.unidirectional_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.output = nn.Linear(self.unidirectional_units, self.vocabulary_size)

        # Move model to device
        self.to(device)

        logger.info(f"PyTorch LSTM model initialized on {device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Output tensor of shape (batch_size, vocabulary_size)
        """
        # Embedding layer
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Bidirectional LSTM
        x, _ = self.bidirectional_lstm(x)  # (batch, seq_len, bidirectional_units)

        # Dropout
        x = self.dropout1(x)

        # Unidirectional LSTM (take only last output)
        x, _ = self.unidirectional_lstm(x)  # (batch, seq_len, unidirectional_units)
        x = x[:, -1, :]  # Take last timestep (batch, unidirectional_units)

        # Dropout
        x = self.dropout2(x)

        # Output layer
        x = self.output(x)  # (batch, vocabulary_size)

        return x

    def summary(self) -> None:
        """Display model summary using Rich formatting."""
        print_panel(
            "PyTorch LSTM Text Prediction Model Architecture\n"
            f"Vocabulary Size: {self.vocabulary_size:,}\n"
            f"Sequence Length: {self.sequence_length}\n"
            f"Embedding Dimension: {self.embedding_dim}\n"
            f"Bidirectional LSTM Units: {self.bidirectional_units}\n"
            f"Unidirectional LSTM Units: {self.unidirectional_units}\n"
            f"Dropout Rate: {self.dropout_rate}\n"
            f"Device: {self.device}",
            title="Model Configuration",
            style="cyan",
            border_style="cyan",
        )

        # Create Rich table for model layers
        table = Table(
            title="Model Layers",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )

        table.add_column("Layer (type)", style="cyan", no_wrap=True)
        table.add_column("Output Shape", style="green")
        table.add_column("Param #", style="yellow", justify="right")

        # Calculate parameters for each layer
        embedding_params = self.vocabulary_size * self.embedding_dim
        bidirectional_params = (
            4
            * (self.embedding_dim + self.bidirectional_units // 2 + 1)
            * (self.bidirectional_units // 2)
            * 2
        )
        unidirectional_params = (
            4
            * (self.bidirectional_units + self.unidirectional_units + 1)
            * self.unidirectional_units
        )
        output_params = (self.unidirectional_units + 1) * self.vocabulary_size

        table.add_row(
            "embedding (Embedding)",
            f"(None, {self.sequence_length}, {self.embedding_dim})",
            f"{embedding_params:,}",
        )
        table.add_row(
            "bidirectional_lstm (LSTM)",
            f"(None, {self.sequence_length}, {self.bidirectional_units})",
            f"{bidirectional_params:,}",
        )
        table.add_row(
            "dropout_1 (Dropout)", f"(None, {self.sequence_length}, {self.bidirectional_units})", "0"
        )
        table.add_row(
            "unidirectional_lstm (LSTM)",
            f"(None, {self.unidirectional_units})",
            f"{unidirectional_params:,}",
        )
        table.add_row("dropout_2 (Dropout)", f"(None, {self.unidirectional_units})", "0")
        table.add_row(
            "output (Linear)", f"(None, {self.vocabulary_size})", f"{output_params:,}"
        )

        console.print()
        console.print(table)
        console.print()

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Create parameters summary table
        params_table = Table(
            title="Parameters Summary",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )

        params_table.add_column("Parameter Type", style="cyan")
        params_table.add_column("Count", style="green", justify="right")

        params_table.add_row("Total params", f"{total_params:,}")
        params_table.add_row("Trainable params", f"{trainable_params:,}")
        params_table.add_row("Non-trainable params", "0")

        console.print(params_table)
        console.print()

        print_success(
            f"Model has {total_params:,} total parameters "
            f"({trainable_params:,} trainable)",
            title="Model Summary Complete",
        )


def create_pytorch_lstm_model(device: str = "cuda") -> LSTMModelPyTorch:
    """Create and initialize PyTorch LSTM model.

    Args:
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Initialized LSTMModelPyTorch instance
    """
    return LSTMModelPyTorch(device=device)
