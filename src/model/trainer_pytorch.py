"""PyTorch model trainer with GPU acceleration for LSTM text prediction.

This module handles the training process for the PyTorch LSTM model with
GPU support, progress tracking, and checkpoint saving.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

from .lstm_model_pytorch import LSTMModelPyTorch
from ..utils.config import (
    GPU_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    TOKENIZER_CONFIG_PATH,
)
from ..utils.logger import console, print_panel, print_success, setup_logger

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# PyTorch Model Trainer Class
# ============================================================================


class ModelTrainerPyTorch:
    """PyTorch model trainer with GPU acceleration.

    This class handles the complete training process including:
    - Data preparation and batching
    - GPU-accelerated training
    - Progress tracking and visualization
    - Checkpoint saving
    - Model and tokenizer persistence

    Attributes:
        model: PyTorch LSTM model
        device: Device (cuda or cpu)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """

    def __init__(
        self,
        model: LSTMModelPyTorch,
        device: str = "cuda",
        epochs: int = EPOCHS,
        batch_size: int = GPU_BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
    ):
        """Initialize PyTorch model trainer.

        Args:
            model: PyTorch LSTM model
            device: Device to use ('cuda' or 'cpu')
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logger.info("ModelTrainerPyTorch initialized")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tokenizer_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Train the PyTorch LSTM model.

        Args:
            X_train: Training input sequences
            y_train: Training target labels
            X_val: Validation input sequences
            y_val: Validation target labels
            tokenizer_config: Tokenizer configuration dictionary

        Returns:
            Dictionary containing training results and file paths
        """
        logger.info("Starting model training")

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).long().to(self.device)
        y_train_tensor = torch.from_numpy(y_train).long().to(self.device)
        X_val_tensor = torch.from_numpy(X_val).long().to(self.device)
        y_val_tensor = torch.from_numpy(y_val).long().to(self.device)

        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

        # Display training configuration
        self._display_training_config(X_train.shape[0], X_val.shape[0])

        # Training loop
        logger.info("Training model...")
        print_panel(
            f"Training LSTM model for {self.epochs} epochs\n"
            f"Batch size: {self.batch_size}\n"
            f"Device: {self.device.upper()}",
            title="Training Started",
            style="bold green",
            border_style="green",
        )

        best_val_loss = float("inf")
        training_history = []

        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(X_train_tensor, y_train_tensor)

            # Validation phase
            val_loss, val_acc = self._validate_epoch(X_val_tensor, y_val_tensor)

            # Calculate perplexity
            train_perplexity = np.exp(train_loss)
            val_perplexity = np.exp(val_loss)

            # Store history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_perplexity": train_perplexity,
                    "val_perplexity": val_perplexity,
                }
            )

            # Display epoch results
            self._display_epoch_results(
                epoch + 1,
                train_loss,
                train_acc,
                train_perplexity,
                val_loss,
                val_acc,
                val_perplexity,
            )

            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch + 1, val_loss)

        # Save final model
        logger.info("Training complete, saving final model...")
        self._save_model()
        self._save_tokenizer(tokenizer_config)

        # Prepare results
        model_weights_path = MODELS_DIR / "lstm_weights_pytorch.pth"
        final_results = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "final_train_loss": training_history[-1]["train_loss"],
            "final_train_accuracy": training_history[-1]["train_acc"],
            "final_val_loss": training_history[-1]["val_loss"],
            "final_val_accuracy": training_history[-1]["val_acc"],
            "best_val_loss": best_val_loss,
            "model_weights_path": str(model_weights_path),
            "tokenizer_config_path": str(TOKENIZER_CONFIG_PATH),
            "training_history": training_history,
        }

        print_success(
            f"Training completed successfully!\n"
            f"Final validation accuracy: {final_results['final_val_accuracy']:.4f}\n"
            f"Best validation loss: {best_val_loss:.4f}",
            title="Training Complete",
        )

        return final_results

    def _train_epoch(
        self, X_train: torch.Tensor, y_train: torch.Tensor
    ) -> tuple[float, float]:
        """Train for one epoch.

        Args:
            X_train: Training input tensor
            y_train: Training target tensor

        Returns:
            Tuple of (average loss, average accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Create batches
        num_batches = (len(X_train) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(X_train))

            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == batch_y).sum().item()

            total_loss += loss.item() * len(batch_X)
            total_correct += correct
            total_samples += len(batch_X)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def _validate_epoch(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> tuple[float, float]:
        """Validate for one epoch.

        Args:
            X_val: Validation input tensor
            y_val: Validation target tensor

        Returns:
            Tuple of (average loss, average accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            # Create batches
            num_batches = (len(X_val) + self.batch_size - 1) // self.batch_size

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_val))

                batch_X = X_val[start_idx:end_idx]
                batch_y = y_val[start_idx:end_idx]

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == batch_y).sum().item()

                total_loss += loss.item() * len(batch_X)
                total_correct += correct
                total_samples += len(batch_X)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def _display_training_config(self, train_samples: int, val_samples: int) -> None:
        """Display training configuration table."""
        table = Table(
            title="Training Configuration",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )

        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Epochs", str(self.epochs))
        table.add_row("Batch Size", str(self.batch_size))
        table.add_row("Training Samples", f"{train_samples:,}")
        table.add_row("Validation Samples", f"{val_samples:,}")
        table.add_row("Device", self.device.upper())
        table.add_row("Vocabulary Size", f"{self.model.vocabulary_size:,}")

        console.print()
        console.print(table)
        console.print()

    def _display_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_perplexity: float,
        val_loss: float,
        val_acc: float,
        val_perplexity: float,
    ) -> None:
        """Display epoch results table."""
        table = Table(
            title=f"Epoch {epoch}/{self.epochs}",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )

        table.add_column("Metric", style="cyan")
        table.add_column("Training", style="green", justify="right")
        table.add_column("Validation", style="blue", justify="right")

        table.add_row("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}")
        table.add_row("Accuracy", f"{train_acc:.4f}", f"{val_acc:.4f}")
        table.add_row("Perplexity", f"{train_perplexity:.4f}", f"{val_perplexity:.4f}")

        console.print()
        console.print(table)

        # Progress bar
        progress = (epoch / self.epochs) * 100
        bar_length = 50
        filled = int(bar_length * epoch // self.epochs)
        bar = "█" * filled + "░" * (bar_length - filled)
        console.print(f"Progress: [{bar}] {progress:.1f}%")
        console.print()

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
        """
        checkpoint_dir = MODELS_DIR / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}_loss_{val_loss:.4f}_pytorch.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_model(self) -> None:
        """Save final model weights."""
        model_path = MODELS_DIR / "lstm_weights_pytorch.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model weights saved to {model_path}")

    def _save_tokenizer(self, tokenizer_config: dict[str, Any]) -> None:
        """Save tokenizer configuration.

        Args:
            tokenizer_config: Tokenizer configuration dictionary
        """
        with open(TOKENIZER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        logger.info(f"Tokenizer config saved to {TOKENIZER_CONFIG_PATH}")
