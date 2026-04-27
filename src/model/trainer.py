"""Model training module for LSTM text prediction.

This module provides the ModelTrainer class for training the LSTM model with
GPU acceleration, progress tracking, validation evaluation, and model persistence.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf
from rich.table import Table
from tensorflow import keras

from .gpu_manager import GPUManager
from .lstm_model import LSTMModel
from ..utils.config import (
    CPU_BATCH_SIZE,
    EPOCHS,
    GPU_BATCH_SIZE,
    MODEL_ARCHITECTURE_PATH,
    MODEL_WEIGHTS_PATH,
    MODELS_DIR,
    TOKENIZER_CONFIG_PATH,
    VOCABULARY_SIZE,
)
from ..utils.logger import (
    console,
    create_progress_bar,
    create_table,
    print_error,
    print_panel,
    print_success,
    setup_logger,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# ModelTrainer Class
# ============================================================================


class ModelTrainer:
    """Trains LSTM model with GPU acceleration and progress tracking.

    This class handles the complete training pipeline including:
    - GPU/CPU batch size selection
    - Training loop with epoch progress display
    - Validation evaluation after each epoch
    - Model checkpoint saving after each epoch
    - Final model architecture, weights, and tokenizer saving
    - Rich progress bars and tables for training metrics

    Attributes:
        model: LSTMModel instance to train
        gpu_manager: GPUManager instance for hardware configuration
        batch_size: Batch size for training (256 for CPU, 512 for GPU)
        epochs: Number of training epochs (default: 50)
        history: Training history from model.fit()
    """

    def __init__(
        self,
        model: LSTMModel,
        gpu_manager: GPUManager,
        epochs: int | None = None,
    ) -> None:
        """Initialize ModelTrainer.

        Args:
            model: LSTMModel instance to train
            gpu_manager: GPUManager instance for hardware configuration
            epochs: Number of training epochs. Defaults to EPOCHS from config.
        """
        self.model: Final[LSTMModel] = model
        self.gpu_manager: Final[GPUManager] = gpu_manager
        self.epochs: Final[int] = epochs if epochs is not None else EPOCHS

        # Determine batch size based on GPU availability
        self.batch_size: Final[int] = self.gpu_manager.get_batch_size(
            cpu_batch_size=CPU_BATCH_SIZE,
            gpu_batch_size=GPU_BATCH_SIZE,
        )

        self.history: keras.callbacks.History | None = None

        logger.info("ModelTrainer initialized")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(
            f"Device: {self.gpu_manager.device_name} "
            f"({'GPU' if self.gpu_manager.gpu_available else 'CPU'})"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tokenizer_config: dict[str, object],
    ) -> dict[str, object]:
        """Train the LSTM model with progress tracking and checkpointing.

        This method performs the complete training pipeline:
        1. Converts target labels to one-hot encoding
        2. Displays training configuration
        3. Creates checkpoint callback for saving after each epoch
        4. Trains model with progress display
        5. Displays training results
        6. Saves final model and tokenizer

        Args:
            X_train: Training input sequences (shape: [num_samples, sequence_length])
            y_train: Training target labels (shape: [num_samples])
            X_val: Validation input sequences (shape: [num_samples, sequence_length])
            y_val: Validation target labels (shape: [num_samples])
            tokenizer_config: Tokenizer configuration dictionary to save

        Returns:
            Dictionary containing training results:
                - 'epochs': Number of epochs trained
                - 'batch_size': Batch size used
                - 'train_samples': Number of training samples
                - 'val_samples': Number of validation samples
                - 'final_train_loss': Final training loss
                - 'final_train_accuracy': Final training accuracy
                - 'final_val_loss': Final validation loss
                - 'final_val_accuracy': Final validation accuracy
                - 'model_architecture_path': Path to saved model architecture
                - 'model_weights_path': Path to saved model weights
                - 'tokenizer_config_path': Path to saved tokenizer config
        """
        logger.info("Starting model training")

        # Convert target labels to one-hot encoding
        logger.info("Converting targets to one-hot encoding...")
        y_train_categorical = keras.utils.to_categorical(y_train, num_classes=VOCABULARY_SIZE)
        y_val_categorical = keras.utils.to_categorical(y_val, num_classes=VOCABULARY_SIZE)

        logger.info(f"Training data shape: X={X_train.shape}, y={y_train_categorical.shape}")
        logger.info(f"Validation data shape: X={X_val.shape}, y={y_val_categorical.shape}")

        # Display training configuration
        self._display_training_config(
            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        # Create checkpoint callback
        checkpoint_dir = MODELS_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "epoch_{epoch:02d}_loss_{val_loss:.4f}.weights.h5"

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        )

        # Create custom callback for epoch progress display
        progress_callback = EpochProgressCallback(
            epochs=self.epochs,
            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        # Train the model
        logger.info("Training model...")
        print_panel(
            f"Training LSTM model for {self.epochs} epochs\n"
            f"Batch size: {self.batch_size}\n"
            f"Device: {self.gpu_manager.device_name}",
            title="Training Started",
            style="bold green",
            border_style="green",
        )

        try:
            self.history = self.model.get_model().fit(
                X_train,
                y_train_categorical,
                validation_data=(X_val, y_val_categorical),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[checkpoint_callback, progress_callback],
                verbose=0,  # Disable default progress bar
            )

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            print_error(
                f"Training failed with error: {e}",
                title="Training Error",
            )
            raise

        # Display training results
        self._display_training_results()

        # Save final model and tokenizer
        logger.info("Saving final model and tokenizer...")
        self._save_model_and_tokenizer(tokenizer_config)

        # Prepare results dictionary
        results = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'model_architecture_path': str(MODEL_ARCHITECTURE_PATH),
            'model_weights_path': str(MODEL_WEIGHTS_PATH),
            'tokenizer_config_path': str(TOKENIZER_CONFIG_PATH),
        }

        logger.info("Training pipeline completed")
        return results

    def _display_training_config(
        self,
        train_samples: int,
        val_samples: int,
    ) -> None:
        """Display training configuration with Rich table.

        Args:
            train_samples: Number of training samples
            val_samples: Number of validation samples
        """
        config_table = create_table(
            "Training Configuration",
            "Parameter",
            "Value",
        )

        config_table.add_row("Epochs", str(self.epochs))
        config_table.add_row("Batch Size", str(self.batch_size))
        config_table.add_row("Training Samples", f"{train_samples:,}")
        config_table.add_row("Validation Samples", f"{val_samples:,}")
        config_table.add_row(
            "Device",
            f"{self.gpu_manager.device_name} "
            f"({'GPU' if self.gpu_manager.gpu_available else 'CPU'})",
        )
        config_table.add_row("Vocabulary Size", f"{VOCABULARY_SIZE:,}")

        console.print()
        console.print(config_table)
        console.print()

    def _display_training_results(self) -> None:
        """Display training results with Rich tables."""
        if self.history is None:
            logger.warning("No training history available")
            return

        # Create metrics table for final epoch
        metrics_table = create_table(
            f"Training Results (Epoch {self.epochs})",
            "Metric",
            "Training",
            "Validation",
        )

        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]

        metrics_table.add_row(
            "Loss",
            f"{final_train_loss:.4f}",
            f"{final_val_loss:.4f}",
        )
        metrics_table.add_row(
            "Accuracy",
            f"{final_train_acc:.4f}",
            f"{final_val_acc:.4f}",
        )

        # Add perplexity if available
        if 'perplexity' in self.history.history:
            final_train_perplexity = self.history.history['perplexity'][-1]
            final_val_perplexity = self.history.history['val_perplexity'][-1]
            metrics_table.add_row(
                "Perplexity",
                f"{final_train_perplexity:.4f}",
                f"{final_val_perplexity:.4f}",
            )

        console.print()
        console.print(metrics_table)
        console.print()

        # Create epoch-by-epoch summary table (show first 5, last 5)
        epoch_table = create_table(
            "Epoch Summary",
            "Epoch",
            "Train Loss",
            "Train Acc",
            "Val Loss",
            "Val Acc",
        )

        # Determine which epochs to show
        total_epochs = len(self.history.history['loss'])
        if total_epochs <= 10:
            epochs_to_show: list[int] | range = range(total_epochs)
        else:
            epochs_to_show = list(range(5)) + list(range(total_epochs - 5, total_epochs))

        prev_epoch = -1
        for i in epochs_to_show:
            # Add separator if there's a gap
            if i > prev_epoch + 1:
                epoch_table.add_row("...", "...", "...", "...", "...")

            epoch_table.add_row(
                str(i + 1),
                f"{self.history.history['loss'][i]:.4f}",
                f"{self.history.history['accuracy'][i]:.4f}",
                f"{self.history.history['val_loss'][i]:.4f}",
                f"{self.history.history['val_accuracy'][i]:.4f}",
            )
            prev_epoch = i

        console.print(epoch_table)
        console.print()

        print_success(
            f"Training completed successfully!\n"
            f"Final validation accuracy: {final_val_acc:.4f}\n"
            f"Final validation loss: {final_val_loss:.4f}",
            title="Training Complete",
        )

    def _save_model_and_tokenizer(
        self,
        tokenizer_config: dict[str, object],
    ) -> None:
        """Save model architecture, weights, and tokenizer configuration.

        Args:
            tokenizer_config: Tokenizer configuration dictionary to save
        """
        # Save model architecture to JSON
        logger.info(f"Saving model architecture to {MODEL_ARCHITECTURE_PATH}")
        self.model.save_architecture(str(MODEL_ARCHITECTURE_PATH))

        # Save model weights to H5
        logger.info(f"Saving model weights to {MODEL_WEIGHTS_PATH}")
        self.model.save_weights(str(MODEL_WEIGHTS_PATH))

        # Save tokenizer configuration to JSON
        logger.info(f"Saving tokenizer configuration to {TOKENIZER_CONFIG_PATH}")
        with open(TOKENIZER_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        # Display saved files
        files_table = create_table(
            "Saved Files",
            "File Type",
            "Path",
        )

        files_table.add_row("Model Architecture", str(MODEL_ARCHITECTURE_PATH))
        files_table.add_row("Model Weights", str(MODEL_WEIGHTS_PATH))
        files_table.add_row("Tokenizer Config", str(TOKENIZER_CONFIG_PATH))

        console.print()
        console.print(files_table)
        console.print()

        print_success(
            "Model and tokenizer saved successfully!",
            title="Save Complete",
        )

    def get_history(self) -> keras.callbacks.History | None:
        """Get training history.

        Returns:
            Training history object or None if training hasn't been performed
        """
        return self.history


# ============================================================================
# Custom Callback for Epoch Progress Display
# ============================================================================


class EpochProgressCallback(keras.callbacks.Callback):
    """Custom callback for displaying epoch progress with Rich.

    This callback displays training progress after each epoch using Rich
    tables and progress bars. It shows loss, accuracy, and perplexity metrics
    for both training and validation sets.

    Attributes:
        epochs: Total number of epochs
        train_samples: Number of training samples
        val_samples: Number of validation samples
        current_epoch: Current epoch number
    """

    def __init__(
        self,
        epochs: int,
        train_samples: int,
        val_samples: int,
    ) -> None:
        """Initialize EpochProgressCallback.

        Args:
            epochs: Total number of epochs
            train_samples: Number of training samples
            val_samples: Number of validation samples
        """
        super().__init__()
        self.epochs = epochs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.current_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dictionary of metrics from the epoch
        """
        if logs is None:
            logs = {}

        self.current_epoch = epoch + 1

        # Create metrics table for this epoch
        metrics_table = Table(
            title=f"Epoch {self.current_epoch}/{self.epochs}",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )

        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Training", style="green", justify="right")
        metrics_table.add_column("Validation", style="yellow", justify="right")

        # Add loss
        train_loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        metrics_table.add_row("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}")

        # Add accuracy
        train_acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        metrics_table.add_row("Accuracy", f"{train_acc:.4f}", f"{val_acc:.4f}")

        # Add perplexity if available
        if 'perplexity' in logs:
            train_perplexity = logs.get('perplexity', 0.0)
            val_perplexity = logs.get('val_perplexity', 0.0)
            metrics_table.add_row(
                "Perplexity",
                f"{train_perplexity:.4f}",
                f"{val_perplexity:.4f}",
            )

        console.print()
        console.print(metrics_table)

        # Display progress bar
        progress_pct = (self.current_epoch / self.epochs) * 100
        progress_bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
        console.print(
            f"[cyan]Progress: [{progress_bar}] {progress_pct:.1f}%[/cyan]"
        )
        console.print()


# ============================================================================
# Module-Level Functions
# ============================================================================


def create_trainer(
    model: LSTMModel,
    gpu_manager: GPUManager,
    epochs: int | None = None,
) -> ModelTrainer:
    """Create ModelTrainer instance.

    This is a convenience function for creating a ModelTrainer.

    Args:
        model: LSTMModel instance to train
        gpu_manager: GPUManager instance for hardware configuration
        epochs: Number of training epochs. Defaults to EPOCHS from config.

    Returns:
        Initialized ModelTrainer instance
    """
    return ModelTrainer(model=model, gpu_manager=gpu_manager, epochs=epochs)
