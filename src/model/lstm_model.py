"""LSTM model architecture for text prediction.

This module defines the bidirectional LSTM neural network architecture
for next-word prediction with embedding, LSTM layers, dropout, and dense output.
"""

import tensorflow as tf
from rich.table import Table
from tensorflow import keras
from tensorflow.keras import layers

from ..utils.config import (
    BIDIRECTIONAL_LSTM_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    OUTPUT_UNITS,
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
# Custom Metrics
# ============================================================================


class Perplexity(keras.metrics.Metric):
    """Custom metric for calculating perplexity.

    Perplexity is a measurement of how well a probability model predicts a sample.
    Lower perplexity indicates better prediction performance.

    Perplexity = exp(cross_entropy_loss)

    Attributes:
        total_loss: Accumulated cross-entropy loss
        count: Number of samples processed
    """

    def __init__(self, name: str = "perplexity", **kwargs) -> None:
        """Initialize perplexity metric.

        Args:
            name: Name of the metric (default: "perplexity")
            **kwargs: Additional keyword arguments for base Metric class
        """
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Update metric state with batch results.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            sample_weight: Optional sample weights
        """
        # Calculate categorical crossentropy loss
        loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply sample weights if provided
        if sample_weight is not None:
            loss = loss * sample_weight

        # Update accumulated loss and count
        self.total_loss.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.cast(tf.size(loss), tf.float32))

    def result(self) -> tf.Tensor:
        """Calculate perplexity from accumulated loss.

        Returns:
            Perplexity value (exp of average cross-entropy loss)
        """
        # Calculate average loss
        avg_loss = self.total_loss / self.count

        # Perplexity = exp(average_loss)
        return tf.exp(avg_loss)

    def reset_state(self) -> None:
        """Reset metric state for new epoch."""
        self.total_loss.assign(0.0)
        self.count.assign(0.0)


# ============================================================================
# LSTM Model Class
# ============================================================================


class LSTMModel:
    """Bidirectional LSTM model for text prediction.

    This class implements a neural network architecture with:
    - Embedding layer (256 dimensions, 10,000 vocabulary)
    - Bidirectional LSTM layer (512 units)
    - Dropout layer (0.3 rate)
    - Unidirectional LSTM layer (256 units)
    - Dropout layer (0.3 rate)
    - Dense output layer (10,000 units, softmax activation)

    The model is compiled with categorical crossentropy loss, Adam optimizer,
    and tracks accuracy and perplexity metrics.

    Attributes:
        model: Compiled Keras Sequential model
        vocabulary_size: Size of the vocabulary (10,000)
        sequence_length: Length of input sequences (50)
        embedding_dim: Dimension of embedding vectors (256)
        bidirectional_units: Units in bidirectional LSTM (512)
        unidirectional_units: Units in unidirectional LSTM (256)
        dropout_rate: Dropout rate after LSTM layers (0.3)
        learning_rate: Learning rate for Adam optimizer (0.001)
    """

    def __init__(self) -> None:
        """Initialize LSTM model with configured architecture."""
        self.vocabulary_size = VOCABULARY_SIZE
        self.sequence_length = SEQUENCE_LENGTH
        self.embedding_dim = EMBEDDING_DIM
        self.bidirectional_units = BIDIRECTIONAL_LSTM_UNITS
        self.unidirectional_units = UNIDIRECTIONAL_LSTM_UNITS
        self.dropout_rate = DROPOUT_RATE
        self.learning_rate = LEARNING_RATE

        logger.info("Initializing LSTM model architecture")

        # Build the model
        self.model = self._build_model()

        # Compile the model
        self._compile_model()

        logger.info("LSTM model initialized successfully")

    def _build_model(self) -> keras.Sequential:
        """Build the LSTM model architecture.

        Architecture:
        1. Embedding layer: Maps vocabulary indices to dense vectors
        2. Bidirectional LSTM: Processes sequences in both directions
        3. Dropout: Regularization after bidirectional LSTM
        4. Unidirectional LSTM: Additional recurrent processing
        5. Dropout: Regularization after unidirectional LSTM
        6. Dense output: Softmax activation for word probabilities

        Returns:
            Keras Sequential model with LSTM architecture
        """
        model = keras.Sequential(
            [
                # Embedding layer: vocabulary_size -> embedding_dim
                layers.Embedding(
                    input_dim=self.vocabulary_size,
                    output_dim=self.embedding_dim,
                    input_length=self.sequence_length,
                    name="embedding",
                ),
                # Bidirectional LSTM layer: 512 units (256 forward + 256 backward)
                layers.Bidirectional(
                    layers.LSTM(
                        self.bidirectional_units // 2,
                        return_sequences=True,
                        name="lstm_bidirectional_forward",
                    ),
                    name="bidirectional_lstm",
                ),
                # Dropout after bidirectional LSTM
                layers.Dropout(self.dropout_rate, name="dropout_1"),
                # Unidirectional LSTM layer: 256 units
                layers.LSTM(
                    self.unidirectional_units,
                    return_sequences=False,
                    name="unidirectional_lstm",
                ),
                # Dropout after unidirectional LSTM
                layers.Dropout(self.dropout_rate, name="dropout_2"),
                # Dense output layer with softmax activation
                layers.Dense(
                    self.vocabulary_size,
                    activation="softmax",
                    name="output",
                ),
            ],
            name="lstm_text_prediction",
        )

        logger.info("Model architecture built successfully")
        return model

    def _compile_model(self) -> None:
        """Compile the model with loss, optimizer, and metrics.

        Configuration:
        - Loss: Categorical crossentropy (for multi-class classification)
        - Optimizer: Adam with learning rate 0.001
        - Metrics: Accuracy and perplexity
        """
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy", Perplexity()],
        )

        logger.info(
            f"Model compiled with Adam optimizer (lr={self.learning_rate}), "
            "categorical crossentropy loss, accuracy and perplexity metrics"
        )

    def summary(self) -> None:
        """Display model summary using Rich formatting.

        Shows:
        - Model architecture with layer types and output shapes
        - Number of parameters per layer
        - Total trainable and non-trainable parameters
        """
        # Print model architecture panel
        print_panel(
            "LSTM Text Prediction Model Architecture\n"
            f"Vocabulary Size: {self.vocabulary_size:,}\n"
            f"Sequence Length: {self.sequence_length}\n"
            f"Embedding Dimension: {self.embedding_dim}\n"
            f"Bidirectional LSTM Units: {self.bidirectional_units}\n"
            f"Unidirectional LSTM Units: {self.unidirectional_units}\n"
            f"Dropout Rate: {self.dropout_rate}\n"
            f"Learning Rate: {self.learning_rate}",
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

        # Add layer information to table
        for layer in self.model.layers:
            layer_name = f"{layer.name} ({layer.__class__.__name__})"
            # Get output shape - handle both single and multiple outputs
            try:
                if hasattr(layer, 'output_shape'):
                    output_shape = str(layer.output_shape)
                else:
                    # Build the model first to get output shapes
                    if not self.model.built:
                        self.model.build(input_shape=(None, self.sequence_length))
                    output_shape = str(layer.output.shape)
            except Exception:
                output_shape = "Multiple"
            params = f"{layer.count_params():,}"
            table.add_row(layer_name, output_shape, params)

        # Print the table
        console.print()
        console.print(table)
        console.print()

        # Calculate total parameters
        trainable_params = sum(
            [layer.count_params() for layer in self.model.layers if layer.trainable]
        )
        non_trainable_params = sum(
            [layer.count_params() for layer in self.model.layers if not layer.trainable]
        )
        total_params = trainable_params + non_trainable_params

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
        params_table.add_row("Non-trainable params", f"{non_trainable_params:,}")

        console.print(params_table)
        console.print()

        print_success(
            f"Model has {total_params:,} total parameters "
            f"({trainable_params:,} trainable)",
            title="Model Summary Complete",
        )

    def get_model(self) -> keras.Sequential:
        """Get the compiled Keras model.

        Returns:
            Compiled Keras Sequential model
        """
        return self.model

    def save_architecture(self, filepath: str) -> None:
        """Save model architecture to JSON file.

        Args:
            filepath: Path to save the model architecture JSON
        """
        architecture_json = self.model.to_json()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(architecture_json)
        logger.info(f"Model architecture saved to {filepath}")

    def save_weights(self, filepath: str) -> None:
        """Save model weights to H5 file.

        Args:
            filepath: Path to save the model weights
        """
        self.model.save_weights(filepath)
        logger.info(f"Model weights saved to {filepath}")

    @staticmethod
    def load_model(architecture_path: str, weights_path: str) -> keras.Sequential:
        """Load model from architecture JSON and weights H5 files.

        Args:
            architecture_path: Path to model architecture JSON file
            weights_path: Path to model weights H5 file

        Returns:
            Loaded Keras model with weights
        """
        # Load architecture from JSON
        with open(architecture_path, "r", encoding="utf-8") as f:
            architecture_json = f.read()

        # Create model from JSON with custom objects
        model = keras.models.model_from_json(
            architecture_json, custom_objects={"Perplexity": Perplexity}
        )

        # Load weights
        model.load_weights(weights_path)

        logger.info(f"Model loaded from {architecture_path} and {weights_path}")
        return model


# ============================================================================
# Module-Level Functions
# ============================================================================


def create_lstm_model() -> LSTMModel:
    """Create and initialize LSTM model.

    This is a convenience function for creating an LSTM model instance.

    Returns:
        Initialized LSTMModel instance
    """
    return LSTMModel()
