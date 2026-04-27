"""Comprehensive tests for model components.

This module tests GPU manager configuration, LSTM model architecture,
model trainer with small dataset, and predictor with mock model.
"""

import json
import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

from src.model.gpu_manager import GPUManager
from src.model.lstm_model import LSTMModel, Perplexity
from src.model.trainer import ModelTrainer
from src.model.predictor import Predictor
from src.data.tokenizer import Tokenizer
from src.utils.config import (
    BIDIRECTIONAL_LSTM_UNITS,
    CPU_BATCH_SIZE,
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_STOP_WORDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    GPU_BATCH_SIZE,
    LEARNING_RATE,
    MAX_BATCH_SIZE,
    MAX_TOP_K,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    SEQUENCE_LENGTH,
    UNIDIRECTIONAL_LSTM_UNITS,
    VOCABULARY_SIZE,
)


# ============================================================================
# GPU Manager Tests
# ============================================================================


class TestGPUManager:
    """Test suite for GPUManager class."""

    @patch('tensorflow.config.list_physical_devices')
    @patch('tensorflow.config.experimental.get_device_details')
    @patch('tensorflow.config.experimental.set_memory_growth')
    def test_gpu_detection_with_gpu_available(
        self,
        mock_set_memory_growth,
        mock_get_device_details,
        mock_list_devices,
    ):
        """Test GPU manager detects and configures GPU when available."""
        # Mock GPU device
        mock_gpu = Mock()
        mock_list_devices.return_value = [mock_gpu]
        mock_get_device_details.return_value = {'device_name': 'NVIDIA GeForce RTX 3080'}

        gpu_manager = GPUManager()

        assert gpu_manager.gpu_available is True
        assert gpu_manager.device_name == 'NVIDIA GeForce RTX 3080'
        mock_set_memory_growth.assert_called_once_with(mock_gpu, True)

    @patch('tensorflow.config.list_physical_devices')
    @patch('tensorflow.config.set_visible_devices')
    def test_gpu_detection_without_gpu(
        self,
        mock_set_visible_devices,
        mock_list_devices,
    ):
        """Test GPU manager configures CPU-only mode when no GPU available."""
        mock_list_devices.return_value = []

        gpu_manager = GPUManager()

        assert gpu_manager.gpu_available is False
        assert gpu_manager.device_name == 'CPU'
        assert gpu_manager.memory_info == 'N/A'
        mock_set_visible_devices.assert_called_once_with([], 'GPU')

    @patch('tensorflow.config.list_physical_devices')
    @patch('tensorflow.config.experimental.get_device_details')
    @patch('tensorflow.config.experimental.set_memory_growth')
    def test_get_batch_size_with_gpu(
        self,
        mock_set_memory_growth,
        mock_get_device_details,
        mock_list_devices,
    ):
        """Test get_batch_size returns GPU batch size when GPU available."""
        mock_gpu = Mock()
        mock_gpu.name = '/physical_device:GPU:0'
        mock_list_devices.return_value = [mock_gpu]
        mock_get_device_details.return_value = {'device_name': 'NVIDIA GPU'}

        gpu_manager = GPUManager()
        batch_size = gpu_manager.get_batch_size(
            cpu_batch_size=CPU_BATCH_SIZE,
            gpu_batch_size=GPU_BATCH_SIZE,
        )

        assert batch_size == GPU_BATCH_SIZE

    @patch('tensorflow.config.list_physical_devices')
    def test_get_batch_size_with_cpu(self, mock_list_devices):
        """Test get_batch_size returns CPU batch size when no GPU."""
        mock_list_devices.return_value = []

        gpu_manager = GPUManager()
        batch_size = gpu_manager.get_batch_size(
            cpu_batch_size=CPU_BATCH_SIZE,
            gpu_batch_size=GPU_BATCH_SIZE,
        )

        assert batch_size == CPU_BATCH_SIZE

    @patch('tensorflow.config.list_physical_devices')
    @patch('tensorflow.config.experimental.get_device_details')
    @patch('tensorflow.config.experimental.set_memory_growth')
    def test_get_device_info(
        self,
        mock_set_memory_growth,
        mock_get_device_details,
        mock_list_devices,
    ):
        """Test get_device_info returns correct device information."""
        mock_gpu = Mock()
        mock_gpu.name = '/physical_device:GPU:0'
        mock_list_devices.return_value = [mock_gpu]
        mock_get_device_details.return_value = {'device_name': 'NVIDIA GPU'}

        gpu_manager = GPUManager()
        device_info = gpu_manager.get_device_info()

        assert device_info['gpu_available'] is True
        assert device_info['device_name'] == 'NVIDIA GPU'
        assert 'memory_info' in device_info


# ============================================================================
# LSTM Model Tests
# ============================================================================


class TestLSTMModel:
    """Test suite for LSTMModel class."""

    def test_model_initialization(self):
        """Test LSTM model initializes with correct configuration."""
        model = LSTMModel()

        assert model.vocabulary_size == VOCABULARY_SIZE
        assert model.sequence_length == SEQUENCE_LENGTH
        assert model.embedding_dim == EMBEDDING_DIM
        assert model.bidirectional_units == BIDIRECTIONAL_LSTM_UNITS
        assert model.unidirectional_units == UNIDIRECTIONAL_LSTM_UNITS
        assert model.dropout_rate == DROPOUT_RATE
        assert model.learning_rate == LEARNING_RATE
        assert model.model is not None

    def test_model_architecture_layers(self):
        """Test LSTM model has correct layer architecture."""
        model = LSTMModel()
        keras_model = model.get_model()

        # Check number of layers (embedding, bidirectional, dropout, lstm, dropout, dense)
        assert len(keras_model.layers) == 6

        # Check layer types
        assert isinstance(keras_model.layers[0], tf.keras.layers.Embedding)
        assert isinstance(keras_model.layers[1], tf.keras.layers.Bidirectional)
        assert isinstance(keras_model.layers[2], tf.keras.layers.Dropout)
        assert isinstance(keras_model.layers[3], tf.keras.layers.LSTM)
        assert isinstance(keras_model.layers[4], tf.keras.layers.Dropout)
        assert isinstance(keras_model.layers[5], tf.keras.layers.Dense)

    def test_model_embedding_layer_config(self):
        """Test embedding layer has correct configuration."""
        model = LSTMModel()
        keras_model = model.get_model()
        embedding_layer = keras_model.layers[0]

        assert embedding_layer.input_dim == VOCABULARY_SIZE
        assert embedding_layer.output_dim == EMBEDDING_DIM
        # input_length is deprecated in newer Keras versions, skip this check

    def test_model_output_layer_config(self):
        """Test output layer has correct configuration."""
        model = LSTMModel()
        keras_model = model.get_model()
        output_layer = keras_model.layers[-1]

        assert output_layer.units == VOCABULARY_SIZE
        assert output_layer.activation.__name__ == 'softmax'

    def test_model_compilation(self):
        """Test model is compiled with correct loss and optimizer."""
        model = LSTMModel()
        keras_model = model.get_model()

        assert keras_model.loss == 'categorical_crossentropy'
        # Check optimizer type - may be wrapped in LossScaleOptimizer for mixed precision
        optimizer = keras_model.optimizer
        if hasattr(optimizer, 'inner_optimizer'):
            # Mixed precision wraps the optimizer
            assert isinstance(optimizer.inner_optimizer, tf.keras.optimizers.Adam)
        else:
            assert isinstance(optimizer, tf.keras.optimizers.Adam)

    def test_model_input_output_shapes(self):
        """Test model accepts correct input shape and produces correct output shape."""
        model = LSTMModel()
        keras_model = model.get_model()

        # Create sample input
        sample_input = np.random.randint(0, VOCABULARY_SIZE, size=(1, SEQUENCE_LENGTH))
        
        # Get prediction
        output = keras_model.predict(sample_input, verbose=0)

        # Check output shape
        assert output.shape == (1, VOCABULARY_SIZE)
        
        # Check output is probability distribution (sums to ~1)
        # Use higher tolerance for float16 precision in mixed precision mode
        assert np.isclose(np.sum(output), 1.0, atol=0.01)

    def test_get_model_returns_keras_model(self):
        """Test get_model returns a Keras Sequential model."""
        model = LSTMModel()
        keras_model = model.get_model()

        assert isinstance(keras_model, tf.keras.Sequential)

    def test_save_architecture(self, tmp_path):
        """Test model architecture can be saved to JSON."""
        model = LSTMModel()
        filepath = tmp_path / "model_architecture.json"

        model.save_architecture(str(filepath))

        assert filepath.exists()
        
        # Verify JSON is valid
        with open(filepath, 'r') as f:
            architecture = json.load(f)
        
        assert 'config' in architecture
        assert 'class_name' in architecture

    def test_save_weights(self, tmp_path):
        """Test model weights can be saved to H5 file."""
        model = LSTMModel()
        filepath = tmp_path / "model_weights.weights.h5"

        # Build the model first by calling it with sample data
        sample_input = np.random.randint(0, VOCABULARY_SIZE, size=(1, SEQUENCE_LENGTH))
        model.get_model().predict(sample_input, verbose=0)

        model.save_weights(str(filepath))

        assert filepath.exists()

    def test_load_model(self, tmp_path):
        """Test model can be loaded from architecture and weights files."""
        # Create and save model
        model = LSTMModel()
        arch_path = tmp_path / "architecture.json"
        weights_path = tmp_path / "weights.weights.h5"
        
        # Build the model first by calling it with sample data
        sample_input = np.random.randint(0, VOCABULARY_SIZE, size=(1, SEQUENCE_LENGTH))
        model.get_model().predict(sample_input, verbose=0)
        
        model.save_architecture(str(arch_path))
        model.save_weights(str(weights_path))

        # Load model
        loaded_model = LSTMModel.load_model(str(arch_path), str(weights_path))

        assert loaded_model is not None
        assert isinstance(loaded_model, tf.keras.Sequential)


class TestPerplexity:
    """Test suite for Perplexity metric."""

    def test_perplexity_initialization(self):
        """Test Perplexity metric initializes correctly."""
        perplexity = Perplexity()

        assert perplexity.name == 'perplexity'
        assert perplexity.total_loss.numpy() == 0.0
        assert perplexity.count.numpy() == 0.0

    def test_perplexity_update_state(self):
        """Test Perplexity metric updates state correctly."""
        perplexity = Perplexity()

        # Create sample data
        y_true = tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=tf.float32)

        perplexity.update_state(y_true, y_pred)

        assert perplexity.count.numpy() == 2.0
        assert perplexity.total_loss.numpy() > 0.0

    def test_perplexity_result(self):
        """Test Perplexity metric calculates result correctly."""
        perplexity = Perplexity()

        # Create sample data
        y_true = tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=tf.float32)

        perplexity.update_state(y_true, y_pred)
        result = perplexity.result()

        # Perplexity should be exp(loss), which is > 1
        assert result.numpy() > 1.0

    def test_perplexity_reset_state(self):
        """Test Perplexity metric resets state correctly."""
        perplexity = Perplexity()

        # Update state
        y_true = tf.constant([[0, 1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)
        perplexity.update_state(y_true, y_pred)

        # Reset
        perplexity.reset_state()

        assert perplexity.total_loss.numpy() == 0.0
        assert perplexity.count.numpy() == 0.0


# ============================================================================
# Model Trainer Tests
# ============================================================================


class TestModelTrainerWithSmallDataset:
    """Test suite for ModelTrainer with small dataset."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for testing."""
        # Create small dataset (50 samples)
        X_train = np.random.randint(0, VOCABULARY_SIZE, size=(50, SEQUENCE_LENGTH), dtype=np.int32)
        y_train = np.random.randint(0, VOCABULARY_SIZE, size=(50,), dtype=np.int32)
        X_val = np.random.randint(0, VOCABULARY_SIZE, size=(10, SEQUENCE_LENGTH), dtype=np.int32)
        y_val = np.random.randint(0, VOCABULARY_SIZE, size=(10,), dtype=np.int32)
        return X_train, y_train, X_val, y_val

    @pytest.fixture
    def tokenizer_config(self):
        """Create sample tokenizer config."""
        return {
            'vocabulary_size': VOCABULARY_SIZE,
            'word_to_index': {'test': 1, 'word': 2, 'sample': 3},
            'index_to_word': {1: 'test', 2: 'word', 3: 'sample'},
        }

    @pytest.fixture
    def mock_model(self):
        """Create a mock LSTMModel."""
        model = MagicMock(spec=LSTMModel)
        keras_model = MagicMock()
        
        # Mock fit method to return history
        mock_history = MagicMock()
        mock_history.history = {
            'loss': [2.5, 2.3],
            'accuracy': [0.3, 0.35],
            'val_loss': [2.7, 2.5],
            'val_accuracy': [0.28, 0.32],
        }
        keras_model.fit.return_value = mock_history
        
        model.get_model.return_value = keras_model
        model.save_architecture = MagicMock()
        model.save_weights = MagicMock()
        return model

    @pytest.fixture
    def mock_gpu_manager(self):
        """Create a mock GPUManager."""
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_available = False
        gpu_manager.device_name = "CPU"
        gpu_manager.get_batch_size.return_value = CPU_BATCH_SIZE
        return gpu_manager

    def test_trainer_with_small_dataset(
        self,
        small_dataset,
        tokenizer_config,
        mock_model,
        mock_gpu_manager,
        tmp_path,
    ):
        """Test ModelTrainer trains successfully with small dataset."""
        X_train, y_train, X_val, y_val = small_dataset

        # Patch paths to use tmp_path
        with patch('src.model.trainer.MODEL_ARCHITECTURE_PATH', tmp_path / 'model.json'), \
             patch('src.model.trainer.MODEL_WEIGHTS_PATH', tmp_path / 'weights.h5'), \
             patch('src.model.trainer.TOKENIZER_CONFIG_PATH', tmp_path / 'tokenizer.json'), \
             patch('src.model.trainer.MODELS_DIR', tmp_path):

            trainer = ModelTrainer(
                model=mock_model,
                gpu_manager=mock_gpu_manager,
                epochs=2,
            )

            results = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tokenizer_config=tokenizer_config,
            )

            # Verify training was called
            mock_model.get_model().fit.assert_called_once()

            # Verify results
            assert results['epochs'] == 2
            assert results['batch_size'] == CPU_BATCH_SIZE
            assert results['train_samples'] == 50
            assert results['val_samples'] == 10
            assert 'final_train_loss' in results
            assert 'final_train_accuracy' in results
            assert 'final_val_loss' in results
            assert 'final_val_accuracy' in results

    def test_trainer_saves_model_and_tokenizer(
        self,
        small_dataset,
        tokenizer_config,
        mock_model,
        mock_gpu_manager,
        tmp_path,
    ):
        """Test ModelTrainer saves model and tokenizer after training."""
        X_train, y_train, X_val, y_val = small_dataset

        with patch('src.model.trainer.MODEL_ARCHITECTURE_PATH', tmp_path / 'model.json'), \
             patch('src.model.trainer.MODEL_WEIGHTS_PATH', tmp_path / 'weights.h5'), \
             patch('src.model.trainer.TOKENIZER_CONFIG_PATH', tmp_path / 'tokenizer.json'), \
             patch('src.model.trainer.MODELS_DIR', tmp_path):

            trainer = ModelTrainer(
                model=mock_model,
                gpu_manager=mock_gpu_manager,
                epochs=2,
            )

            trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tokenizer_config=tokenizer_config,
            )

            # Verify save methods were called
            mock_model.save_architecture.assert_called_once()
            mock_model.save_weights.assert_called_once()

            # Verify tokenizer config was saved
            tokenizer_path = tmp_path / 'tokenizer.json'
            assert tokenizer_path.exists()

    def test_trainer_creates_checkpoints(
        self,
        small_dataset,
        tokenizer_config,
        mock_model,
        mock_gpu_manager,
        tmp_path,
    ):
        """Test ModelTrainer creates checkpoint directory."""
        X_train, y_train, X_val, y_val = small_dataset

        with patch('src.model.trainer.MODELS_DIR', tmp_path), \
             patch('src.model.trainer.MODEL_ARCHITECTURE_PATH', tmp_path / 'model.json'), \
             patch('src.model.trainer.MODEL_WEIGHTS_PATH', tmp_path / 'weights.h5'), \
             patch('src.model.trainer.TOKENIZER_CONFIG_PATH', tmp_path / 'tokenizer.json'):

            trainer = ModelTrainer(
                model=mock_model,
                gpu_manager=mock_gpu_manager,
                epochs=2,
            )

            trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tokenizer_config=tokenizer_config,
            )

            # Verify checkpoint directory was created
            checkpoint_dir = tmp_path / 'checkpoints'
            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()


# ============================================================================
# Predictor Tests
# ============================================================================


class TestPredictor:
    """Test suite for Predictor class with mock model."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock Tokenizer."""
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.text_to_sequence.return_value = [1, 2, 3, 4, 5]
        tokenizer.get_index_word.return_value = "predicted"
        return tokenizer

    @pytest.fixture
    def mock_keras_model(self):
        """Create a mock Keras model."""
        model = MagicMock()
        # Mock predict to return probabilities for batch
        def predict_side_effect(input_array, verbose=0):
            batch_size = input_array.shape[0]
            predictions = np.zeros((batch_size, VOCABULARY_SIZE))
            for i in range(batch_size):
                predictions[i, 10] = 0.5  # Highest probability at index 10
                predictions[i, 20] = 0.3
                predictions[i, 30] = 0.2
            return predictions
        
        model.predict.side_effect = predict_side_effect
        return model

    @pytest.fixture
    def predictor_with_mock(self, mock_keras_model, mock_tokenizer):
        """Create a Predictor with mocked model and tokenizer."""
        predictor = Predictor()
        predictor.model = mock_keras_model
        predictor.tokenizer = mock_tokenizer
        predictor.is_loaded = True
        return predictor

    def test_predictor_initialization(self):
        """Test Predictor initializes correctly."""
        predictor = Predictor()

        assert predictor.model is None
        assert predictor.tokenizer is None
        assert predictor.sequence_length == SEQUENCE_LENGTH
        assert predictor.is_loaded is False

    def test_load_model_missing_files(self, tmp_path):
        """Test load_model raises error when files are missing."""
        predictor = Predictor()

        with pytest.raises(FileNotFoundError):
            predictor.load_model(
                architecture_path=tmp_path / "missing.json",
                weights_path=tmp_path / "missing.h5",
                tokenizer_path=tmp_path / "missing.json",
            )

    def test_predict_next_word_not_loaded(self):
        """Test predict_next_word raises error when model not loaded."""
        predictor = Predictor()

        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_next_word("test text")

    def test_predict_next_word_empty_text(self, predictor_with_mock):
        """Test predict_next_word raises error for empty text."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            predictor_with_mock.predict_next_word("")

    def test_predict_next_word_invalid_temperature(self, predictor_with_mock):
        """Test predict_next_word raises error for invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            predictor_with_mock.predict_next_word("test text", temperature=3.0)

        with pytest.raises(ValueError, match="Temperature must be between"):
            predictor_with_mock.predict_next_word("test text", temperature=0.0)

    def test_predict_next_word_success(self, predictor_with_mock):
        """Test predict_next_word returns predicted word."""
        result = predictor_with_mock.predict_next_word("test text")

        assert result == "predicted"
        predictor_with_mock.model.predict.assert_called_once()
        predictor_with_mock.tokenizer.text_to_sequence.assert_called_once_with("test text")

    def test_predict_next_word_with_temperature(self, predictor_with_mock):
        """Test predict_next_word applies temperature correctly."""
        result = predictor_with_mock.predict_next_word("test text", temperature=0.5)

        assert result == "predicted"
        # Temperature should affect the prediction process
        predictor_with_mock.model.predict.assert_called_once()

    def test_predict_top_k_success(self, predictor_with_mock, mock_tokenizer):
        """Test predict_top_k returns top k predictions."""
        # Mock get_index_word to return different words
        mock_tokenizer.get_index_word.side_effect = lambda idx: f"word_{idx}"
        
        results = predictor_with_mock.predict_top_k("test text", k=3)

        assert len(results) == 3
        assert all('word' in r and 'probability' in r for r in results)
        # Results should be sorted by probability (descending)
        assert results[0]['probability'] >= results[1]['probability']
        assert results[1]['probability'] >= results[2]['probability']

    def test_predict_top_k_invalid_k(self, predictor_with_mock):
        """Test predict_top_k raises error for invalid k."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            predictor_with_mock.predict_top_k("test text", k=0)

    def test_predict_top_k_limits_to_max(self, predictor_with_mock):
        """Test predict_top_k limits k to MAX_TOP_K."""
        results = predictor_with_mock.predict_top_k("test text", k=100)

        # Should be limited to MAX_TOP_K
        assert len(results) <= MAX_TOP_K

    def test_predict_batch_success(self, predictor_with_mock):
        """Test predict_batch processes multiple texts."""
        texts = ["text one", "text two", "text three"]
        
        results = predictor_with_mock.predict_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        # Model predict should be called once with batch
        predictor_with_mock.model.predict.assert_called_once()

    def test_predict_batch_empty_list(self, predictor_with_mock):
        """Test predict_batch raises error for empty list."""
        with pytest.raises(ValueError, match="Batch cannot be empty"):
            predictor_with_mock.predict_batch([])

    def test_predict_batch_exceeds_max_size(self, predictor_with_mock):
        """Test predict_batch raises error when batch size exceeds limit."""
        texts = ["text"] * (MAX_BATCH_SIZE + 1)

        with pytest.raises(ValueError, match="Batch size .* exceeds maximum"):
            predictor_with_mock.predict_batch(texts)

    def test_predict_batch_empty_text_in_batch(self, predictor_with_mock):
        """Test predict_batch raises error for empty text in batch."""
        texts = ["text one", "", "text three"]

        with pytest.raises(ValueError, match="Text at index 1 is empty"):
            predictor_with_mock.predict_batch(texts)

    def test_complete_text_success(self, predictor_with_mock, mock_tokenizer):
        """Test complete_text generates text completion."""
        # Mock to return stop word after 3 iterations
        mock_tokenizer.get_index_word.side_effect = ["word1", "word2", "."]
        
        result = predictor_with_mock.complete_text("test text", max_length=10)

        assert "test text" in result
        assert "word1" in result
        assert "word2" in result
        assert "." in result

    def test_complete_text_max_length(self, predictor_with_mock):
        """Test complete_text respects max_length."""
        result = predictor_with_mock.complete_text("test text", max_length=5)

        # Should generate at most 5 words
        generated_words = result.replace("test text", "").strip().split()
        assert len(generated_words) <= 5

    def test_complete_text_stop_words(self, predictor_with_mock, mock_tokenizer):
        """Test complete_text stops at stop words."""
        # Mock to return stop word
        mock_tokenizer.get_index_word.return_value = "."
        
        result = predictor_with_mock.complete_text(
            "test text",
            max_length=10,
            stop_words=["."],
        )

        # Should stop after encountering stop word
        assert result.endswith(".")

    def test_complete_text_invalid_max_length(self, predictor_with_mock):
        """Test complete_text raises error for invalid max_length."""
        with pytest.raises(ValueError, match="max_length must be at least 1"):
            predictor_with_mock.complete_text("test text", max_length=0)

    def test_prepare_input_pads_short_sequence(self, predictor_with_mock, mock_tokenizer):
        """Test _prepare_input pads sequences shorter than sequence_length."""
        mock_tokenizer.text_to_sequence.return_value = [1, 2, 3]  # Short sequence
        
        input_array = predictor_with_mock._prepare_input("short text")

        assert input_array.shape == (1, SEQUENCE_LENGTH)
        # Should be padded with zeros at the beginning
        assert input_array[0, 0] == 0

    def test_prepare_input_truncates_long_sequence(self, predictor_with_mock, mock_tokenizer):
        """Test _prepare_input truncates sequences longer than sequence_length."""
        long_sequence = list(range(SEQUENCE_LENGTH + 10))
        mock_tokenizer.text_to_sequence.return_value = long_sequence
        
        input_array = predictor_with_mock._prepare_input("long text")

        assert input_array.shape == (1, SEQUENCE_LENGTH)
        # Should keep the last SEQUENCE_LENGTH tokens
        assert input_array[0, -1] == long_sequence[-1]

    def test_apply_temperature_scaling(self, predictor_with_mock):
        """Test _apply_temperature applies temperature scaling correctly."""
        logits = np.array([1.0, 2.0, 3.0])
        
        # Temperature = 1.0 (no change)
        probs_1 = predictor_with_mock._apply_temperature(logits, 1.0)
        assert np.isclose(np.sum(probs_1), 1.0)
        
        # Temperature < 1.0 (more confident)
        probs_low = predictor_with_mock._apply_temperature(logits, 0.5)
        assert np.isclose(np.sum(probs_low), 1.0)
        # Highest logit should have higher probability with lower temperature
        assert probs_low[2] > probs_1[2]
        
        # Temperature > 1.0 (less confident)
        probs_high = predictor_with_mock._apply_temperature(logits, 2.0)
        assert np.isclose(np.sum(probs_high), 1.0)
        # Probabilities should be more uniform with higher temperature
        assert probs_high[2] < probs_1[2]
