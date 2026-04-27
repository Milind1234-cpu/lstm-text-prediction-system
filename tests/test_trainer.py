"""Tests for ModelTrainer class."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.model.trainer import ModelTrainer, EpochProgressCallback
from src.model.lstm_model import LSTMModel
from src.model.gpu_manager import GPUManager
from src.utils.config import (
    CPU_BATCH_SIZE,
    GPU_BATCH_SIZE,
    EPOCHS,
    MODEL_ARCHITECTURE_PATH,
    MODEL_WEIGHTS_PATH,
    TOKENIZER_CONFIG_PATH,
    VOCABULARY_SIZE,
)


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LSTMModel."""
        model = MagicMock(spec=LSTMModel)
        keras_model = MagicMock()
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

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # Small dataset for testing
        X_train = np.random.randint(0, 1000, size=(100, 50), dtype=np.int32)
        y_train = np.random.randint(0, VOCABULARY_SIZE, size=(100,), dtype=np.int32)
        X_val = np.random.randint(0, 1000, size=(20, 50), dtype=np.int32)
        y_val = np.random.randint(0, VOCABULARY_SIZE, size=(20,), dtype=np.int32)
        return X_train, y_train, X_val, y_val

    @pytest.fixture
    def tokenizer_config(self):
        """Create sample tokenizer config."""
        return {
            'vocabulary_size': VOCABULARY_SIZE,
            'word_to_index': {'test': 1, 'word': 2},
            'index_to_word': {1: 'test', 2: 'word'},
        }

    def test_trainer_initialization(self, mock_model, mock_gpu_manager):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(
            model=mock_model,
            gpu_manager=mock_gpu_manager,
            epochs=10,
        )

        assert trainer.model == mock_model
        assert trainer.gpu_manager == mock_gpu_manager
        assert trainer.epochs == 10
        assert trainer.batch_size == CPU_BATCH_SIZE
        assert trainer.history is None

    def test_trainer_default_epochs(self, mock_model, mock_gpu_manager):
        """Test ModelTrainer uses default epochs from config."""
        trainer = ModelTrainer(
            model=mock_model,
            gpu_manager=mock_gpu_manager,
        )

        assert trainer.epochs == EPOCHS

    def test_trainer_gpu_batch_size(self, mock_model):
        """Test ModelTrainer uses GPU batch size when GPU is available."""
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_available = True
        gpu_manager.device_name = "NVIDIA GPU"
        gpu_manager.get_batch_size.return_value = GPU_BATCH_SIZE

        trainer = ModelTrainer(
            model=mock_model,
            gpu_manager=gpu_manager,
        )

        assert trainer.batch_size == GPU_BATCH_SIZE

    def test_get_history_before_training(self, mock_model, mock_gpu_manager):
        """Test get_history returns None before training."""
        trainer = ModelTrainer(
            model=mock_model,
            gpu_manager=mock_gpu_manager,
        )

        assert trainer.get_history() is None

    def test_epoch_progress_callback_initialization(self):
        """Test EpochProgressCallback initialization."""
        callback = EpochProgressCallback(
            epochs=50,
            train_samples=1000,
            val_samples=200,
        )

        assert callback.epochs == 50
        assert callback.train_samples == 1000
        assert callback.val_samples == 200
        assert callback.current_epoch == 0

    def test_epoch_progress_callback_on_epoch_end(self):
        """Test EpochProgressCallback on_epoch_end updates current epoch."""
        callback = EpochProgressCallback(
            epochs=50,
            train_samples=1000,
            val_samples=200,
        )

        logs = {
            'loss': 2.5,
            'accuracy': 0.35,
            'val_loss': 2.8,
            'val_accuracy': 0.32,
        }

        callback.on_epoch_end(epoch=0, logs=logs)
        assert callback.current_epoch == 1

        callback.on_epoch_end(epoch=1, logs=logs)
        assert callback.current_epoch == 2

    def test_epoch_progress_callback_handles_none_logs(self):
        """Test EpochProgressCallback handles None logs gracefully."""
        callback = EpochProgressCallback(
            epochs=50,
            train_samples=1000,
            val_samples=200,
        )

        # Should not raise an exception
        callback.on_epoch_end(epoch=0, logs=None)
        assert callback.current_epoch == 1


class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer (requires actual model)."""

    @pytest.fixture
    def real_model(self):
        """Create a real LSTMModel instance."""
        return LSTMModel()

    @pytest.fixture
    def real_gpu_manager(self):
        """Create a real GPUManager instance."""
        return GPUManager()

    def test_trainer_with_real_model(self, real_model, real_gpu_manager):
        """Test ModelTrainer with real model and GPU manager."""
        trainer = ModelTrainer(
            model=real_model,
            gpu_manager=real_gpu_manager,
            epochs=1,  # Just 1 epoch for testing
        )

        assert trainer.model == real_model
        assert trainer.gpu_manager == real_gpu_manager
        assert trainer.epochs == 1
        assert trainer.batch_size in [CPU_BATCH_SIZE, GPU_BATCH_SIZE]
