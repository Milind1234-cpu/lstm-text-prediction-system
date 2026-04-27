"""Integration test for ModelTrainer with small dataset."""

import json
import numpy as np
import pytest
from pathlib import Path

from src.model.trainer import ModelTrainer
from src.model.lstm_model import LSTMModel
from src.model.gpu_manager import GPUManager
from src.utils.config import (
    MODEL_ARCHITECTURE_PATH,
    MODEL_WEIGHTS_PATH,
    TOKENIZER_CONFIG_PATH,
    VOCABULARY_SIZE,
)


@pytest.mark.integration
class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer with actual training."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for quick training test."""
        # Create small dataset: 200 training samples, 50 validation samples
        np.random.seed(42)
        X_train = np.random.randint(0, 1000, size=(200, 50), dtype=np.int32)
        y_train = np.random.randint(0, VOCABULARY_SIZE, size=(200,), dtype=np.int32)
        X_val = np.random.randint(0, 1000, size=(50, 50), dtype=np.int32)
        y_val = np.random.randint(0, VOCABULARY_SIZE, size=(50,), dtype=np.int32)
        return X_train, y_train, X_val, y_val

    @pytest.fixture
    def tokenizer_config(self):
        """Create sample tokenizer config."""
        return {
            'vocabulary_size': VOCABULARY_SIZE,
            'word_to_index': {'<PAD>': 0, '<UNK>': 1, 'test': 2, 'word': 3},
            'index_to_word': {0: '<PAD>', 1: '<UNK>', 2: 'test', 3: 'word'},
        }

    def test_trainer_full_pipeline(self, small_dataset, tokenizer_config, tmp_path):
        """Test complete training pipeline with small dataset."""
        X_train, y_train, X_val, y_val = small_dataset

        # Create model and GPU manager
        model = LSTMModel()
        gpu_manager = GPUManager()

        # Create trainer with just 2 epochs for quick test
        trainer = ModelTrainer(
            model=model,
            gpu_manager=gpu_manager,
            epochs=2,
        )

        # Train the model
        results = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tokenizer_config=tokenizer_config,
        )

        # Verify results dictionary
        assert 'epochs' in results
        assert results['epochs'] == 2
        assert 'batch_size' in results
        assert 'train_samples' in results
        assert results['train_samples'] == 200
        assert 'val_samples' in results
        assert results['val_samples'] == 50
        assert 'final_train_loss' in results
        assert 'final_train_accuracy' in results
        assert 'final_val_loss' in results
        assert 'final_val_accuracy' in results
        assert 'model_architecture_path' in results
        assert 'model_weights_path' in results
        assert 'tokenizer_config_path' in results

        # Verify training history exists
        history = trainer.get_history()
        assert history is not None
        assert len(history.history['loss']) == 2
        assert len(history.history['val_loss']) == 2
        assert len(history.history['accuracy']) == 2
        assert len(history.history['val_accuracy']) == 2

        # Verify files were saved
        assert MODEL_ARCHITECTURE_PATH.exists()
        assert MODEL_WEIGHTS_PATH.exists()
        assert TOKENIZER_CONFIG_PATH.exists()

        # Verify tokenizer config was saved correctly
        with open(TOKENIZER_CONFIG_PATH, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        assert saved_config['vocabulary_size'] == VOCABULARY_SIZE

        # Verify checkpoint directory was created
        checkpoint_dir = MODEL_ARCHITECTURE_PATH.parent / "checkpoints"
        assert checkpoint_dir.exists()

        # Verify at least one checkpoint was saved
        checkpoints = list(checkpoint_dir.glob("*.weights.h5"))
        assert len(checkpoints) >= 1  # At least 1 checkpoint for 2 epochs

    def test_trainer_metrics_progression(self, small_dataset, tokenizer_config):
        """Test that training metrics are tracked across epochs."""
        X_train, y_train, X_val, y_val = small_dataset

        model = LSTMModel()
        gpu_manager = GPUManager()

        trainer = ModelTrainer(
            model=model,
            gpu_manager=gpu_manager,
            epochs=3,
        )

        results = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tokenizer_config=tokenizer_config,
        )

        history = trainer.get_history()

        # Verify we have metrics for all epochs
        assert len(history.history['loss']) == 3
        assert len(history.history['val_loss']) == 3
        assert len(history.history['accuracy']) == 3
        assert len(history.history['val_accuracy']) == 3

        # Verify metrics are reasonable (loss should be positive, accuracy between 0 and 1)
        for loss in history.history['loss']:
            assert loss > 0

        for acc in history.history['accuracy']:
            assert 0 <= acc <= 1

        for val_loss in history.history['val_loss']:
            assert val_loss > 0

        for val_acc in history.history['val_accuracy']:
            assert 0 <= val_acc <= 1
