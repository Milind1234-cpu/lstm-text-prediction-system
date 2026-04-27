"""PyTorch GPU training pipeline script for LSTM text prediction system.

This script executes the complete training pipeline with GPU acceleration:
1. Load preprocessed data and tokenizer
2. Load training sequences
3. Configure GPU
4. Train LSTM model on GPU
5. Save model and results

Usage:
    python scripts/run_training_pytorch.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json

from src.data.sequence_generator import SequenceGenerator
from src.data.tokenizer import Tokenizer
from src.model.lstm_model_pytorch import LSTMModelPyTorch
from src.model.trainer_pytorch import ModelTrainerPyTorch
from src.utils.config import PROCESSED_DATA_DIR, TOKENIZER_CONFIG_PATH
from src.utils.logger import (
    console,
    print_panel,
    print_success,
    print_error,
    create_table,
    setup_logger,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Main PyTorch Training Pipeline
# ============================================================================


def main() -> None:
    """Execute the PyTorch GPU training pipeline."""
    
    # Display welcome message
    print_panel(
        "LSTM Text Prediction System - PyTorch GPU Training\n\n"
        "This script will execute the following steps:\n"
        "1. Check GPU availability\n"
        "2. Load tokenizer and vocabulary\n"
        "3. Load or generate training sequences\n"
        "4. Initialize PyTorch LSTM model\n"
        "5. Train model on GPU\n"
        "6. Save trained model",
        title="PyTorch GPU Training Pipeline",
        style="bold cyan",
        border_style="cyan",
    )
    
    try:
        # ====================================================================
        # Step 1: GPU Configuration
        # ====================================================================
        
        print_panel(
            "Step 1: Checking GPU availability",
            title="GPU Configuration",
            style="bold blue",
            border_style="blue",
        )
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
            
            print_success(
                f"GPU Available: {gpu_name}\n"
                f"Memory: {gpu_memory:.2f} GB\n"
                f"CUDA Version: {torch.version.cuda}",
                title="GPU Ready",
            )
        else:
            device = "cpu"
            logger.warning("No GPU detected, using CPU")
            print_panel(
                "No GPU detected, training will use CPU\n"
                "This will be slower than GPU training",
                title="CPU Mode",
                style="yellow",
                border_style="yellow",
            )
        
        # ====================================================================
        # Step 2: Load Tokenizer
        # ====================================================================
        
        print_panel(
            "Step 2: Loading tokenizer and vocabulary",
            title="Tokenizer Loading",
            style="bold blue",
            border_style="blue",
        )
        
        # Check if tokenizer exists
        if not TOKENIZER_CONFIG_PATH.exists():
            print_error(
                "Tokenizer not found! Please run the full training pipeline first:\n"
                "python scripts/run_training.py",
                title="Tokenizer Missing",
            )
            sys.exit(1)
        
        # Load tokenizer
        tokenizer = Tokenizer()
        tokenizer.load_vocabulary()
        
        logger.info(f"Tokenizer loaded: {tokenizer.vocabulary_size} words")
        
        print_success(
            f"Vocabulary Size: {tokenizer.vocabulary_size:,}\n"
            f"Coverage: {tokenizer.vocabulary_size / 10000 * 100:.2f}%",
            title="Tokenizer Loaded",
        )
        
        # ====================================================================
        # Step 3: Load Training Sequences
        # ====================================================================
        
        print_panel(
            "Step 3: Loading training sequences",
            title="Sequence Loading",
            style="bold blue",
            border_style="blue",
        )
        
        # Generate or load sequences
        sequence_generator = SequenceGenerator(tokenizer)
        sequence_stats = sequence_generator.generate_sequences(PROCESSED_DATA_DIR)
        
        # Get training and validation data
        X_train, y_train, X_val, y_val = sequence_generator.get_all_data()
        
        logger.info(
            f"Sequences loaded: {sequence_stats['total_sequences']} total, "
            f"{sequence_stats['train_sequences']} train, {sequence_stats['val_sequences']} val"
        )
        
        print_success(
            f"Total Sequences: {sequence_stats['total_sequences']:,}\n"
            f"Training: {sequence_stats['train_sequences']:,}\n"
            f"Validation: {sequence_stats['val_sequences']:,}",
            title="Sequences Ready",
        )
        
        # ====================================================================
        # Step 4: Initialize PyTorch Model
        # ====================================================================
        
        print_panel(
            "Step 4: Initializing PyTorch LSTM model",
            title="Model Initialization",
            style="bold blue",
            border_style="blue",
        )
        
        # Create PyTorch model
        model = LSTMModelPyTorch(device=device)
        model.summary()
        
        # ====================================================================
        # Step 5: Train Model
        # ====================================================================
        
        print_panel(
            "Step 5: Training LSTM model on GPU",
            title="Model Training",
            style="bold blue",
            border_style="blue",
        )
        
        # Create trainer
        trainer = ModelTrainerPyTorch(
            model=model,
            device=device,
        )
        
        # Prepare tokenizer config
        with open(TOKENIZER_CONFIG_PATH, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        
        # Train model
        training_results = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tokenizer_config=tokenizer_config,
        )
        
        logger.info(
            f"Model training complete: {training_results['epochs']} epochs, "
            f"final val accuracy: {training_results['final_val_accuracy']:.4f}"
        )
        
        # ====================================================================
        # Step 6: Training Summary
        # ====================================================================
        
        print_panel(
            "Step 6: PyTorch GPU training complete!",
            title="Training Summary",
            style="bold green",
            border_style="green",
        )
        
        # Create comprehensive summary table
        summary_table = create_table(
            "PyTorch Training Summary",
            "Stage",
            "Key Metrics",
        )
        
        summary_table.add_row(
            "GPU Configuration",
            f"{gpu_name if device == 'cuda' else 'CPU'} "
            f"({gpu_memory:.2f} GB)" if device == "cuda" else "CPU"
        )
        
        summary_table.add_row(
            "Tokenizer",
            f"{tokenizer.vocabulary_size:,} words"
        )
        
        summary_table.add_row(
            "Training Sequences",
            f"{sequence_stats['total_sequences']:,} sequences "
            f"({sequence_stats['train_sequences']:,} train, "
            f"{sequence_stats['val_sequences']:,} val)"
        )
        
        summary_table.add_row(
            "Model Training",
            f"{training_results['epochs']} epochs, "
            f"batch size {training_results['batch_size']}, "
            f"val accuracy {training_results['final_val_accuracy']:.4f}"
        )
        
        summary_table.add_row(
            "Model Files",
            f"Weights: {training_results['model_weights_path']}\n"
            f"Tokenizer: {training_results['tokenizer_config_path']}"
        )
        
        console.print()
        console.print(summary_table)
        console.print()
        
        print_success(
            "PyTorch GPU training completed successfully!\n\n"
            "Next steps:\n"
            "1. The model is now trained and saved\n"
            "2. Restart the API server to use the new model\n"
            "3. Test predictions using the API endpoints",
            title="Training Complete",
        )
        
        logger.info("PyTorch GPU training pipeline completed successfully")
    
    except KeyboardInterrupt:
        logger.warning("Training pipeline interrupted by user")
        print_panel(
            "Training pipeline interrupted by user (Ctrl+C)",
            title="Interrupted",
            style="yellow",
            border_style="yellow",
        )
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        print_error(
            f"Training pipeline failed with error:\n{e}\n\n"
            f"Check the logs for more details.",
            title="Pipeline Failed",
        )
        sys.exit(1)


# ============================================================================
# Script Entry Point
# ============================================================================


if __name__ == "__main__":
    main()
