"""Training pipeline script for LSTM text prediction system.

This script executes the complete training pipeline:
1. Wikipedia data collection
2. Text preprocessing
3. Tokenization and vocabulary building
4. Training sequence generation
5. GPU configuration
6. LSTM model training
7. Model and tokenizer saving

Usage:
    python scripts/run_training.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collector import DataCollector
from src.data.preprocessor import TextPreprocessor
from src.data.tokenizer import Tokenizer
from src.data.sequence_generator import SequenceGenerator
from src.model.gpu_manager import GPUManager
from src.model.lstm_model import LSTMModel
from src.model.trainer import ModelTrainer
from src.utils.config import PROCESSED_DATA_DIR
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
# Main Training Pipeline
# ============================================================================


def main() -> None:
    """Execute the complete training pipeline."""
    
    # Display welcome message
    print_panel(
        "LSTM Text Prediction System - Training Pipeline\n\n"
        "This script will execute the following steps:\n"
        "1. Collect Wikipedia articles on AI/ML topics\n"
        "2. Preprocess and clean text data\n"
        "3. Build vocabulary and tokenize text\n"
        "4. Generate training sequences\n"
        "5. Configure GPU acceleration\n"
        "6. Train LSTM model\n"
        "7. Save model and tokenizer",
        title="Training Pipeline",
        style="bold cyan",
        border_style="cyan",
    )
    
    try:
        # ====================================================================
        # Step 1: Data Collection
        # ====================================================================
        
        print_panel(
            "Step 1: Collecting Wikipedia articles",
            title="Data Collection",
            style="bold blue",
            border_style="blue",
        )
        
        collector = DataCollector()
        collection_stats = collector.collect_articles()
        
        logger.info(
            f"Data collection complete: {collection_stats['articles_collected']} articles, "
            f"{collection_stats['total_characters']} characters"
        )
        
        # ====================================================================
        # Step 2: Text Preprocessing
        # ====================================================================
        
        print_panel(
            "Step 2: Preprocessing and cleaning text",
            title="Text Preprocessing",
            style="bold blue",
            border_style="blue",
        )
        
        preprocessor = TextPreprocessor()
        preprocessing_stats = preprocessor.preprocess_all_files()
        
        logger.info(
            f"Text preprocessing complete: {preprocessing_stats['files_processed']} files, "
            f"{preprocessing_stats['lines_kept']} lines kept"
        )
        
        # ====================================================================
        # Step 3: Tokenization
        # ====================================================================
        
        print_panel(
            "Step 3: Building vocabulary and tokenizing text",
            title="Tokenization",
            style="bold blue",
            border_style="blue",
        )
        
        tokenizer = Tokenizer()
        vocab_stats = tokenizer.build_vocabulary()
        tokenizer.save_vocabulary()
        
        logger.info(
            f"Tokenization complete: {vocab_stats['vocabulary_size']} words, "
            f"{vocab_stats['coverage']:.2f}% coverage"
        )
        
        # ====================================================================
        # Step 4: Sequence Generation
        # ====================================================================
        
        print_panel(
            "Step 4: Generating training sequences",
            title="Sequence Generation",
            style="bold blue",
            border_style="blue",
        )
        
        sequence_generator = SequenceGenerator(tokenizer)
        sequence_stats = sequence_generator.generate_sequences(PROCESSED_DATA_DIR)
        
        # Get training and validation data
        X_train, y_train, X_val, y_val = sequence_generator.get_all_data()
        
        logger.info(
            f"Sequence generation complete: {sequence_stats['total_sequences']} sequences, "
            f"{sequence_stats['train_sequences']} train, {sequence_stats['val_sequences']} val"
        )
        
        # ====================================================================
        # Step 5: GPU Configuration
        # ====================================================================
        
        print_panel(
            "Step 5: Configuring GPU acceleration",
            title="GPU Configuration",
            style="bold blue",
            border_style="blue",
        )
        
        gpu_manager = GPUManager()
        device_info = gpu_manager.get_device_info()
        
        logger.info(
            f"GPU configuration complete: {device_info['device_name']} "
            f"({'GPU' if device_info['gpu_available'] else 'CPU'})"
        )
        
        # ====================================================================
        # Step 6: Model Training
        # ====================================================================
        
        print_panel(
            "Step 6: Training LSTM model",
            title="Model Training",
            style="bold blue",
            border_style="blue",
        )
        
        # Create LSTM model
        lstm_model = LSTMModel()
        lstm_model.summary()
        
        # Create trainer
        trainer = ModelTrainer(
            model=lstm_model,
            gpu_manager=gpu_manager,
        )
        
        # Prepare tokenizer config for saving
        tokenizer_config = {
            'vocabulary_size': tokenizer.vocabulary_size,
            'unknown_token': tokenizer.unknown_token,
            'padding_token': tokenizer.padding_token,
            'word_to_index': tokenizer.word_to_index,
            'index_to_word': {str(k): v for k, v in tokenizer.index_to_word.items()},
            'actual_vocabulary_size': len(tokenizer.word_to_index),
            'total_words_in_corpus': vocab_stats['total_words'],
            'unique_words_in_corpus': vocab_stats['unique_words'],
        }
        
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
        # Step 7: Training Summary
        # ====================================================================
        
        print_panel(
            "Step 7: Training pipeline complete!",
            title="Training Summary",
            style="bold green",
            border_style="green",
        )
        
        # Create comprehensive summary table
        summary_table = create_table(
            "Training Pipeline Summary",
            "Stage",
            "Key Metrics",
        )
        
        summary_table.add_row(
            "Data Collection",
            f"{collection_stats['articles_collected']} articles, "
            f"{collection_stats['total_characters']:,} characters"
        )
        
        summary_table.add_row(
            "Text Preprocessing",
            f"{preprocessing_stats['lines_kept']:,} lines kept, "
            f"{preprocessing_stats['total_characters']:,} characters"
        )
        
        summary_table.add_row(
            "Tokenization",
            f"{vocab_stats['vocabulary_size']:,} words, "
            f"{vocab_stats['coverage']:.2f}% coverage"
        )
        
        summary_table.add_row(
            "Sequence Generation",
            f"{sequence_stats['total_sequences']:,} sequences "
            f"({sequence_stats['train_sequences']:,} train, "
            f"{sequence_stats['val_sequences']:,} val)"
        )
        
        summary_table.add_row(
            "GPU Configuration",
            f"{device_info['device_name']} "
            f"({'GPU' if device_info['gpu_available'] else 'CPU'})"
        )
        
        summary_table.add_row(
            "Model Training",
            f"{training_results['epochs']} epochs, "
            f"batch size {training_results['batch_size']}, "
            f"val accuracy {training_results['final_val_accuracy']:.4f}"
        )
        
        summary_table.add_row(
            "Model Files",
            f"Architecture: {training_results['model_architecture_path']}\n"
            f"Weights: {training_results['model_weights_path']}\n"
            f"Tokenizer: {training_results['tokenizer_config_path']}"
        )
        
        console.print()
        console.print(summary_table)
        console.print()
        
        print_success(
            "Training pipeline completed successfully!\n\n"
            "Next steps:\n"
            "1. Start the API server: python scripts/run_api.py\n"
            "2. Access API documentation: http://localhost:8000/docs\n"
            "3. Test predictions using the API endpoints",
            title="Pipeline Complete",
        )
        
        logger.info("Training pipeline completed successfully")
    
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
