"""Sequence generation module for LSTM training.

This module provides functionality to generate fixed-length training sequences
from tokenized text using a sliding window approach. It creates input sequences
of 50 tokens and target outputs of the next word, splits data into train/validation
sets, and displays progress using Rich progress bars.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Final

from .tokenizer import Tokenizer
from ..utils.config import (
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    TRAIN_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
)
from ..utils.logger import (
    setup_logger,
    console,
    print_panel,
    print_success,
    print_error,
    create_progress_bar,
    create_table,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)

# ============================================================================
# SequenceGenerator Class
# ============================================================================


class SequenceGenerator:
    """Generates fixed-length training sequences from tokenized text.
    
    This class creates training sequences using a sliding window approach with
    configurable sequence length and stride. Each sequence consists of an input
    sequence of N tokens and a target output of the next word. The sequences are
    split into training and validation sets for model training.
    
    Attributes:
        tokenizer: Tokenizer instance for text-to-sequence conversion
        sequence_length: Length of input sequences (default: 50)
        stride: Stride for sliding window (default: 1)
        train_split_ratio: Ratio of data for training (default: 0.8)
        validation_split_ratio: Ratio of data for validation (default: 0.2)
        X_train: Training input sequences
        y_train: Training target outputs
        X_val: Validation input sequences
        y_val: Validation target outputs
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        sequence_length: int | None = None,
        stride: int | None = None,
        train_split_ratio: float | None = None,
    ) -> None:
        """Initialize the SequenceGenerator.
        
        Args:
            tokenizer: Tokenizer instance with fitted vocabulary
            sequence_length: Length of input sequences. Defaults to SEQUENCE_LENGTH from config.
            stride: Stride for sliding window. Defaults to SEQUENCE_STRIDE from config.
            train_split_ratio: Ratio for train split. Defaults to TRAIN_SPLIT_RATIO from config.
        
        Raises:
            ValueError: If tokenizer is not fitted or if split ratios are invalid
        """
        if not tokenizer.is_fitted:
            error_msg = "Tokenizer must be fitted before creating SequenceGenerator"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.tokenizer: Final[Tokenizer] = tokenizer
        self.sequence_length: Final[int] = (
            sequence_length if sequence_length is not None else SEQUENCE_LENGTH
        )
        self.stride: Final[int] = (
            stride if stride is not None else SEQUENCE_STRIDE
        )
        self.train_split_ratio: Final[float] = (
            train_split_ratio if train_split_ratio is not None else TRAIN_SPLIT_RATIO
        )
        self.validation_split_ratio: Final[float] = (
            1.0 - self.train_split_ratio
        )
        
        # Validate split ratios
        if not (0.0 < self.train_split_ratio < 1.0):
            error_msg = f"Invalid train_split_ratio: {self.train_split_ratio}. Must be between 0 and 1."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize sequence storage
        self.X_train: np.ndarray[tuple[int, int], np.dtype[np.int32]] | None = None
        self.y_train: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None
        self.X_val: np.ndarray[tuple[int, int], np.dtype[np.int32]] | None = None
        self.y_val: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None
        
        logger.info("SequenceGenerator initialized")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Stride: {self.stride}")
        logger.info(f"Train split ratio: {self.train_split_ratio:.2%}")
        logger.info(f"Validation split ratio: {self.validation_split_ratio:.2%}")
    
    def generate_sequences(
        self,
        corpus_dir: Path,
    ) -> dict[str, int]:
        """Generate training sequences from text corpus.
        
        Reads all text files from the corpus directory, tokenizes them, and creates
        fixed-length sequences using a sliding window approach. Each sequence consists
        of an input sequence of length `sequence_length` and a target output of the
        next word. The sequences are then split into training and validation sets.
        
        Args:
            corpus_dir: Directory containing preprocessed text files
        
        Returns:
            Dictionary containing sequence statistics:
                - 'total_sequences': Total number of sequences generated
                - 'train_sequences': Number of training sequences
                - 'val_sequences': Number of validation sequences
                - 'sequence_length': Length of input sequences
                - 'vocabulary_size': Size of vocabulary
        
        Raises:
            FileNotFoundError: If corpus directory does not exist
            ValueError: If no text files found or insufficient data for sequences
        """
        if not corpus_dir.exists():
            error_msg = f"Corpus directory does not exist: {corpus_dir}"
            logger.error(error_msg)
            print_error(error_msg, title="Directory Not Found")
            raise FileNotFoundError(error_msg)
        
        # Get all text files
        text_files = list(corpus_dir.glob("*.txt"))
        
        if not text_files:
            error_msg = f"No text files found in {corpus_dir}"
            logger.error(error_msg)
            print_error(error_msg, title="No Files Found")
            raise ValueError(error_msg)
        
        print_panel(
            f"Generating training sequences\n"
            f"Corpus directory: {corpus_dir}\n"
            f"Files: {len(text_files)}\n"
            f"Sequence length: {self.sequence_length}\n"
            f"Stride: {self.stride}",
            title="Sequence Generation",
            style="bold blue",
            border_style="blue",
        )
        
        # Tokenize all text files and concatenate into single sequence
        logger.info("Tokenizing corpus...")
        all_tokens: list[int] = []
        
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Tokenizing files...",
                total=len(text_files),
            )
            
            for text_file in text_files:
                try:
                    text = text_file.read_text(encoding='utf-8')
                    tokens = self.tokenizer.text_to_sequence(text)
                    all_tokens.extend(tokens)
                    logger.debug(f"Tokenized {text_file.name}: {len(tokens)} tokens")
                
                except Exception as e:
                    logger.error(f"Failed to tokenize file '{text_file.name}': {e}")
                
                finally:
                    progress.update(task, advance=1)
        
        total_tokens = len(all_tokens)
        logger.info(f"Total tokens: {total_tokens:,}")
        
        # Validate sufficient data for sequences
        min_tokens_required = self.sequence_length + 1  # +1 for target
        if total_tokens < min_tokens_required:
            error_msg = (
                f"Insufficient tokens for sequence generation. "
                f"Required: {min_tokens_required}, Available: {total_tokens}"
            )
            logger.error(error_msg)
            print_error(error_msg, title="Insufficient Data")
            raise ValueError(error_msg)
        
        # Generate sequences using sliding window
        logger.info("Generating sequences with sliding window...")
        sequences: list[list[int]] = []
        targets: list[int] = []
        
        # Calculate number of sequences
        num_sequences = (total_tokens - self.sequence_length) // self.stride
        
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Creating sequences...",
                total=num_sequences,
            )
            
            for i in range(0, total_tokens - self.sequence_length, self.stride):
                # Input sequence: tokens[i:i+sequence_length]
                # Target: tokens[i+sequence_length]
                input_seq = all_tokens[i:i + self.sequence_length]
                target = all_tokens[i + self.sequence_length]
                
                sequences.append(input_seq)
                targets.append(target)
                
                progress.update(task, advance=1)
        
        total_sequences = len(sequences)
        logger.info(f"Generated {total_sequences:,} sequences")
        
        # Convert to numpy arrays
        X = np.array(sequences, dtype=np.int32)
        y = np.array(targets, dtype=np.int32)
        
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Split into train and validation sets
        logger.info("Splitting into train and validation sets...")
        split_index = int(total_sequences * self.train_split_ratio)
        
        self.X_train = X[:split_index]
        self.y_train = y[:split_index]
        self.X_val = X[split_index:]
        self.y_val = y[split_index:]
        
        train_sequences = len(self.X_train)
        val_sequences = len(self.X_val)
        
        logger.info(f"Training sequences: {train_sequences:,}")
        logger.info(f"Validation sequences: {val_sequences:,}")
        
        # Display sequence statistics
        self._display_sequence_stats(
            total_sequences=total_sequences,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            total_tokens=total_tokens,
        )
        
        return {
            'total_sequences': total_sequences,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'sequence_length': self.sequence_length,
            'vocabulary_size': self.tokenizer.get_vocabulary_size(),
        }
    
    def get_train_data(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ]:
        """Get training data.
        
        Returns:
            Tuple of (X_train, y_train) numpy arrays
        
        Raises:
            ValueError: If sequences have not been generated yet
        """
        if self.X_train is None or self.y_train is None:
            error_msg = "Sequences not generated. Call generate_sequences() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.X_train, self.y_train
    
    def get_validation_data(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ]:
        """Get validation data.
        
        Returns:
            Tuple of (X_val, y_val) numpy arrays
        
        Raises:
            ValueError: If sequences have not been generated yet
        """
        if self.X_val is None or self.y_val is None:
            error_msg = "Sequences not generated. Call generate_sequences() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.X_val, self.y_val
    
    def get_all_data(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int, int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ]:
        """Get all training and validation data.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val) numpy arrays
        
        Raises:
            ValueError: If sequences have not been generated yet
        """
        if (self.X_train is None or self.y_train is None or
            self.X_val is None or self.y_val is None):
            error_msg = "Sequences not generated. Call generate_sequences() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.X_train, self.y_train, self.X_val, self.y_val
    
    def _display_sequence_stats(
        self,
        total_sequences: int,
        train_sequences: int,
        val_sequences: int,
        total_tokens: int,
    ) -> None:
        """Display sequence generation statistics with Rich table.
        
        Args:
            total_sequences: Total number of sequences generated
            train_sequences: Number of training sequences
            val_sequences: Number of validation sequences
            total_tokens: Total number of tokens in corpus
        """
        # Create statistics table
        table = create_table(
            "Sequence Generation Statistics",
            "Metric",
            "Value",
        )
        
        table.add_row("Total Tokens", f"{total_tokens:,}")
        table.add_row("Sequence Length", f"{self.sequence_length}")
        table.add_row("Stride", f"{self.stride}")
        table.add_row("Total Sequences", f"{total_sequences:,}")
        table.add_row("Training Sequences", f"{train_sequences:,} ({self.train_split_ratio:.1%})")
        table.add_row("Validation Sequences", f"{val_sequences:,} ({self.validation_split_ratio:.1%})")
        table.add_row("Vocabulary Size", f"{self.tokenizer.get_vocabulary_size():,}")
        
        console.print(table)
        
        # Display data shapes
        shapes_table = create_table(
            "Data Shapes",
            "Dataset",
            "Input Shape (X)",
            "Target Shape (y)",
        )
        
        if self.X_train is not None and self.y_train is not None:
            shapes_table.add_row(
                "Training",
                f"{self.X_train.shape}",
                f"{self.y_train.shape}",
            )
        
        if self.X_val is not None and self.y_val is not None:
            shapes_table.add_row(
                "Validation",
                f"{self.X_val.shape}",
                f"{self.y_val.shape}",
            )
        
        console.print(shapes_table)
        
        print_success(
            f"Sequences generated successfully!\n"
            f"Total: {total_sequences:,} sequences\n"
            f"Train: {train_sequences:,} | Val: {val_sequences:,}",
            title="Sequence Generation Complete",
        )


# ============================================================================
# Main Execution (for testing)
# ============================================================================


def main() -> None:
    """Main function to test sequence generator functionality."""
    from ..utils.config import PROCESSED_DATA_DIR
    
    # Create and load tokenizer
    print_panel(
        "Loading tokenizer...",
        title="Step 1: Tokenizer Setup",
        style="bold cyan",
        border_style="cyan",
    )
    
    tokenizer = Tokenizer()
    
    try:
        tokenizer.load_vocabulary()
    except FileNotFoundError:
        logger.warning("Tokenizer config not found. Building vocabulary...")
        tokenizer.build_vocabulary()
        tokenizer.save_vocabulary()
    
    # Create sequence generator
    print_panel(
        f"Creating sequence generator...\n"
        f"Sequence length: {SEQUENCE_LENGTH}\n"
        f"Stride: {SEQUENCE_STRIDE}\n"
        f"Train split: {TRAIN_SPLIT_RATIO:.1%}",
        title="Step 2: Sequence Generator Setup",
        style="bold cyan",
        border_style="cyan",
    )
    
    generator = SequenceGenerator(tokenizer)
    
    # Generate sequences
    print_panel(
        f"Generating sequences from corpus...\n"
        f"Corpus directory: {PROCESSED_DATA_DIR}",
        title="Step 3: Sequence Generation",
        style="bold cyan",
        border_style="cyan",
    )
    
    stats = generator.generate_sequences(PROCESSED_DATA_DIR)
    
    # Get data
    X_train, y_train = generator.get_train_data()
    X_val, y_val = generator.get_validation_data()
    
    # Display sample sequences
    print_panel(
        "Displaying sample sequences...",
        title="Step 4: Sample Sequences",
        style="bold cyan",
        border_style="cyan",
    )
    
    sample_table = create_table(
        "Sample Training Sequences (First 3)",
        "Index",
        "Input Sequence (first 10 tokens)",
        "Target",
    )
    
    for i in range(min(3, len(X_train))):
        input_seq = X_train[i][:10]  # Show first 10 tokens
        target = y_train[i]
        
        # Convert to words for display
        input_words = [tokenizer.get_index_word(idx) for idx in input_seq]
        target_word = tokenizer.get_index_word(target)
        
        sample_table.add_row(
            str(i),
            " ".join(input_words) + "...",
            target_word,
        )
    
    console.print(sample_table)
    
    # Display final statistics
    print_panel(
        f"Total Sequences: {stats['total_sequences']:,}\n"
        f"Training: {stats['train_sequences']:,}\n"
        f"Validation: {stats['val_sequences']:,}\n"
        f"Sequence Length: {stats['sequence_length']}\n"
        f"Vocabulary Size: {stats['vocabulary_size']:,}",
        title="Final Statistics",
        style="bold green",
        border_style="green",
    )


if __name__ == "__main__":
    main()
