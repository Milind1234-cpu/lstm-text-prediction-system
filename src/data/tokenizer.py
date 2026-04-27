"""Tokenization module with fixed vocabulary.

This module provides functionality to build a vocabulary from a text corpus,
tokenize text to integer sequences, handle out-of-vocabulary words, and save/load
vocabulary configurations. It uses the 10,000 most frequent words and displays
vocabulary statistics using Rich tables.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Final

from ..utils.config import (
    PROCESSED_DATA_DIR,
    TOKENIZER_CONFIG_PATH,
    VOCABULARY_SIZE,
    UNKNOWN_TOKEN,
    PADDING_TOKEN,
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
# Tokenizer Class
# ============================================================================


class Tokenizer:
    """Tokenizes text with a fixed vocabulary of most frequent words.
    
    This class builds a vocabulary from a text corpus, assigns unique integer
    indices to each word, handles out-of-vocabulary words with a special unknown
    token, and provides bidirectional conversion between text and integer sequences.
    The vocabulary can be saved to and loaded from JSON files for inference use.
    
    Attributes:
        vocabulary_size: Maximum number of words in the vocabulary
        unknown_token: Special token for out-of-vocabulary words
        padding_token: Special token for padding sequences
        word_to_index: Dictionary mapping words to integer indices
        index_to_word: Dictionary mapping integer indices to words
        word_counts: Counter tracking word frequencies in the corpus
        is_fitted: Boolean indicating if vocabulary has been built
    """
    
    def __init__(
        self,
        vocabulary_size: int | None = None,
        unknown_token: str | None = None,
        padding_token: str | None = None,
    ) -> None:
        """Initialize the Tokenizer.
        
        Args:
            vocabulary_size: Maximum vocabulary size. Defaults to VOCABULARY_SIZE from config.
            unknown_token: Token for unknown words. Defaults to UNKNOWN_TOKEN from config.
            padding_token: Token for padding. Defaults to PADDING_TOKEN from config.
        """
        self.vocabulary_size: Final[int] = (
            vocabulary_size if vocabulary_size is not None else VOCABULARY_SIZE
        )
        self.unknown_token: Final[str] = (
            unknown_token if unknown_token is not None else UNKNOWN_TOKEN
        )
        self.padding_token: Final[str] = (
            padding_token if padding_token is not None else PADDING_TOKEN
        )
        
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}
        self.word_counts: Counter[str] = Counter()
        self.is_fitted: bool = False
        
        logger.info("Tokenizer initialized")
        logger.info(f"Vocabulary size: {self.vocabulary_size}")
        logger.info(f"Unknown token: {self.unknown_token}")
        logger.info(f"Padding token: {self.padding_token}")
    
    def build_vocabulary(
        self,
        corpus_dir: Path | None = None,
    ) -> dict[str, int | float]:
        """Build vocabulary from text corpus.
        
        Reads all text files from the corpus directory, counts word frequencies,
        and creates a vocabulary of the most frequent words. Special tokens
        (padding and unknown) are added to the vocabulary with reserved indices.
        
        Args:
            corpus_dir: Directory containing text files. Defaults to PROCESSED_DATA_DIR.
        
        Returns:
            Dictionary containing vocabulary statistics:
                - 'vocabulary_size': Number of words in vocabulary (including special tokens)
                - 'total_words': Total number of words in corpus
                - 'unique_words': Number of unique words in corpus
                - 'coverage': Percentage of corpus words covered by vocabulary
        
        Raises:
            FileNotFoundError: If corpus directory does not exist
            ValueError: If no text files found in corpus directory
        """
        corpus_dir = corpus_dir if corpus_dir is not None else PROCESSED_DATA_DIR
        
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
            f"Building vocabulary from corpus\n"
            f"Corpus directory: {corpus_dir}\n"
            f"Files: {len(text_files)}\n"
            f"Target vocabulary size: {self.vocabulary_size}",
            title="Vocabulary Building",
            style="bold blue",
            border_style="blue",
        )
        
        # Count word frequencies
        logger.info("Counting word frequencies...")
        
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Counting words...",
                total=len(text_files),
            )
            
            for text_file in text_files:
                try:
                    text = text_file.read_text(encoding='utf-8')
                    words = text.split()
                    self.word_counts.update(words)
                    logger.debug(f"Processed {text_file.name}: {len(words)} words")
                
                except Exception as e:
                    logger.error(f"Failed to read file '{text_file.name}': {e}")
                
                finally:
                    progress.update(task, advance=1)
        
        total_words = sum(self.word_counts.values())
        unique_words = len(self.word_counts)
        
        logger.info(f"Total words in corpus: {total_words:,}")
        logger.info(f"Unique words in corpus: {unique_words:,}")
        
        # Build vocabulary with most frequent words
        # Reserve indices 0 and 1 for special tokens
        self.word_to_index = {
            self.padding_token: 0,
            self.unknown_token: 1,
        }
        self.index_to_word = {
            0: self.padding_token,
            1: self.unknown_token,
        }
        
        # Get most frequent words (excluding special tokens if they exist in corpus)
        # We need vocabulary_size - 2 words (since we already have 2 special tokens)
        most_common_words = [
            word for word, _ in self.word_counts.most_common()
            if word not in {self.padding_token, self.unknown_token}
        ][:self.vocabulary_size - 2]
        
        # Assign indices to most frequent words (starting from index 2)
        for idx, word in enumerate(most_common_words, start=2):
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
        
        self.is_fitted = True
        
        # Calculate coverage (percentage of corpus words in vocabulary)
        words_in_vocab = sum(
            count for word, count in self.word_counts.items()
            if word in self.word_to_index
        )
        coverage = (words_in_vocab / total_words * 100) if total_words > 0 else 0.0
        
        logger.info(f"Vocabulary built with {len(self.word_to_index)} words")
        logger.info(f"Corpus coverage: {coverage:.2f}%")
        
        # Display vocabulary statistics
        self._display_vocabulary_stats(total_words, unique_words, coverage)
        
        return {
            'vocabulary_size': len(self.word_to_index),
            'total_words': total_words,
            'unique_words': unique_words,
            'coverage': coverage,
        }
    
    def text_to_sequence(self, text: str) -> list[int]:
        """Convert text to integer sequence.
        
        Tokenizes the input text and converts each word to its corresponding
        integer index. Out-of-vocabulary words are mapped to the unknown token index.
        
        Args:
            text: Input text string to convert
        
        Returns:
            List of integer indices representing the text
        
        Raises:
            ValueError: If vocabulary has not been built yet
        """
        if not self.is_fitted:
            error_msg = "Vocabulary not built. Call build_vocabulary() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        words = text.split()
        unknown_idx = self.word_to_index[self.unknown_token]
        
        sequence = [
            self.word_to_index.get(word, unknown_idx)
            for word in words
        ]
        
        logger.debug(f"Converted text to sequence: {len(words)} words -> {len(sequence)} indices")
        
        return sequence
    
    def sequence_to_text(self, sequence: list[int]) -> str:
        """Convert integer sequence to text.
        
        Converts a list of integer indices back to their corresponding words.
        Invalid indices are mapped to the unknown token.
        
        Args:
            sequence: List of integer indices to convert
        
        Returns:
            Text string reconstructed from the sequence
        
        Raises:
            ValueError: If vocabulary has not been built yet
        """
        if not self.is_fitted:
            error_msg = "Vocabulary not built. Call build_vocabulary() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        words = [
            self.index_to_word.get(idx, self.unknown_token)
            for idx in sequence
        ]
        
        text = ' '.join(words)
        
        logger.debug(f"Converted sequence to text: {len(sequence)} indices -> {len(words)} words")
        
        return text
    
    def save_vocabulary(self, filepath: Path | None = None) -> None:
        """Save vocabulary to JSON file.
        
        Saves the word-to-index mapping, vocabulary configuration, and statistics
        to a JSON file for later loading during inference.
        
        Args:
            filepath: Path to save vocabulary. Defaults to TOKENIZER_CONFIG_PATH.
        
        Raises:
            ValueError: If vocabulary has not been built yet
            IOError: If file cannot be written
        """
        if not self.is_fitted:
            error_msg = "Vocabulary not built. Call build_vocabulary() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        filepath = filepath if filepath is not None else TOKENIZER_CONFIG_PATH
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare vocabulary data
        vocab_data = {
            'vocabulary_size': self.vocabulary_size,
            'unknown_token': self.unknown_token,
            'padding_token': self.padding_token,
            'word_to_index': self.word_to_index,
            'index_to_word': {str(k): v for k, v in self.index_to_word.items()},
            'actual_vocabulary_size': len(self.word_to_index),
            'total_words_in_corpus': sum(self.word_counts.values()),
            'unique_words_in_corpus': len(self.word_counts),
        }
        
        # Save to JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Vocabulary saved to {filepath}")
            print_success(
                f"Vocabulary saved successfully!\n"
                f"Location: {filepath}\n"
                f"Size: {len(self.word_to_index)} words",
                title="Vocabulary Saved",
            )
        
        except Exception as e:
            error_msg = f"Failed to save vocabulary: {e}"
            logger.error(error_msg)
            print_error(error_msg, title="Save Failed")
            raise IOError(error_msg) from e
    
    def load_vocabulary(self, filepath: Path | None = None) -> None:
        """Load vocabulary from JSON file.
        
        Loads a previously saved vocabulary configuration from a JSON file,
        restoring the word-to-index and index-to-word mappings.
        
        Args:
            filepath: Path to load vocabulary from. Defaults to TOKENIZER_CONFIG_PATH.
        
        Raises:
            FileNotFoundError: If vocabulary file does not exist
            ValueError: If vocabulary file is invalid or corrupted
            IOError: If file cannot be read
        """
        filepath = filepath if filepath is not None else TOKENIZER_CONFIG_PATH
        
        if not filepath.exists():
            error_msg = f"Vocabulary file does not exist: {filepath}"
            logger.error(error_msg)
            print_error(error_msg, title="File Not Found")
            raise FileNotFoundError(error_msg)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Validate required fields
            required_fields = [
                'vocabulary_size',
                'unknown_token',
                'padding_token',
                'word_to_index',
                'index_to_word',
            ]
            
            missing_fields = [
                field for field in required_fields
                if field not in vocab_data
            ]
            
            if missing_fields:
                error_msg = f"Invalid vocabulary file. Missing fields: {missing_fields}"
                logger.error(error_msg)
                print_error(error_msg, title="Invalid File")
                raise ValueError(error_msg)
            
            # Load vocabulary data
            # Note: We don't override the instance's vocabulary_size, unknown_token, padding_token
            # as they were set during initialization. We just load the mappings.
            self.word_to_index = vocab_data['word_to_index']
            self.index_to_word = {
                int(k): v for k, v in vocab_data['index_to_word'].items()
            }
            self.is_fitted = True
            
            logger.info(f"Vocabulary loaded from {filepath}")
            logger.info(f"Vocabulary size: {len(self.word_to_index)}")
            
            print_success(
                f"Vocabulary loaded successfully!\n"
                f"Location: {filepath}\n"
                f"Size: {len(self.word_to_index)} words",
                title="Vocabulary Loaded",
            )
        
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse vocabulary file: {e}"
            logger.error(error_msg)
            print_error(error_msg, title="Parse Error")
            raise ValueError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Failed to load vocabulary: {e}"
            logger.error(error_msg)
            print_error(error_msg, title="Load Failed")
            raise IOError(error_msg) from e
    
    def get_vocabulary_size(self) -> int:
        """Get the actual vocabulary size.
        
        Returns:
            Number of words in the vocabulary (including special tokens)
        """
        return len(self.word_to_index)
    
    def get_word_index(self, word: str) -> int:
        """Get the index for a specific word.
        
        Args:
            word: Word to look up
        
        Returns:
            Integer index for the word, or unknown token index if not in vocabulary
        """
        return self.word_to_index.get(word, self.word_to_index[self.unknown_token])
    
    def get_index_word(self, index: int) -> str:
        """Get the word for a specific index.
        
        Args:
            index: Index to look up
        
        Returns:
            Word corresponding to the index, or unknown token if index is invalid
        """
        return self.index_to_word.get(index, self.unknown_token)
    
    def _display_vocabulary_stats(
        self,
        total_words: int,
        unique_words: int,
        coverage: float,
    ) -> None:
        """Display vocabulary statistics with Rich table.
        
        Args:
            total_words: Total number of words in corpus
            unique_words: Number of unique words in corpus
            coverage: Percentage of corpus words covered by vocabulary
        """
        # Create statistics table
        table = create_table(
            "Vocabulary Statistics",
            "Metric",
            "Value",
        )
        
        table.add_row("Target Vocabulary Size", f"{self.vocabulary_size:,}")
        table.add_row("Actual Vocabulary Size", f"{len(self.word_to_index):,}")
        table.add_row("Special Tokens", "2 (PAD, UNK)")
        table.add_row("Total Words in Corpus", f"{total_words:,}")
        table.add_row("Unique Words in Corpus", f"{unique_words:,}")
        table.add_row("Corpus Coverage", f"{coverage:.2f}%")
        table.add_row(
            "Out-of-Vocabulary Rate",
            f"{(100 - coverage):.2f}%"
        )
        
        console.print(table)
        
        # Display top 20 most frequent words
        top_words_table = create_table(
            "Top 20 Most Frequent Words",
            "Rank",
            "Word",
            "Frequency",
            "Index",
        )
        
        for rank, (word, count) in enumerate(self.word_counts.most_common(20), start=1):
            word_index = self.word_to_index.get(word, self.word_to_index[self.unknown_token])
            top_words_table.add_row(
                str(rank),
                word,
                f"{count:,}",
                str(word_index),
            )
        
        console.print(top_words_table)
        
        print_success(
            f"Vocabulary built successfully!\n"
            f"Size: {len(self.word_to_index):,} words\n"
            f"Coverage: {coverage:.2f}%",
            title="Vocabulary Building Complete",
        )


# ============================================================================
# Main Execution (for testing)
# ============================================================================


def main() -> None:
    """Main function to test tokenizer functionality."""
    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Build vocabulary
    stats = tokenizer.build_vocabulary()
    
    # Save vocabulary
    tokenizer.save_vocabulary()
    
    # Test text-to-sequence conversion
    test_text = "machine learning is a field of artificial intelligence"
    print_panel(
        f"Test text: {test_text}",
        title="Testing Text-to-Sequence",
        style="bold cyan",
        border_style="cyan",
    )
    
    sequence = tokenizer.text_to_sequence(test_text)
    console.print(f"Sequence: {sequence}\n", style="cyan")
    
    # Test sequence-to-text conversion
    reconstructed_text = tokenizer.sequence_to_text(sequence)
    console.print(f"Reconstructed: {reconstructed_text}\n", style="cyan")
    
    # Test with unknown words
    test_text_with_unknown = "machine learning uses xyzabc123 algorithms"
    print_panel(
        f"Test text with unknown word: {test_text_with_unknown}",
        title="Testing Unknown Token Handling",
        style="bold cyan",
        border_style="cyan",
    )
    
    sequence_with_unknown = tokenizer.text_to_sequence(test_text_with_unknown)
    console.print(f"Sequence: {sequence_with_unknown}\n", style="cyan")
    
    reconstructed_with_unknown = tokenizer.sequence_to_text(sequence_with_unknown)
    console.print(f"Reconstructed: {reconstructed_with_unknown}\n", style="cyan")
    
    # Display final statistics
    print_panel(
        f"Vocabulary Size: {stats['vocabulary_size']:,}\n"
        f"Total Words: {stats['total_words']:,}\n"
        f"Unique Words: {stats['unique_words']:,}\n"
        f"Coverage: {stats['coverage']:.2f}%",
        title="Final Statistics",
        style="bold green",
        border_style="green",
    )


if __name__ == "__main__":
    main()
