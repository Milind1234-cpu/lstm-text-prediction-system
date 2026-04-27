"""Text preprocessing module.

This module provides functionality to clean and preprocess raw text data for
LSTM training. It performs lowercase conversion, removes URLs/emails/special
characters, normalizes whitespace, filters short lines, and preserves sentence
boundaries while saving cleaned text to the processed directory.
"""

import logging
import re
from pathlib import Path
from typing import Final

from ..utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MIN_WORDS_PER_LINE,
)
from ..utils.logger import (
    setup_logger,
    console,
    print_panel,
    print_success,
    create_progress_bar,
    create_table,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)

# ============================================================================
# Regular Expression Patterns
# ============================================================================

# Pattern to match URLs (http, https, ftp, www)
URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
    r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    re.IGNORECASE
)

# Pattern to match email addresses
EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    re.IGNORECASE
)

# Pattern to match special characters (keep only letters, numbers, basic punctuation, and whitespace)
SPECIAL_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'[^a-zA-Z0-9\s.,!?;:\'\"-]'
)

# Pattern to match multiple whitespace characters
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r'\s+')

# ============================================================================
# TextPreprocessor Class
# ============================================================================


class TextPreprocessor:
    """Preprocesses and cleans raw text data for LSTM training.
    
    This class performs comprehensive text cleaning including lowercase conversion,
    URL/email/special character removal, whitespace normalization, short line
    filtering, and sentence boundary preservation. Processed text is saved to
    the data/processed directory with progress tracking.
    
    Attributes:
        input_dir: Directory path containing raw text files
        output_dir: Directory path where processed text will be saved
        min_words_per_line: Minimum number of words required to keep a line
        lines_processed: Total number of lines processed
        lines_kept: Number of lines kept after filtering
        lines_removed: Number of lines removed during filtering
        total_characters: Total character count in processed text
    """
    
    def __init__(
        self,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        min_words_per_line: int | None = None,
    ) -> None:
        """Initialize the TextPreprocessor.
        
        Args:
            input_dir: Directory containing raw text files. Defaults to RAW_DATA_DIR from config.
            output_dir: Directory to save processed text. Defaults to PROCESSED_DATA_DIR from config.
            min_words_per_line: Minimum words per line to keep. Defaults to MIN_WORDS_PER_LINE from config.
        """
        self.input_dir: Final[Path] = input_dir if input_dir is not None else RAW_DATA_DIR
        self.output_dir: Final[Path] = output_dir if output_dir is not None else PROCESSED_DATA_DIR
        self.min_words_per_line: Final[int] = (
            min_words_per_line if min_words_per_line is not None else MIN_WORDS_PER_LINE
        )
        
        self.lines_processed: int = 0
        self.lines_kept: int = 0
        self.lines_removed: int = 0
        self.total_characters: int = 0
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TextPreprocessor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Minimum words per line: {self.min_words_per_line}")
    
    def preprocess_all_files(self) -> dict[str, int]:
        """Preprocess all text files in the input directory.
        
        Reads all .txt files from the input directory, applies preprocessing
        to each file, and saves the cleaned text to the output directory.
        Displays progress using Rich progress bars.
        
        Returns:
            Dictionary containing preprocessing statistics:
                - 'files_processed': Number of files successfully processed
                - 'lines_processed': Total number of lines processed
                - 'lines_kept': Number of lines kept after filtering
                - 'lines_removed': Number of lines removed during filtering
                - 'total_characters': Total character count in processed text
        
        Raises:
            FileNotFoundError: If input directory does not exist
            IOError: If files cannot be read or written
        """
        # Get all text files in input directory
        text_files = list(self.input_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in {self.input_dir}")
            print_panel(
                f"No text files found in {self.input_dir}",
                title="Warning",
                style="yellow",
                border_style="yellow",
            )
            return {
                'files_processed': 0,
                'lines_processed': 0,
                'lines_kept': 0,
                'lines_removed': 0,
                'total_characters': 0,
            }
        
        print_panel(
            f"Starting text preprocessing\n"
            f"Input: {self.input_dir}\n"
            f"Output: {self.output_dir}\n"
            f"Files: {len(text_files)}\n"
            f"Min words per line: {self.min_words_per_line}",
            title="Text Preprocessing",
            style="bold blue",
            border_style="blue",
        )
        
        files_processed = 0
        
        # Create progress bar with Rich
        with create_progress_bar() as progress:
            task = progress.add_task(
                "Processing files...",
                total=len(text_files),
            )
            
            for text_file in text_files:
                try:
                    logger.info(f"Processing file: {text_file.name}")
                    
                    # Read raw text
                    raw_text = text_file.read_text(encoding='utf-8')
                    
                    # Preprocess text
                    processed_text = self.preprocess_text(raw_text)
                    
                    # Save processed text
                    output_file = self.output_dir / text_file.name
                    output_file.write_text(processed_text, encoding='utf-8')
                    
                    files_processed += 1
                    logger.info(
                        f"Successfully processed: {text_file.name} "
                        f"({len(processed_text)} characters)"
                    )
                
                except Exception as e:
                    logger.error(f"Failed to process file '{text_file.name}': {e}")
                
                finally:
                    progress.update(task, advance=1)
        
        # Display preprocessing summary
        self._display_summary(files_processed)
        
        return {
            'files_processed': files_processed,
            'lines_processed': self.lines_processed,
            'lines_kept': self.lines_kept,
            'lines_removed': self.lines_removed,
            'total_characters': self.total_characters,
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text string.
        
        Applies all preprocessing steps to the input text:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove special characters
        5. Normalize whitespace
        6. Filter lines with fewer than min_words_per_line words
        7. Preserve sentence boundaries
        
        Args:
            text: Raw text string to preprocess
        
        Returns:
            Cleaned and preprocessed text string
        """
        # Step 1: Convert to lowercase
        text = text.lower()
        logger.debug("Applied lowercase conversion")
        
        # Step 2: Remove URLs
        text = URL_PATTERN.sub('', text)
        logger.debug("Removed URLs")
        
        # Step 3: Remove email addresses
        text = EMAIL_PATTERN.sub('', text)
        logger.debug("Removed email addresses")
        
        # Step 4: Remove special characters (keep letters, numbers, basic punctuation)
        text = SPECIAL_CHAR_PATTERN.sub('', text)
        logger.debug("Removed special characters")
        
        # Step 5: Normalize whitespace to single spaces
        text = WHITESPACE_PATTERN.sub(' ', text)
        logger.debug("Normalized whitespace")
        
        # Step 6: Filter lines with fewer than min_words_per_line words
        # Split by newlines to preserve sentence boundaries
        lines = text.split('\n')
        filtered_lines: list[str] = []
        
        for line in lines:
            self.lines_processed += 1
            
            # Strip leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                self.lines_removed += 1
                continue
            
            # Count words in line
            words = line.split()
            
            # Keep line if it has enough words
            if len(words) >= self.min_words_per_line:
                filtered_lines.append(line)
                self.lines_kept += 1
            else:
                self.lines_removed += 1
                logger.debug(
                    f"Removed line with {len(words)} words "
                    f"(min: {self.min_words_per_line})"
                )
        
        # Step 7: Join lines with newlines to preserve sentence boundaries
        processed_text = '\n'.join(filtered_lines)
        
        # Update character count
        self.total_characters += len(processed_text)
        
        logger.debug(
            f"Preprocessing complete: {len(processed_text)} characters, "
            f"{len(filtered_lines)} lines kept"
        )
        
        return processed_text
    
    def _display_summary(self, files_processed: int) -> None:
        """Display preprocessing summary with statistics.
        
        Args:
            files_processed: Number of files successfully processed
        """
        # Create summary table
        table = create_table(
            "Preprocessing Summary",
            "Metric",
            "Value",
        )
        
        table.add_row("Files Processed", str(files_processed))
        table.add_row("Lines Processed", f"{self.lines_processed:,}")
        table.add_row("Lines Kept", f"{self.lines_kept:,}")
        table.add_row("Lines Removed", f"{self.lines_removed:,}")
        table.add_row(
            "Retention Rate",
            f"{(self.lines_kept / self.lines_processed * 100):.1f}%"
            if self.lines_processed > 0 else "0%"
        )
        table.add_row("Total Characters", f"{self.total_characters:,}")
        table.add_row(
            "Average Characters per File",
            f"{self.total_characters // files_processed:,}"
            if files_processed > 0 else "0"
        )
        
        console.print(table)
        
        # Display success message
        print_success(
            f"Successfully preprocessed {files_processed} files!\n"
            f"Lines kept: {self.lines_kept:,} / {self.lines_processed:,}\n"
            f"Total characters: {self.total_characters:,}",
            title="Preprocessing Complete",
        )
        
        logger.info(
            f"Preprocessing complete: {files_processed} files, "
            f"{self.lines_kept} lines kept, {self.lines_removed} lines removed, "
            f"{self.total_characters} characters"
        )


# ============================================================================
# Main Execution (for testing)
# ============================================================================


def main() -> None:
    """Main function to run text preprocessing."""
    preprocessor = TextPreprocessor()
    stats = preprocessor.preprocess_all_files()
    
    print_panel(
        f"Files: {stats['files_processed']}\n"
        f"Lines Kept: {stats['lines_kept']:,}\n"
        f"Lines Removed: {stats['lines_removed']:,}\n"
        f"Characters: {stats['total_characters']:,}",
        title="Final Statistics",
        style="bold green",
        border_style="green",
    )


if __name__ == "__main__":
    main()
