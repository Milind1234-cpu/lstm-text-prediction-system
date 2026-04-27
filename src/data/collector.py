"""Wikipedia data collection module.

This module provides functionality to collect Wikipedia articles on AI/ML topics
for training data. It retrieves articles using the Wikipedia API, extracts plain
text content, handles errors gracefully, and saves raw articles to disk.
"""

import logging
from pathlib import Path
from typing import Any, Final

import wikipedia  # type: ignore[import-untyped]
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..utils.config import WIKIPEDIA_TOPICS, RAW_DATA_DIR, NUM_ARTICLES
from ..utils.logger import (
    setup_logger,
    console,
    print_panel,
    print_success,
    print_error,
    create_table,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)

# ============================================================================
# DataCollector Class
# ============================================================================


class DataCollector:
    """Collects Wikipedia articles on AI/ML topics for training data.
    
    This class retrieves articles from Wikipedia, extracts plain text content
    without markup, handles retrieval errors gracefully, and saves raw articles
    to the data/raw directory.
    
    Attributes:
        topics: List of Wikipedia article topics to retrieve
        output_dir: Directory path where raw articles will be saved
        articles_collected: Number of articles successfully collected
        total_characters: Total character count across all collected articles
    """
    
    def __init__(
        self,
        topics: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize the DataCollector.
        
        Args:
            topics: List of Wikipedia topics to collect. Defaults to WIKIPEDIA_TOPICS from config.
            output_dir: Directory to save raw articles. Defaults to RAW_DATA_DIR from config.
        """
        self.topics: Final[list[str]] = topics if topics is not None else WIKIPEDIA_TOPICS
        self.output_dir: Final[Path] = output_dir if output_dir is not None else RAW_DATA_DIR
        self.articles_collected: int = 0
        self.total_characters: int = 0
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataCollector initialized with {len(self.topics)} topics")
        logger.info(f"Output directory: {self.output_dir}")
    
    def collect_articles(self) -> dict[str, int]:
        """Collect Wikipedia articles for all configured topics.
        
        Retrieves articles from Wikipedia API, extracts plain text content,
        handles errors gracefully with logging, and saves each article to
        a separate file in the output directory.
        
        Returns:
            Dictionary containing collection statistics:
                - 'articles_collected': Number of articles successfully retrieved
                - 'total_characters': Total character count across all articles
                - 'failed_retrievals': Number of failed article retrievals
        
        Raises:
            No exceptions are raised. All errors are logged and collection continues.
        """
        print_panel(
            f"Starting Wikipedia data collection\n"
            f"Topics: {len(self.topics)}\n"
            f"Output: {self.output_dir}",
            title="Data Collection",
            style="bold blue",
            border_style="blue",
        )
        
        failed_retrievals = 0
        
        # Create progress bar with Rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Collecting articles...",
                total=len(self.topics),
            )
            
            for topic in self.topics:
                try:
                    # Retrieve article content from Wikipedia
                    logger.info(f"Retrieving article: {topic}")
                    article_content = self._retrieve_article(topic)
                    
                    if article_content:
                        # Save article to file
                        self._save_article(topic, article_content)
                        self.articles_collected += 1
                        self.total_characters += len(article_content)
                        logger.info(
                            f"Successfully collected: {topic} "
                            f"({len(article_content)} characters)"
                        )
                    else:
                        failed_retrievals += 1
                        logger.warning(f"Empty content for article: {topic}")
                
                except Exception as e:
                    # Log error and continue with remaining articles
                    failed_retrievals += 1
                    logger.error(f"Failed to retrieve article '{topic}': {e}")
                    print_error(
                        f"Failed to retrieve: {topic}\nError: {str(e)}",
                        title="Retrieval Error",
                    )
                
                finally:
                    # Update progress bar
                    progress.update(task, advance=1)
        
        # Display collection summary
        self._display_summary(failed_retrievals)
        
        return {
            'articles_collected': self.articles_collected,
            'total_characters': self.total_characters,
            'failed_retrievals': failed_retrievals,
        }
    
    def _retrieve_article(self, topic: str) -> str:
        """Retrieve plain text content for a Wikipedia article.
        
        Args:
            topic: Wikipedia article title to retrieve
        
        Returns:
            Plain text content of the article without markup
        
        Raises:
            wikipedia.exceptions.DisambiguationError: If topic is ambiguous
            wikipedia.exceptions.PageError: If page does not exist
            Exception: For other Wikipedia API errors
        """
        try:
            # Set language to English
            wikipedia.set_lang("en")
            
            # Retrieve article page
            page = wikipedia.page(topic, auto_suggest=False)
            
            # Extract plain text content (no HTML markup)
            content: str = str(page.content)
            
            return content
        
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages by using the first option
            logger.warning(
                f"Disambiguation page for '{topic}'. "
                f"Using first option: {e.options[0] if e.options else 'None'}"
            )
            if e.options:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return str(page.content)
            raise
        
        except wikipedia.exceptions.PageError:
            logger.error(f"Wikipedia page not found: {topic}")
            raise
        
        except Exception as e:
            logger.error(f"Error retrieving Wikipedia article '{topic}': {e}")
            raise
    
    def _save_article(self, topic: str, content: str) -> None:
        """Save article content to a file in the output directory.
        
        Args:
            topic: Article topic (used for filename)
            content: Article text content to save
        
        Raises:
            IOError: If file cannot be written
        """
        # Create safe filename from topic (replace special characters)
        safe_filename = self._create_safe_filename(topic)
        filepath = self.output_dir / f"{safe_filename}.txt"
        
        try:
            # Write article content to file
            filepath.write_text(content, encoding='utf-8')
            logger.debug(f"Saved article to: {filepath}")
        
        except IOError as e:
            logger.error(f"Failed to save article '{topic}' to {filepath}: {e}")
            raise
    
    def _create_safe_filename(self, topic: str) -> str:
        """Create a safe filename from a topic string.
        
        Args:
            topic: Article topic string
        
        Returns:
            Safe filename string with special characters replaced
        """
        # Replace spaces with underscores and remove special characters
        safe_name = topic.replace(' ', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        return safe_name.lower()
    
    def _display_summary(self, failed_retrievals: int) -> None:
        """Display collection summary with statistics.
        
        Args:
            failed_retrievals: Number of failed article retrievals
        """
        # Create summary table
        table = create_table(
            "Collection Summary",
            "Metric",
            "Value",
        )
        
        table.add_row("Total Topics", str(len(self.topics)))
        table.add_row("Articles Collected", str(self.articles_collected))
        table.add_row("Failed Retrievals", str(failed_retrievals))
        table.add_row("Total Characters", f"{self.total_characters:,}")
        table.add_row(
            "Average Characters per Article",
            f"{self.total_characters // self.articles_collected:,}" 
            if self.articles_collected > 0 else "0"
        )
        
        console.print(table)
        
        # Display success or warning message
        if failed_retrievals == 0:
            print_success(
                f"Successfully collected all {self.articles_collected} articles!\n"
                f"Total characters: {self.total_characters:,}",
                title="Collection Complete",
            )
        else:
            print_panel(
                f"Collected {self.articles_collected} articles with {failed_retrievals} failures.\n"
                f"Total characters: {self.total_characters:,}",
                title="Collection Complete with Warnings",
                style="yellow",
                border_style="yellow",
            )
        
        logger.info(
            f"Collection complete: {self.articles_collected} articles, "
            f"{self.total_characters} characters, {failed_retrievals} failures"
        )


# ============================================================================
# Main Execution (for testing)
# ============================================================================


def main() -> None:
    """Main function to run data collection."""
    collector = DataCollector()
    stats = collector.collect_articles()
    
    print_panel(
        f"Articles: {stats['articles_collected']}\n"
        f"Characters: {stats['total_characters']:,}\n"
        f"Failures: {stats['failed_retrievals']}",
        title="Final Statistics",
        style="bold green",
        border_style="green",
    )


if __name__ == "__main__":
    main()
