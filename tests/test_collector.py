"""Unit tests for Wikipedia data collector.

Tests the DataCollector class functionality including article retrieval,
error handling, and file saving.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import wikipedia

from src.data.collector import DataCollector


class TestDataCollector:
    """Test suite for DataCollector class."""
    
    def test_initialization(self, tmp_path: Path) -> None:
        """Test DataCollector initialization with custom parameters."""
        topics = ["Test Topic 1", "Test Topic 2"]
        collector = DataCollector(topics=topics, output_dir=tmp_path)
        
        assert collector.topics == topics
        assert collector.output_dir == tmp_path
        assert collector.articles_collected == 0
        assert collector.total_characters == 0
        assert tmp_path.exists()
    
    def test_initialization_with_defaults(self) -> None:
        """Test DataCollector initialization with default parameters."""
        collector = DataCollector()
        
        assert len(collector.topics) == 20  # Default WIKIPEDIA_TOPICS has 20 topics
        assert collector.output_dir.exists()
    
    def test_create_safe_filename(self, tmp_path: Path) -> None:
        """Test safe filename creation from topic strings."""
        collector = DataCollector(topics=[], output_dir=tmp_path)
        
        # Test with spaces
        assert collector._create_safe_filename("Machine Learning") == "machine_learning"
        
        # Test with special characters
        assert collector._create_safe_filename("Transformer (ML)") == "transformer_ml"
        
        # Test with multiple spaces
        assert collector._create_safe_filename("Long  Short  Term") == "long__short__term"
    
    @patch('src.data.collector.wikipedia.page')
    def test_retrieve_article_success(self, mock_page: Mock, tmp_path: Path) -> None:
        """Test successful article retrieval from Wikipedia."""
        # Mock Wikipedia page
        mock_page_obj = Mock()
        mock_page_obj.content = "This is test article content."
        mock_page.return_value = mock_page_obj
        
        collector = DataCollector(topics=[], output_dir=tmp_path)
        content = collector._retrieve_article("Test Topic")
        
        assert content == "This is test article content."
        mock_page.assert_called_once_with("Test Topic", auto_suggest=False)
    
    @patch('src.data.collector.wikipedia.page')
    def test_retrieve_article_disambiguation(self, mock_page: Mock, tmp_path: Path) -> None:
        """Test article retrieval with disambiguation page."""
        # First call raises DisambiguationError, second call succeeds
        disambiguation_error = wikipedia.exceptions.DisambiguationError(
            "Test",
            ["Option 1", "Option 2"]
        )
        
        mock_page_obj = Mock()
        mock_page_obj.content = "Disambiguated content."
        
        mock_page.side_effect = [disambiguation_error, mock_page_obj]
        
        collector = DataCollector(topics=[], output_dir=tmp_path)
        content = collector._retrieve_article("Test Topic")
        
        assert content == "Disambiguated content."
        assert mock_page.call_count == 2
    
    @patch('src.data.collector.wikipedia.page')
    def test_retrieve_article_page_not_found(self, mock_page: Mock, tmp_path: Path) -> None:
        """Test article retrieval with non-existent page."""
        mock_page.side_effect = wikipedia.exceptions.PageError("Test")
        
        collector = DataCollector(topics=[], output_dir=tmp_path)
        
        with pytest.raises(wikipedia.exceptions.PageError):
            collector._retrieve_article("Nonexistent Topic")
    
    def test_save_article(self, tmp_path: Path) -> None:
        """Test saving article content to file."""
        collector = DataCollector(topics=[], output_dir=tmp_path)
        
        topic = "Test Topic"
        content = "This is test content for the article."
        
        collector._save_article(topic, content)
        
        # Verify file was created
        expected_file = tmp_path / "test_topic.txt"
        assert expected_file.exists()
        
        # Verify content
        saved_content = expected_file.read_text(encoding='utf-8')
        assert saved_content == content
    
    @patch('src.data.collector.wikipedia.page')
    def test_collect_articles_success(self, mock_page: Mock, tmp_path: Path) -> None:
        """Test successful collection of multiple articles."""
        # Mock Wikipedia pages
        mock_page_obj1 = Mock()
        mock_page_obj1.content = "Content for article 1."
        
        mock_page_obj2 = Mock()
        mock_page_obj2.content = "Content for article 2."
        
        mock_page.side_effect = [mock_page_obj1, mock_page_obj2]
        
        topics = ["Topic 1", "Topic 2"]
        collector = DataCollector(topics=topics, output_dir=tmp_path)
        
        stats = collector.collect_articles()
        
        # Verify statistics
        assert stats['articles_collected'] == 2
        assert stats['failed_retrievals'] == 0
        assert stats['total_characters'] == len("Content for article 1.") + len("Content for article 2.")
        
        # Verify files were created
        assert (tmp_path / "topic_1.txt").exists()
        assert (tmp_path / "topic_2.txt").exists()
    
    @patch('src.data.collector.wikipedia.page')
    def test_collect_articles_with_failures(self, mock_page: Mock, tmp_path: Path) -> None:
        """Test article collection with some failures."""
        # First article succeeds, second fails
        mock_page_obj = Mock()
        mock_page_obj.content = "Content for article 1."
        
        mock_page.side_effect = [
            mock_page_obj,
            wikipedia.exceptions.PageError("Topic 2")
        ]
        
        topics = ["Topic 1", "Topic 2"]
        collector = DataCollector(topics=topics, output_dir=tmp_path)
        
        stats = collector.collect_articles()
        
        # Verify statistics
        assert stats['articles_collected'] == 1
        assert stats['failed_retrievals'] == 1
        assert stats['total_characters'] == len("Content for article 1.")
        
        # Verify only successful article was saved
        assert (tmp_path / "topic_1.txt").exists()
        assert not (tmp_path / "topic_2.txt").exists()
    
    @patch('src.data.collector.wikipedia.page')
    def test_collect_articles_error_handling(self, mock_page: Mock, tmp_path: Path) -> None:
        """Test that errors are logged and collection continues."""
        # All articles fail with different errors
        mock_page.side_effect = [
            wikipedia.exceptions.PageError("Topic 1"),
            Exception("Network error"),
            wikipedia.exceptions.PageError("Topic 3")
        ]
        
        topics = ["Topic 1", "Topic 2", "Topic 3"]
        collector = DataCollector(topics=topics, output_dir=tmp_path)
        
        stats = collector.collect_articles()
        
        # Verify all failed but collection completed
        assert stats['articles_collected'] == 0
        assert stats['failed_retrievals'] == 3
        assert stats['total_characters'] == 0
