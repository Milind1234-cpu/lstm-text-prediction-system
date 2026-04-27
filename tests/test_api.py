"""Comprehensive tests for FastAPI endpoints.

This module tests all API endpoints with valid and invalid inputs,
CORS headers, and request logging.

**Validates: Requirements 21.1, 21.5**
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from src.api.app import app
from src.model.predictor import Predictor
from src.data.tokenizer import Tokenizer
from src.utils.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    MAX_BATCH_SIZE,
    MAX_TOP_K,
    VOCABULARY_SIZE,
)


# ============================================================================
# Test Client and Fixtures
# ============================================================================


@pytest.fixture
def mock_predictor():
    """Create a mock Predictor for testing."""
    predictor = MagicMock(spec=Predictor)
    predictor.is_loaded = True
    predictor.sequence_length = 50
    predictor.model = MagicMock()
    predictor.model.count_params.return_value = 15234560
    predictor.model.trainable_weights = [MagicMock()]
    predictor.model.trainable_weights[0].numpy.return_value = np.zeros(15234560)
    
    # Mock tokenizer
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1,
        "machine": 2,
        "learning": 3,
        "is": 4,
        "powerful": 5,
    }
    predictor.tokenizer = tokenizer
    
    # Mock prediction methods
    predictor.predict_next_word.return_value = "powerful"
    predictor.predict_top_k.return_value = [
        {"word": "powerful", "probability": 0.234},
        {"word": "useful", "probability": 0.156},
        {"word": "important", "probability": 0.123},
    ]
    predictor.predict_batch.return_value = ["powerful", "revolutionary"]
    predictor.complete_text.return_value = "machine learning is powerful and useful."
    
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Create test client with mocked predictor."""
    # Patch the predictor loading in app startup
    with patch('src.api.app.Predictor', return_value=mock_predictor):
        with patch('src.api.endpoints.prediction._predictor', mock_predictor):
            with patch('src.api.endpoints.health._predictor', mock_predictor):
                with patch('src.api.endpoints.model_info._predictor', mock_predictor):
                    client = TestClient(app)
                    yield client


# ============================================================================
# Root Endpoint Tests
# ============================================================================


class TestRootEndpoint:
    """Test suite for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test GET / returns welcome message."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs_url" in data
        assert data["docs_url"] == "/docs"


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_check_healthy(self, client, mock_predictor):
        """Test GET /health returns healthy status."""
        mock_predictor.is_loaded = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu_available" in data
        assert "uptime" in data
    
    def test_health_check_unhealthy(self, client, mock_predictor):
        """Test GET /health returns unhealthy when model not loaded."""
        mock_predictor.is_loaded = False
        
        response = client.get("/health")
        
        # Should still return 200 but with unhealthy status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


# ============================================================================
# Metrics Endpoint Tests
# ============================================================================


class TestMetricsEndpoint:
    """Test suite for metrics endpoint."""
    
    def test_get_metrics(self, client):
        """Test GET /metrics returns usage statistics."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_response_time" in data
        assert "total_predictions" in data
        assert "errors" in data


# ============================================================================
# Prediction Endpoint Tests
# ============================================================================


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""
    
    def test_predict_valid_input(self, client, mock_predictor):
        """Test POST /predict with valid input."""
        request_data = {
            "text": "machine learning is",
            "temperature": 1.0
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "input_text" in data
        assert data["prediction"] == "powerful"
        assert data["input_text"] == "machine learning is"
        
        # Verify predictor was called
        mock_predictor.predict_next_word.assert_called_once()
    
    def test_predict_default_temperature(self, client, mock_predictor):
        """Test POST /predict with default temperature."""
        request_data = {
            "text": "neural networks"
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        # Should use default temperature
        mock_predictor.predict_next_word.assert_called_with(
            text="neural networks",
            temperature=DEFAULT_TEMPERATURE
        )
    
    def test_predict_empty_text(self, client):
        """Test POST /predict with empty text."""
        request_data = {
            "text": "",
            "temperature": 1.0
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_temperature_too_low(self, client):
        """Test POST /predict with temperature too low."""
        request_data = {
            "text": "test text",
            "temperature": 0.05  # Below minimum
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_temperature_too_high(self, client):
        """Test POST /predict with temperature too high."""
        request_data = {
            "text": "test text",
            "temperature": 3.0  # Above maximum
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_predictor_error(self, client, mock_predictor):
        """Test POST /predict handles predictor errors."""
        mock_predictor.predict_next_word.side_effect = ValueError("Invalid input")
        
        request_data = {
            "text": "test text",
            "temperature": 1.0
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 400  # Bad request


# ============================================================================
# Top-K Prediction Endpoint Tests
# ============================================================================


class TestTopKEndpoint:
    """Test suite for /predict/top-k endpoint."""
    
    def test_top_k_valid_input(self, client, mock_predictor):
        """Test POST /predict/top-k with valid input."""
        request_data = {
            "text": "machine learning is",
            "k": 3,
            "temperature": 1.0
        }
        
        response = client.post("/predict/top-k", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "input_text" in data
        assert len(data["predictions"]) == 3
        assert data["predictions"][0]["word"] == "powerful"
        assert "probability" in data["predictions"][0]
        
        # Verify predictor was called
        mock_predictor.predict_top_k.assert_called_once()
    
    def test_top_k_default_k(self, client, mock_predictor):
        """Test POST /predict/top-k with default k."""
        request_data = {
            "text": "neural networks"
        }
        
        response = client.post("/predict/top-k", json=request_data)
        
        assert response.status_code == 200
        # Should use default k
        mock_predictor.predict_top_k.assert_called_with(
            text="neural networks",
            k=DEFAULT_TOP_K,
            temperature=DEFAULT_TEMPERATURE
        )
    
    def test_top_k_invalid_k_zero(self, client):
        """Test POST /predict/top-k with k=0."""
        request_data = {
            "text": "test text",
            "k": 0
        }
        
        response = client.post("/predict/top-k", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_top_k_invalid_k_too_large(self, client):
        """Test POST /predict/top-k with k exceeding maximum."""
        request_data = {
            "text": "test text",
            "k": MAX_TOP_K + 1
        }
        
        response = client.post("/predict/top-k", json=request_data)
        
        assert response.status_code == 422  # Validation error


# ============================================================================
# Batch Prediction Endpoint Tests
# ============================================================================


class TestBatchEndpoint:
    """Test suite for /predict/batch endpoint."""
    
    def test_batch_valid_input(self, client, mock_predictor):
        """Test POST /predict/batch with valid input."""
        request_data = {
            "texts": ["machine learning is", "artificial intelligence is"],
            "temperature": 1.0
        }
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "input_texts" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0] == "powerful"
        assert data["predictions"][1] == "revolutionary"
        
        # Verify predictor was called
        mock_predictor.predict_batch.assert_called_once()
    
    def test_batch_empty_list(self, client):
        """Test POST /predict/batch with empty texts list."""
        request_data = {
            "texts": [],
            "temperature": 1.0
        }
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_empty_text_in_list(self, client):
        """Test POST /predict/batch with empty text in list."""
        request_data = {
            "texts": ["valid text", "", "another valid text"],
            "temperature": 1.0
        }
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_exceeds_max_size(self, client):
        """Test POST /predict/batch with batch size exceeding maximum."""
        request_data = {
            "texts": ["text"] * (MAX_BATCH_SIZE + 1),
            "temperature": 1.0
        }
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 422  # Validation error


# ============================================================================
# Text Completion Endpoint Tests
# ============================================================================


class TestCompleteEndpoint:
    """Test suite for /predict/complete endpoint."""
    
    def test_complete_valid_input(self, client, mock_predictor):
        """Test POST /predict/complete with valid input."""
        request_data = {
            "text": "machine learning is",
            "max_length": 10,
            "temperature": 1.0
        }
        
        response = client.post("/predict/complete", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "completion" in data
        assert "input_text" in data
        assert "stopped_by" in data
        assert data["completion"] == "machine learning is powerful and useful."
        
        # Verify predictor was called
        mock_predictor.complete_text.assert_called_once()
    
    def test_complete_with_stop_words(self, client, mock_predictor):
        """Test POST /predict/complete with custom stop words."""
        request_data = {
            "text": "the future of AI",
            "max_length": 20,
            "stop_words": [".", "!", "?"],
            "temperature": 1.0
        }
        
        response = client.post("/predict/complete", json=request_data)
        
        assert response.status_code == 200
        # Verify stop_words were passed
        mock_predictor.complete_text.assert_called_with(
            text="the future of AI",
            max_length=20,
            stop_words=[".", "!", "?"],
            temperature=1.0
        )
    
    def test_complete_invalid_max_length(self, client):
        """Test POST /predict/complete with invalid max_length."""
        request_data = {
            "text": "test text",
            "max_length": 0
        }
        
        response = client.post("/predict/complete", json=request_data)
        
        assert response.status_code == 422  # Validation error


# ============================================================================
# Model Info Endpoint Tests
# ============================================================================


class TestModelInfoEndpoint:
    """Test suite for /model/info endpoint."""
    
    def test_model_info(self, client, mock_predictor):
        """Test GET /model/info returns model architecture."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "lstm_equations" in data
        assert "parameters" in data
        
        # Verify architecture details
        assert "embedding_dim" in data["architecture"]
        assert "bidirectional_lstm_units" in data["architecture"]
        assert "vocabulary_size" in data["architecture"]
        
        # Verify LSTM equations
        assert "forget_gate" in data["lstm_equations"]
        assert "input_gate" in data["lstm_equations"]
        
        # Verify parameters
        assert "total_params" in data["parameters"]
        assert "trainable_params" in data["parameters"]


# ============================================================================
# Vocabulary Endpoint Tests
# ============================================================================


class TestVocabularyEndpoint:
    """Test suite for /model/vocabulary endpoint."""
    
    def test_vocabulary_search_with_query(self, client, mock_predictor):
        """Test GET /model/vocabulary with search query."""
        response = client.get("/model/vocabulary?query=learn")
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "matches" in data
        assert "total_matches" in data
        assert data["query"] == "learn"
    
    def test_vocabulary_list_without_query(self, client, mock_predictor):
        """Test GET /model/vocabulary without query (list mode)."""
        response = client.get("/model/vocabulary")
        
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert "total_matches" in data
        assert data["query"] is None
        # Should return some vocabulary words
        assert len(data["matches"]) > 0


# ============================================================================
# CORS Tests
# ============================================================================


class TestCORS:
    """Test suite for CORS headers."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/")
        
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
    
    def test_cors_preflight_request(self, client):
        """Test CORS preflight OPTIONS request."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method."""
        response = client.get("/predict")  # Should be POST
        
        assert response.status_code == 405
    
    def test_422_validation_error(self, client):
        """Test 422 validation error for invalid request body."""
        response = client.post("/predict", json={"invalid": "data"})
        
        assert response.status_code == 422


# ============================================================================
# Temperature Variation Tests
# ============================================================================


class TestTemperatureVariation:
    """Test suite for temperature parameter effects."""
    
    def test_temperature_low(self, client, mock_predictor):
        """Test prediction with low temperature (more confident)."""
        request_data = {
            "text": "test text",
            "temperature": 0.1
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        # Verify temperature was passed correctly
        mock_predictor.predict_next_word.assert_called_with(
            text="test text",
            temperature=0.1
        )
    
    def test_temperature_high(self, client, mock_predictor):
        """Test prediction with high temperature (more random)."""
        request_data = {
            "text": "test text",
            "temperature": 2.0
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        # Verify temperature was passed correctly
        mock_predictor.predict_next_word.assert_called_with(
            text="test text",
            temperature=2.0
        )
