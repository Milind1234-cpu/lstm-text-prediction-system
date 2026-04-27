"""Performance tests for LSTM text prediction API.

This module tests performance requirements including:
- Average response time < 500ms
- Batch prediction performance
- Concurrent request handling

**Validates: Requirements 21.6**
"""

import pytest
import time
import concurrent.futures
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from src.api.app import app
from src.model.predictor import Predictor
from src.data.tokenizer import Tokenizer


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
    
    # Mock prediction methods with realistic delays
    def predict_with_delay(*args, **kwargs):
        time.sleep(0.01)  # Simulate 10ms processing time
        return "powerful"
    
    def predict_top_k_with_delay(*args, **kwargs):
        time.sleep(0.015)  # Simulate 15ms processing time
        return [
            {"word": "powerful", "probability": 0.234},
            {"word": "useful", "probability": 0.156},
            {"word": "important", "probability": 0.123},
        ]
    
    def predict_batch_with_delay(texts, *args, **kwargs):
        time.sleep(0.005 * len(texts))  # Simulate 5ms per text
        return ["powerful"] * len(texts)
    
    def complete_text_with_delay(*args, **kwargs):
        time.sleep(0.02)  # Simulate 20ms processing time
        return "machine learning is powerful and useful."
    
    predictor.predict_next_word.side_effect = predict_with_delay
    predictor.predict_top_k.side_effect = predict_top_k_with_delay
    predictor.predict_batch.side_effect = predict_batch_with_delay
    predictor.complete_text.side_effect = complete_text_with_delay
    
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Create test client with mocked predictor."""
    with patch('src.api.app.Predictor', return_value=mock_predictor):
        with patch('src.api.endpoints.prediction._predictor', mock_predictor):
            with patch('src.api.endpoints.health._predictor', mock_predictor):
                with patch('src.api.endpoints.model_info._predictor', mock_predictor):
                    client = TestClient(app)
                    yield client


# ============================================================================
# Response Time Tests
# ============================================================================


class TestResponseTime:
    """Test suite for response time requirements."""
    
    def test_predict_response_time(self, client):
        """Test that /predict responds in less than 500ms."""
        request_data = {
            "text": "machine learning is",
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 500, f"Response time {response_time_ms:.2f}ms exceeds 500ms limit"
    
    def test_top_k_response_time(self, client):
        """Test that /predict/top-k responds in less than 500ms."""
        request_data = {
            "text": "machine learning is",
            "k": 5,
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict/top-k", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 500, f"Response time {response_time_ms:.2f}ms exceeds 500ms limit"
    
    def test_batch_response_time(self, client):
        """Test that /predict/batch responds in less than 500ms for small batches."""
        request_data = {
            "texts": ["machine learning", "deep learning", "neural networks"],
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict/batch", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 500, f"Response time {response_time_ms:.2f}ms exceeds 500ms limit"
    
    def test_complete_response_time(self, client):
        """Test that /predict/complete responds in less than 500ms."""
        request_data = {
            "text": "machine learning is",
            "max_length": 10,
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict/complete", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 500, f"Response time {response_time_ms:.2f}ms exceeds 500ms limit"
    
    def test_health_response_time(self, client):
        """Test that /health responds quickly (< 100ms)."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 100, f"Health check response time {response_time_ms:.2f}ms exceeds 100ms limit"
    
    def test_metrics_response_time(self, client):
        """Test that /metrics responds quickly (< 100ms)."""
        start_time = time.time()
        response = client.get("/metrics")
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 100, f"Metrics response time {response_time_ms:.2f}ms exceeds 100ms limit"


# ============================================================================
# Average Response Time Tests
# ============================================================================


class TestAverageResponseTime:
    """Test suite for average response time over multiple requests."""
    
    def test_predict_average_response_time(self, client):
        """Test average response time for /predict over 10 requests."""
        request_data = {
            "text": "machine learning is",
            "temperature": 1.0
        }
        
        response_times = []
        num_requests = 10
        
        for _ in range(num_requests):
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)
        
        avg_response_time = sum(response_times) / len(response_times)
        
        assert avg_response_time < 500, f"Average response time {avg_response_time:.2f}ms exceeds 500ms limit"
        print(f"\n/predict average response time: {avg_response_time:.2f}ms")
    
    def test_top_k_average_response_time(self, client):
        """Test average response time for /predict/top-k over 10 requests."""
        request_data = {
            "text": "neural networks are",
            "k": 5,
            "temperature": 1.0
        }
        
        response_times = []
        num_requests = 10
        
        for _ in range(num_requests):
            start_time = time.time()
            response = client.post("/predict/top-k", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)
        
        avg_response_time = sum(response_times) / len(response_times)
        
        assert avg_response_time < 500, f"Average response time {avg_response_time:.2f}ms exceeds 500ms limit"
        print(f"\n/predict/top-k average response time: {avg_response_time:.2f}ms")


# ============================================================================
# Batch Performance Tests
# ============================================================================


class TestBatchPerformance:
    """Test suite for batch prediction performance."""
    
    def test_batch_small_size(self, client):
        """Test batch prediction with small batch (5 texts)."""
        request_data = {
            "texts": [f"text {i}" for i in range(5)],
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict/batch", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 5
        assert response_time_ms < 500
        print(f"\nBatch (5 texts) response time: {response_time_ms:.2f}ms")
    
    def test_batch_medium_size(self, client):
        """Test batch prediction with medium batch (10 texts)."""
        request_data = {
            "texts": [f"text {i}" for i in range(10)],
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict/batch", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 10
        assert response_time_ms < 500
        print(f"\nBatch (10 texts) response time: {response_time_ms:.2f}ms")
    
    def test_batch_large_size(self, client):
        """Test batch prediction with large batch (20 texts - maximum)."""
        request_data = {
            "texts": [f"text {i}" for i in range(20)],
            "temperature": 1.0
        }
        
        start_time = time.time()
        response = client.post("/predict/batch", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 20
        # Allow more time for larger batches
        assert response_time_ms < 1000
        print(f"\nBatch (20 texts) response time: {response_time_ms:.2f}ms")
    
    def test_batch_efficiency(self, client):
        """Test that batch prediction is more efficient than individual requests."""
        # Time for 5 individual requests
        individual_times = []
        for i in range(5):
            request_data = {
                "text": f"text {i}",
                "temperature": 1.0
            }
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()
            assert response.status_code == 200
            individual_times.append(end_time - start_time)
        
        total_individual_time = sum(individual_times) * 1000
        
        # Time for batch request with 5 texts
        batch_request_data = {
            "texts": [f"text {i}" for i in range(5)],
            "temperature": 1.0
        }
        start_time = time.time()
        response = client.post("/predict/batch", json=batch_request_data)
        end_time = time.time()
        assert response.status_code == 200
        
        batch_time = (end_time - start_time) * 1000
        
        # Batch should be faster than individual requests
        # (allowing some overhead for test environment)
        print(f"\nIndividual requests (5x): {total_individual_time:.2f}ms")
        print(f"Batch request (5 texts): {batch_time:.2f}ms")
        print(f"Speedup: {total_individual_time / batch_time:.2f}x")
        
        # Batch should be at least somewhat faster
        assert batch_time < total_individual_time


# ============================================================================
# Concurrent Request Tests
# ============================================================================


class TestConcurrentRequests:
    """Test suite for concurrent request handling."""
    
    def test_concurrent_predict_requests(self, client):
        """Test handling multiple concurrent /predict requests."""
        def make_request(text):
            request_data = {
                "text": text,
                "temperature": 1.0
            }
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()
            return response.status_code, (end_time - start_time) * 1000
        
        # Make 5 concurrent requests
        texts = [f"test text {i}" for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, text) for text in texts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(status == 200 for status, _ in results)
        
        # All requests should complete in reasonable time
        response_times = [time_ms for _, time_ms in results]
        max_response_time = max(response_times)
        avg_response_time = sum(response_times) / len(response_times)
        
        assert max_response_time < 1000, f"Max concurrent response time {max_response_time:.2f}ms exceeds 1000ms"
        print(f"\nConcurrent requests (5x):")
        print(f"  Max response time: {max_response_time:.2f}ms")
        print(f"  Avg response time: {avg_response_time:.2f}ms")
    
    def test_concurrent_mixed_requests(self, client):
        """Test handling concurrent requests to different endpoints."""
        def make_predict_request():
            response = client.post("/predict", json={"text": "test", "temperature": 1.0})
            return response.status_code
        
        def make_top_k_request():
            response = client.post("/predict/top-k", json={"text": "test", "k": 3, "temperature": 1.0})
            return response.status_code
        
        def make_health_request():
            response = client.get("/health")
            return response.status_code
        
        # Make concurrent requests to different endpoints
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(make_predict_request),
                executor.submit(make_predict_request),
                executor.submit(make_top_k_request),
                executor.submit(make_top_k_request),
                executor.submit(make_health_request),
                executor.submit(make_health_request),
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        print(f"\nConcurrent mixed requests (6x): All succeeded")


# ============================================================================
# Throughput Tests
# ============================================================================


class TestThroughput:
    """Test suite for API throughput."""
    
    def test_requests_per_second(self, client):
        """Test API can handle multiple requests per second."""
        num_requests = 20
        request_data = {
            "text": "machine learning",
            "temperature": 1.0
        }
        
        start_time = time.time()
        
        for _ in range(num_requests):
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        requests_per_second = num_requests / total_time
        
        print(f"\nThroughput: {requests_per_second:.2f} requests/second")
        
        # Should handle at least 10 requests per second
        assert requests_per_second >= 10, f"Throughput {requests_per_second:.2f} req/s is below 10 req/s minimum"


# ============================================================================
# Stress Tests
# ============================================================================


class TestStress:
    """Test suite for stress testing."""
    
    def test_sustained_load(self, client):
        """Test API under sustained load (50 requests)."""
        num_requests = 50
        request_data = {
            "text": "test text",
            "temperature": 1.0
        }
        
        response_times = []
        failures = 0
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = client.post("/predict", json=request_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append((end_time - start_time) * 1000)
                else:
                    failures += 1
            except Exception:
                failures += 1
        
        success_rate = (num_requests - failures) / num_requests * 100
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        print(f"\nSustained load (50 requests):")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Avg response time: {avg_response_time:.2f}ms")
        print(f"  Failures: {failures}")
        
        # Should have high success rate
        assert success_rate >= 95, f"Success rate {success_rate:.1f}% is below 95% minimum"
        
        # Average response time should still be reasonable
        if response_times:
            assert avg_response_time < 500, f"Average response time {avg_response_time:.2f}ms exceeds 500ms under load"
