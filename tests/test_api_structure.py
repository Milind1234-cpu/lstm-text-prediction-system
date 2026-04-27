"""Tests for API structure and imports.

This module tests that all API components can be imported correctly
and that the basic structure is valid.
"""

import pytest


def test_api_models_import():
    """Test that API models can be imported."""
    from src.api.models import (
        BatchRequest,
        BatchResponse,
        CompleteRequest,
        CompleteResponse,
        ErrorResponse,
        HealthResponse,
        MetricsResponse,
        ModelInfoResponse,
        PredictRequest,
        PredictResponse,
        PredictionItem,
        TopKRequest,
        TopKResponse,
        VocabularyResponse,
        WelcomeResponse,
    )
    
    # Verify classes exist
    assert PredictRequest is not None
    assert TopKRequest is not None
    assert BatchRequest is not None
    assert CompleteRequest is not None
    assert PredictResponse is not None
    assert TopKResponse is not None
    assert BatchResponse is not None
    assert CompleteResponse is not None
    assert HealthResponse is not None
    assert MetricsResponse is not None
    assert WelcomeResponse is not None
    assert VocabularyResponse is not None
    assert ModelInfoResponse is not None
    assert ErrorResponse is not None
    assert PredictionItem is not None


def test_middleware_import():
    """Test that middleware can be imported."""
    from src.api.middleware import (
        MetricsTracker,
        RequestLoggingMiddleware,
        TimingMiddleware,
        get_metrics_tracker,
    )
    
    # Verify classes and functions exist
    assert RequestLoggingMiddleware is not None
    assert TimingMiddleware is not None
    assert MetricsTracker is not None
    assert get_metrics_tracker is not None


def test_endpoints_import():
    """Test that endpoint routers can be imported."""
    from src.api.endpoints import health, model_info, prediction
    
    # Verify modules exist
    assert health is not None
    assert model_info is not None
    assert prediction is not None
    
    # Verify routers exist
    assert hasattr(health, 'router')
    assert hasattr(model_info, 'router')
    assert hasattr(prediction, 'router')


def test_app_import():
    """Test that FastAPI app can be imported."""
    from src.api import app
    
    # Verify app exists
    assert app is not None
    
    # Verify app is FastAPI instance
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


def test_pydantic_model_validation():
    """Test that Pydantic models validate correctly."""
    from src.api.models import PredictRequest, TopKRequest
    
    # Test valid request
    request = PredictRequest(text="test text", temperature=1.0)
    assert request.text == "test text"
    assert request.temperature == 1.0
    
    # Test default temperature
    request = PredictRequest(text="test text")
    assert request.temperature == 1.0
    
    # Test invalid temperature (should raise validation error)
    with pytest.raises(Exception):  # Pydantic ValidationError
        PredictRequest(text="test text", temperature=3.0)
    
    # Test empty text (should raise validation error)
    with pytest.raises(Exception):  # Pydantic ValidationError
        PredictRequest(text="", temperature=1.0)
    
    # Test TopKRequest
    request = TopKRequest(text="test text", k=5, temperature=1.0)
    assert request.k == 5
    
    # Test invalid k (should raise validation error)
    with pytest.raises(Exception):  # Pydantic ValidationError
        TopKRequest(text="test text", k=100, temperature=1.0)


def test_metrics_tracker():
    """Test that MetricsTracker works correctly."""
    from src.api.middleware.timing import MetricsTracker
    
    tracker = MetricsTracker()
    
    # Test recording requests
    tracker.record_request("/predict", 45.2)
    tracker.record_request("/predict", 50.8)
    tracker.record_request("/predict/top-k", 60.5)
    
    # Test request counts
    assert tracker.request_counts["/predict"] == 2
    assert tracker.request_counts["/predict/top-k"] == 1
    
    # Test average response times
    avg_times = tracker.get_avg_response_times()
    assert avg_times["/predict"] == pytest.approx((45.2 + 50.8) / 2)
    assert avg_times["/predict/top-k"] == pytest.approx(60.5)
    
    # Test recording predictions
    tracker.record_prediction(5)
    assert tracker.total_predictions == 5
    
    # Test recording errors
    tracker.record_error("validation_error")
    tracker.record_error("validation_error")
    tracker.record_error("internal_error")
    assert tracker.error_counts["validation_error"] == 2
    assert tracker.error_counts["internal_error"] == 1
    
    # Test get_metrics
    metrics = tracker.get_metrics()
    assert "total_requests" in metrics
    assert "avg_response_time" in metrics
    assert "total_predictions" in metrics
    assert "errors" in metrics
