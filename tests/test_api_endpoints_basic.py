"""Basic tests for API endpoints without requiring model loading.

This module tests endpoint structure and basic functionality
without requiring the full model to be loaded.
"""

import pytest
from fastapi.testclient import TestClient


def test_api_routes_exist():
    """Test that all expected routes exist in the app."""
    from src.api import app
    
    # Get all route paths
    routes = [route.path for route in app.routes]
    
    # Check expected routes
    expected_routes = [
        "/",
        "/health",
        "/metrics",
        "/predict",
        "/predict/top-k",
        "/predict/batch",
        "/predict/complete",
        "/model/info",
        "/model/vocabulary",
    ]
    
    for expected_route in expected_routes:
        assert expected_route in routes, f"Route {expected_route} not found"


def test_openapi_schema():
    """Test that OpenAPI schema is generated correctly."""
    from src.api import app
    
    # Get OpenAPI schema
    schema = app.openapi()
    
    # Check basic schema structure
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    
    # Check API info
    assert schema["info"]["title"] == "LSTM Text Prediction API"
    assert schema["info"]["version"] == "1.0.0"
    
    # Check paths exist
    paths = schema["paths"]
    assert "/" in paths
    assert "/health" in paths
    assert "/metrics" in paths
    assert "/predict" in paths
    assert "/predict/top-k" in paths
    assert "/predict/batch" in paths
    assert "/predict/complete" in paths
    assert "/model/info" in paths
    assert "/model/vocabulary" in paths


def test_request_models_in_schema():
    """Test that request models are properly defined in OpenAPI schema."""
    from src.api import app
    
    schema = app.openapi()
    components = schema.get("components", {})
    schemas = components.get("schemas", {})
    
    # Check request models exist
    assert "PredictRequest" in schemas
    assert "TopKRequest" in schemas
    assert "BatchRequest" in schemas
    assert "CompleteRequest" in schemas
    
    # Check PredictRequest structure
    predict_request = schemas["PredictRequest"]
    assert "properties" in predict_request
    assert "text" in predict_request["properties"]
    assert "temperature" in predict_request["properties"]
    
    # Check required fields
    assert "required" in predict_request
    assert "text" in predict_request["required"]


def test_response_models_in_schema():
    """Test that response models are properly defined in OpenAPI schema."""
    from src.api import app
    
    schema = app.openapi()
    components = schema.get("components", {})
    schemas = components.get("schemas", {})
    
    # Check response models exist
    assert "PredictResponse" in schemas
    assert "TopKResponse" in schemas
    assert "BatchResponse" in schemas
    assert "CompleteResponse" in schemas
    assert "HealthResponse" in schemas
    assert "MetricsResponse" in schemas
    assert "WelcomeResponse" in schemas
    assert "VocabularyResponse" in schemas
    assert "ModelInfoResponse" in schemas


def test_middleware_registered():
    """Test that middleware is properly registered."""
    from src.api import app
    
    # Check middleware count
    assert len(app.user_middleware) > 0
    
    # Get middleware class names
    middleware_names = [m.cls.__name__ for m in app.user_middleware]
    
    # Check expected middleware
    assert "CORSMiddleware" in middleware_names
    assert "TimingMiddleware" in middleware_names
    assert "RequestLoggingMiddleware" in middleware_names


def test_cors_configuration():
    """Test that CORS is configured correctly."""
    from src.api import app
    
    # Find CORS middleware
    cors_middleware = None
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            cors_middleware = middleware
            break
    
    assert cors_middleware is not None, "CORS middleware not found"
    
    # Check CORS options
    options = cors_middleware.options
    assert options["allow_origins"] == ["*"]
    assert options["allow_methods"] == ["*"]
    assert options["allow_headers"] == ["*"]


def test_endpoint_tags():
    """Test that endpoints have proper tags for organization."""
    from src.api import app
    
    schema = app.openapi()
    paths = schema["paths"]
    
    # Check prediction endpoints have "Prediction" tag
    predict_endpoint = paths["/predict"]["post"]
    assert "tags" in predict_endpoint
    assert "Prediction" in predict_endpoint["tags"]
    
    # Check model info endpoints have "Model Information" tag
    model_info_endpoint = paths["/model/info"]["get"]
    assert "tags" in model_info_endpoint
    assert "Model Information" in model_info_endpoint["tags"]
    
    # Check health endpoints have "Health & Metrics" tag
    health_endpoint = paths["/health"]["get"]
    assert "tags" in health_endpoint
    assert "Health & Metrics" in health_endpoint["tags"]


def test_endpoint_summaries():
    """Test that endpoints have descriptive summaries."""
    from src.api import app
    
    schema = app.openapi()
    paths = schema["paths"]
    
    # Check summaries exist
    assert "summary" in paths["/"]["get"]
    assert "summary" in paths["/health"]["get"]
    assert "summary" in paths["/metrics"]["get"]
    assert "summary" in paths["/predict"]["post"]
    assert "summary" in paths["/predict/top-k"]["post"]
    assert "summary" in paths["/predict/batch"]["post"]
    assert "summary" in paths["/predict/complete"]["post"]
    assert "summary" in paths["/model/info"]["get"]
    assert "summary" in paths["/model/vocabulary"]["get"]


def test_endpoint_descriptions():
    """Test that endpoints have detailed descriptions."""
    from src.api import app
    
    schema = app.openapi()
    paths = schema["paths"]
    
    # Check descriptions exist
    assert "description" in paths["/"]["get"]
    assert "description" in paths["/health"]["get"]
    assert "description" in paths["/predict"]["post"]
    assert "description" in paths["/predict/top-k"]["post"]


def test_endpoint_response_codes():
    """Test that endpoints define proper response codes."""
    from src.api import app
    
    schema = app.openapi()
    paths = schema["paths"]
    
    # Check /predict endpoint responses
    predict_responses = paths["/predict"]["post"]["responses"]
    assert "200" in predict_responses
    assert "400" in predict_responses
    assert "422" in predict_responses
    assert "503" in predict_responses
    
    # Check /health endpoint responses
    health_responses = paths["/health"]["get"]["responses"]
    assert "200" in health_responses
    assert "503" in health_responses
