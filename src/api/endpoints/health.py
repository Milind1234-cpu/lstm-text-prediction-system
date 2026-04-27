"""Health and metrics endpoints for LSTM text prediction API.

This module provides REST endpoints for:
- Root welcome message
- Health check with GPU status
- API usage metrics
"""

import time

import tensorflow as tf
from fastapi import APIRouter, HTTPException, status

from ...model.predictor import Predictor
from ...utils.config import API_VERSION
from ...utils.logger import setup_logger
from ..middleware.timing import get_metrics_tracker
from ..models import HealthResponse, MetricsResponse, WelcomeResponse

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(tags=["Health & Metrics"])


# ============================================================================
# Predictor Instance and Startup Time
# ============================================================================

_predictor: Predictor | None = None
_startup_time: float = time.time()


def set_predictor(predictor: Predictor) -> None:
    """Set the predictor instance for endpoints.
    
    Args:
        predictor: Initialized Predictor instance with loaded model
    """
    global _predictor
    _predictor = predictor
    logger.info("Predictor set for health endpoints")


def get_predictor() -> Predictor:
    """Get the predictor instance.
    
    Returns:
        Predictor instance
    
    Raises:
        HTTPException: If predictor is not initialized
    """
    if _predictor is None:
        logger.error("Predictor not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )
    return _predictor


def reset_startup_time() -> None:
    """Reset the startup time to current time.
    
    This should be called when the API server starts.
    """
    global _startup_time
    _startup_time = time.time()
    logger.info("Startup time reset")


# ============================================================================
# Health and Metrics Endpoints
# ============================================================================


@router.get(
    "/",
    response_model=WelcomeResponse,
    summary="Welcome message",
    description="Get welcome message and API version",
    responses={
        200: {
            "description": "Welcome message",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Welcome to LSTM Text Prediction API",
                        "version": "1.0.0",
                        "docs_url": "/docs"
                    }
                }
            }
        }
    }
)
async def root() -> WelcomeResponse:
    """Get welcome message and API version.
    
    Returns:
        Welcome response with message, version, and docs URL
    """
    logger.info("Root endpoint accessed")
    
    return WelcomeResponse(
        message="Welcome to LSTM Text Prediction API",
        version=API_VERSION,
        docs_url="/docs"
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and GPU availability",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "gpu_available": True,
                        "gpu_name": "NVIDIA GeForce RTX 3080",
                        "uptime": 3600.5
                    }
                }
            }
        },
        503: {
            "description": "Service is unhealthy - model not loaded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "gpu_available": False,
                        "gpu_name": None,
                        "uptime": 120.3
                    }
                }
            }
        }
    }
)
async def health_check() -> HealthResponse:
    """Check API health status and GPU availability.
    
    Returns:
        Health response with status, GPU info, and uptime
    
    Raises:
        HTTPException: If service is unhealthy (model not loaded)
    """
    logger.info("Health check requested")
    
    # Calculate uptime
    uptime = time.time() - _startup_time
    
    # Check GPU availability
    gpu_available = False
    gpu_name = None
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_available = True
            # Get GPU name from first GPU
            gpu_name = gpus[0].name
            # Try to get more detailed name
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                if 'device_name' in gpu_details:
                    gpu_name = gpu_details['device_name']
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not check GPU availability: {e}")
    
    # Check if model is loaded
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            logger.warning("Model not loaded - service unhealthy")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        logger.info("Health check passed - service healthy")
        
        return HealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            uptime=uptime
        )
    
    except HTTPException:
        # Model not loaded - return unhealthy status
        logger.error("Health check failed - model not loaded")
        
        return HealthResponse(
            status="unhealthy",
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            uptime=uptime
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get API metrics",
    description="Get API usage statistics including request counts, response times, and errors",
    responses={
        200: {
            "description": "API metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "total_requests": {
                            "/predict": 150,
                            "/predict/top-k": 75
                        },
                        "avg_response_time": {
                            "/predict": 45.2,
                            "/predict/top-k": 52.8
                        },
                        "total_predictions": 225,
                        "errors": {
                            "validation_error": 5,
                            "internal_error": 1
                        }
                    }
                }
            }
        }
    }
)
async def get_metrics() -> MetricsResponse:
    """Get API usage statistics.
    
    Returns:
        Metrics response with request counts, response times, predictions, and errors
    """
    logger.info("Metrics requested")
    
    # Get metrics from tracker
    metrics_tracker = get_metrics_tracker()
    metrics = metrics_tracker.get_metrics()
    
    logger.info(f"Returning metrics: {metrics['total_predictions']} total predictions")
    
    return MetricsResponse(
        total_requests=metrics['total_requests'],
        avg_response_time=metrics['avg_response_time'],
        total_predictions=metrics['total_predictions'],
        errors=metrics['errors']
    )
