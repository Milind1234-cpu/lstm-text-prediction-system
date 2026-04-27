"""Prediction endpoints for LSTM text prediction API.

This module provides REST endpoints for various prediction modes:
- Single next word prediction
- Top-k predictions with probabilities
- Batch prediction processing
- Text completion with stop words
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from ...model.predictor import Predictor
from ...utils.logger import setup_logger
from ..middleware.timing import get_metrics_tracker
from ..models import (
    BatchRequest,
    BatchResponse,
    CompleteRequest,
    CompleteResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    TopKRequest,
    TopKResponse,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ============================================================================
# Predictor Instance (will be set by app startup)
# ============================================================================

_predictor: Predictor | None = None


def set_predictor(predictor: Predictor) -> None:
    """Set the predictor instance for endpoints.
    
    Args:
        predictor: Initialized Predictor instance with loaded model
    """
    global _predictor
    _predictor = predictor
    logger.info("Predictor set for prediction endpoints")


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


# ============================================================================
# Prediction Endpoints
# ============================================================================


@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict next word",
    description="Predict the single most likely next word given input text",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "powerful",
                        "input_text": "machine learning is"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Temperature must be between 0.1 and 2.0"
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Text cannot be empty or whitespace only"
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def predict_next_word(request: PredictRequest) -> PredictResponse:
    """Predict the single most likely next word.
    
    Args:
        request: Prediction request with text and temperature
    
    Returns:
        Prediction response with predicted word and input text
    
    Raises:
        HTTPException: If validation fails or prediction error occurs
    """
    try:
        predictor = get_predictor()
        
        logger.info(f"Predicting next word for: '{request.text[:50]}...'")
        
        # Generate prediction
        prediction = predictor.predict_next_word(
            text=request.text,
            temperature=request.temperature
        )
        
        # Record metrics
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_prediction(1)
        
        logger.info(f"Prediction successful: '{prediction}'")
        
        return PredictResponse(
            prediction=prediction,
            input_text=request.text
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("validation_error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/top-k",
    response_model=TopKResponse,
    summary="Predict top-k words",
    description="Predict the top-k most likely next words with probabilities",
    responses={
        200: {
            "description": "Successful top-k prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {"word": "powerful", "probability": 0.234},
                            {"word": "useful", "probability": 0.156},
                            {"word": "important", "probability": 0.123}
                        ],
                        "input_text": "machine learning is"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters"
        },
        422: {
            "description": "Validation error"
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def predict_top_k(request: TopKRequest) -> TopKResponse:
    """Predict top-k most likely next words with probabilities.
    
    Args:
        request: Top-k prediction request with text, temperature, and k
    
    Returns:
        Top-k prediction response with predictions and input text
    
    Raises:
        HTTPException: If validation fails or prediction error occurs
    """
    try:
        predictor = get_predictor()
        
        logger.info(f"Predicting top-{request.k} words for: '{request.text[:50]}...'")
        
        # Generate predictions
        predictions = predictor.predict_top_k(
            text=request.text,
            k=request.k,
            temperature=request.temperature
        )
        
        # Convert to Pydantic models
        prediction_items = [
            PredictionItem(
                word=str(p['word']),
                probability=float(p['probability'])
            )
            for p in predictions
        ]
        
        # Record metrics
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_prediction(len(predictions))
        
        logger.info(f"Top-{request.k} prediction successful")
        
        return TopKResponse(
            predictions=prediction_items,
            input_text=request.text
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("validation_error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Top-k prediction error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchResponse,
    summary="Batch predict",
    description="Predict next words for multiple texts in a single request",
    responses={
        200: {
            "description": "Successful batch prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": ["powerful", "revolutionary"],
                        "input_texts": ["machine learning is", "artificial intelligence is"]
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters or batch size exceeded"
        },
        422: {
            "description": "Validation error"
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predict next words for multiple texts in batch.
    
    Args:
        request: Batch prediction request with texts and temperature
    
    Returns:
        Batch prediction response with predictions and input texts
    
    Raises:
        HTTPException: If validation fails or prediction error occurs
    """
    try:
        predictor = get_predictor()
        
        logger.info(f"Batch predicting for {len(request.texts)} texts")
        
        # Generate predictions
        predictions = predictor.predict_batch(
            texts=request.texts,
            temperature=request.temperature
        )
        
        # Record metrics
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_prediction(len(predictions))
        
        logger.info(f"Batch prediction successful: {len(predictions)} predictions")
        
        return BatchResponse(
            predictions=predictions,
            input_texts=request.texts
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("validation_error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/complete",
    response_model=CompleteResponse,
    summary="Complete text",
    description="Generate text completion until stop words or max length",
    responses={
        200: {
            "description": "Successful text completion",
            "content": {
                "application/json": {
                    "example": {
                        "completion": "the future of artificial intelligence is bright and promising.",
                        "input_text": "the future of artificial intelligence",
                        "stopped_by": "."
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters"
        },
        422: {
            "description": "Validation error"
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def complete_text(request: CompleteRequest) -> CompleteResponse:
    """Generate text completion until stop words or max length.
    
    Args:
        request: Text completion request with text, temperature, stop_words, and max_length
    
    Returns:
        Text completion response with completed text and stop reason
    
    Raises:
        HTTPException: If validation fails or prediction error occurs
    """
    try:
        predictor = get_predictor()
        
        logger.info(f"Completing text: '{request.text[:50]}...'")
        
        # Generate completion
        completion = predictor.complete_text(
            text=request.text,
            max_length=request.max_length,
            stop_words=request.stop_words,
            temperature=request.temperature
        )
        
        # Determine what stopped generation
        generated_words = completion[len(request.text):].strip().split()
        if generated_words:
            last_word = generated_words[-1]
            # Check if last word is a stop word
            stop_words = request.stop_words if request.stop_words else ['.', '?', '!', '\n']
            if last_word in stop_words:
                stopped_by = last_word
            else:
                stopped_by = "max_length"
        else:
            stopped_by = "max_length"
        
        # Record metrics
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_prediction(len(generated_words))
        
        logger.info(f"Text completion successful: {len(generated_words)} words generated")
        
        return CompleteResponse(
            completion=completion,
            input_text=request.text,
            stopped_by=stopped_by
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("validation_error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Text completion error: {e}")
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.record_error("internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
