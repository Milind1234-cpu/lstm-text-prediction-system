"""Pydantic request and response models for FastAPI endpoints.

This module defines all request and response schemas with validation
for the LSTM text prediction API endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any

from ..utils.config import (
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_STOP_WORDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    MAX_BATCH_SIZE,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MIN_TEMPERATURE,
)


# ============================================================================
# Request Models
# ============================================================================


class PredictRequest(BaseModel):
    """Request model for single next word prediction.
    
    Attributes:
        text: Input text for prediction
        temperature: Sampling temperature controlling randomness (0.1-2.0)
    """
    
    text: str = Field(
        ...,
        description="Input text for prediction",
        min_length=1,
        examples=["machine learning is"]
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature (0.1-2.0). Lower = more confident, higher = more random",
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class TopKRequest(BaseModel):
    """Request model for top-k predictions with probabilities.
    
    Attributes:
        text: Input text for prediction
        temperature: Sampling temperature controlling randomness (0.1-2.0)
        k: Number of top predictions to return (1-50)
    """
    
    text: str = Field(
        ...,
        description="Input text for prediction",
        min_length=1,
        examples=["neural networks are"]
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature (0.1-2.0). Lower = more confident, higher = more random",
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
    )
    k: int = Field(
        default=DEFAULT_TOP_K,
        description="Number of top predictions to return (1-50)",
        ge=1,
        le=MAX_TOP_K,
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class BatchRequest(BaseModel):
    """Request model for batch prediction processing.
    
    Attributes:
        texts: List of input texts for prediction
        temperature: Sampling temperature applied to all texts (0.1-2.0)
    """
    
    texts: list[str] = Field(
        ...,
        description="List of input texts for batch prediction",
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        examples=[["deep learning", "artificial intelligence"]]
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature (0.1-2.0). Lower = more confident, higher = more random",
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that all texts are not empty or whitespace only."""
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")
        
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}")
        
        return v


class CompleteRequest(BaseModel):
    """Request model for text completion with stop words.
    
    Attributes:
        text: Input text to complete
        temperature: Sampling temperature controlling randomness (0.1-2.0)
        stop_words: List of words that stop generation (optional)
        max_length: Maximum number of words to generate
    """
    
    text: str = Field(
        ...,
        description="Input text to complete",
        min_length=1,
        examples=["the future of artificial intelligence"]
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature (0.1-2.0). Lower = more confident, higher = more random",
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
    )
    stop_words: list[str] | None = Field(
        default=None,
        description="List of words that stop generation. Defaults to ['.', '?', '!', '\\n']",
        examples=[['.', '?', '!']]
    )
    max_length: int = Field(
        default=DEFAULT_MAX_COMPLETION_LENGTH,
        description="Maximum number of words to generate",
        ge=1,
        le=200,
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


# ============================================================================
# Response Models
# ============================================================================


class PredictResponse(BaseModel):
    """Response model for single next word prediction.
    
    Attributes:
        prediction: The predicted next word
        input_text: The original input text
    """
    
    prediction: str = Field(
        ...,
        description="The predicted next word",
        examples=["powerful"]
    )
    input_text: str = Field(
        ...,
        description="The original input text",
        examples=["machine learning is"]
    )


class PredictionItem(BaseModel):
    """Single prediction item with word and probability.
    
    Attributes:
        word: The predicted word
        probability: Probability score for this prediction (0.0-1.0)
    """
    
    word: str = Field(
        ...,
        description="The predicted word",
        examples=["powerful"]
    )
    probability: float = Field(
        ...,
        description="Probability score for this prediction (0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.234]
    )


class TopKResponse(BaseModel):
    """Response model for top-k predictions with probabilities.
    
    Attributes:
        predictions: List of prediction items with words and probabilities
        input_text: The original input text
    """
    
    predictions: list[PredictionItem] = Field(
        ...,
        description="List of top-k predictions sorted by probability (descending)",
        examples=[[
            {"word": "powerful", "probability": 0.234},
            {"word": "useful", "probability": 0.156},
            {"word": "important", "probability": 0.123}
        ]]
    )
    input_text: str = Field(
        ...,
        description="The original input text",
        examples=["machine learning is"]
    )


class BatchResponse(BaseModel):
    """Response model for batch prediction processing.
    
    Attributes:
        predictions: List of predicted next words in same order as input
        input_texts: The original input texts
    """
    
    predictions: list[str] = Field(
        ...,
        description="List of predicted next words in same order as input texts",
        examples=[["powerful", "revolutionary"]]
    )
    input_texts: list[str] = Field(
        ...,
        description="The original input texts",
        examples=[["machine learning is", "artificial intelligence is"]]
    )


class CompleteResponse(BaseModel):
    """Response model for text completion.
    
    Attributes:
        completion: The completed text including input and generated words
        input_text: The original input text
        stopped_by: The stop word that ended generation or 'max_length'
    """
    
    completion: str = Field(
        ...,
        description="The completed text including input and generated words",
        examples=["the future of artificial intelligence is bright and promising."]
    )
    input_text: str = Field(
        ...,
        description="The original input text",
        examples=["the future of artificial intelligence"]
    )
    stopped_by: str = Field(
        ...,
        description="The stop word that ended generation or 'max_length'",
        examples=["."]
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint.
    
    Attributes:
        status: Health status ('healthy' or 'unhealthy')
        gpu_available: Whether GPU is available
        gpu_name: Name of GPU device if available
        uptime: API uptime in seconds
    """
    
    status: str = Field(
        ...,
        description="Health status",
        examples=["healthy"]
    )
    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available",
        examples=[True]
    )
    gpu_name: str | None = Field(
        default=None,
        description="Name of GPU device if available",
        examples=["NVIDIA GeForce RTX 3080"]
    )
    uptime: float = Field(
        ...,
        description="API uptime in seconds",
        examples=[3600.5]
    )


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint.
    
    Attributes:
        total_requests: Dictionary mapping endpoint paths to request counts
        avg_response_time: Dictionary mapping endpoint paths to average response times (ms)
        total_predictions: Total number of predictions generated
        errors: Dictionary mapping error types to error counts
    """
    
    total_requests: dict[str, int] = Field(
        ...,
        description="Total requests per endpoint",
        examples=[{"/predict": 150, "/predict/top-k": 75}]
    )
    avg_response_time: dict[str, float] = Field(
        ...,
        description="Average response time per endpoint in milliseconds",
        examples=[{"/predict": 45.2, "/predict/top-k": 52.8}]
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions generated",
        examples=[225]
    )
    errors: dict[str, int] = Field(
        ...,
        description="Error counts by error type",
        examples=[{"validation_error": 5, "internal_error": 1}]
    )


class WelcomeResponse(BaseModel):
    """Response model for root endpoint.
    
    Attributes:
        message: Welcome message
        version: API version
        docs_url: URL to API documentation
    """
    
    message: str = Field(
        ...,
        description="Welcome message",
        examples=["Welcome to LSTM Text Prediction API"]
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0"]
    )
    docs_url: str = Field(
        ...,
        description="URL to API documentation",
        examples=["/docs"]
    )


class VocabularyResponse(BaseModel):
    """Response model for vocabulary search endpoint.
    
    Attributes:
        query: The search query (or None if listing)
        matches: List of matching words with their token indices
        total_matches: Total number of matches found
    """
    
    query: str | None = Field(
        default=None,
        description="The search query (or None if listing)",
        examples=["learn"]
    )
    matches: list[dict[str, Any]] = Field(
        ...,
        description="List of matching words with their token indices",
        examples=[[
            {"word": "learning", "index": 42},
            {"word": "learned", "index": 156}
        ]]
    )
    total_matches: int = Field(
        ...,
        description="Total number of matches found",
        examples=[2]
    )


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint.
    
    Attributes:
        architecture: Model architecture description
        lstm_equations: LSTM mathematical equations
        parameters: Model parameter counts
    """
    
    architecture: dict[str, Any] = Field(
        ...,
        description="Model architecture description",
        examples=[{
            "embedding_dim": 256,
            "bidirectional_lstm_units": 512,
            "unidirectional_lstm_units": 256,
            "dropout_rate": 0.3,
            "vocabulary_size": 10000
        }]
    )
    lstm_equations: dict[str, Any] = Field(
        ...,
        description="LSTM mathematical equations in LaTeX format",
        examples=[{
            "forget_gate": "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)",
            "input_gate": "i_t = σ(W_i · [h_{t-1}, x_t] + b_i)"
        }]
    )
    parameters: dict[str, int] = Field(
        ...,
        description="Model parameter counts",
        examples=[{
            "total_params": 15234560,
            "trainable_params": 15234560,
            "non_trainable_params": 0
        }]
    )


class ErrorResponse(BaseModel):
    """Response model for error responses.
    
    Attributes:
        detail: Error message
        error_type: Type of error
    """
    
    detail: str = Field(
        ...,
        description="Error message",
        examples=["Temperature must be between 0.1 and 2.0"]
    )
    error_type: str = Field(
        ...,
        description="Type of error",
        examples=["validation_error"]
    )
