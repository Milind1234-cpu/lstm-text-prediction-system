"""Model information endpoints for LSTM text prediction API.

This module provides REST endpoints for:
- Model architecture and LSTM mathematics
- Vocabulary search functionality
"""

from fastapi import APIRouter, HTTPException, Query, status

from ...data.tokenizer import Tokenizer
from ...model.predictor import Predictor
from ...utils.config import (
    BIDIRECTIONAL_LSTM_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    MAX_VOCABULARY_SEARCH_RESULTS,
    UNIDIRECTIONAL_LSTM_UNITS,
    VOCABULARY_SIZE,
)
from ...utils.logger import setup_logger
from ..models import ModelInfoResponse, VocabularyResponse

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/model", tags=["Model Information"])


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
    logger.info("Predictor set for model info endpoints")


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
# Model Information Endpoints
# ============================================================================


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get model architecture details and LSTM mathematical equations",
    responses={
        200: {
            "description": "Model information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "architecture": {
                            "embedding_dim": 256,
                            "bidirectional_lstm_units": 512,
                            "unidirectional_lstm_units": 256,
                            "dropout_rate": 0.3,
                            "vocabulary_size": 10000
                        },
                        "lstm_equations": {
                            "forget_gate": "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)",
                            "input_gate": "i_t = σ(W_i · [h_{t-1}, x_t] + b_i)"
                        },
                        "parameters": {
                            "total_params": 15234560,
                            "trainable_params": 15234560,
                            "non_trainable_params": 0
                        }
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def get_model_info() -> ModelInfoResponse:
    """Get model architecture and LSTM mathematical equations.
    
    Returns:
        Model information including architecture, LSTM equations, and parameters
    
    Raises:
        HTTPException: If model is not loaded
    """
    try:
        predictor = get_predictor()
        
        logger.info("Retrieving model information")
        
        # Architecture information
        architecture = {
            "embedding_dim": EMBEDDING_DIM,
            "bidirectional_lstm_units": BIDIRECTIONAL_LSTM_UNITS,
            "unidirectional_lstm_units": UNIDIRECTIONAL_LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "vocabulary_size": VOCABULARY_SIZE,
            "sequence_length": predictor.sequence_length,
        }
        
        # LSTM mathematical equations in LaTeX format
        lstm_equations = {
            "forget_gate": r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)",
            "forget_gate_description": "Controls what information to discard from cell state",
            
            "input_gate": r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)",
            "input_gate_description": "Controls what new information to store in cell state",
            
            "candidate_cell_state": r"\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)",
            "candidate_cell_state_description": "Candidate values to add to cell state",
            
            "cell_state_update": r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t",
            "cell_state_update_description": "Update cell state by forgetting and adding new information",
            
            "output_gate": r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)",
            "output_gate_description": "Controls what information to output from cell state",
            
            "hidden_state": r"h_t = o_t \odot \tanh(C_t)",
            "hidden_state_description": "Final hidden state output",
            
            "bidirectional_processing": r"h_t^{bidir} = [\overrightarrow{h_t}, \overleftarrow{h_t}]",
            "bidirectional_processing_description": "Concatenate forward and backward LSTM outputs",
            
            "notation": {
                "sigma": r"\sigma = sigmoid activation function",
                "tanh": r"\tanh = hyperbolic tangent activation function",
                "odot": r"\odot = element-wise multiplication (Hadamard product)",
                "W": "Weight matrices",
                "b": "Bias vectors",
                "h": "Hidden state",
                "C": "Cell state",
                "x": "Input",
            }
        }
        
        # Model parameters (count from loaded model)
        if predictor.model is not None:
            total_params = predictor.model.count_params()
            trainable_params = sum([w.numpy().size for w in predictor.model.trainable_weights])
            non_trainable_params = total_params - trainable_params
        else:
            total_params = 0
            trainable_params = 0
            non_trainable_params = 0
        
        parameters = {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "non_trainable_params": int(non_trainable_params),
        }
        
        logger.info("Model information retrieved successfully")
        
        return ModelInfoResponse(
            architecture=architecture,
            lstm_equations=lstm_equations,
            parameters=parameters
        )
    
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/vocabulary",
    response_model=VocabularyResponse,
    summary="Search vocabulary",
    description="Search the model vocabulary or list words",
    responses={
        200: {
            "description": "Vocabulary search results",
            "content": {
                "application/json": {
                    "example": {
                        "query": "learn",
                        "matches": [
                            {"word": "learning", "index": 42},
                            {"word": "learned", "index": 156}
                        ],
                        "total_matches": 2
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable - model not loaded"
        }
    }
)
async def search_vocabulary(
    query: str | None = Query(
        default=None,
        description="Search query for vocabulary words (case-insensitive). If not provided, returns first 100 words.",
        examples=["learn"]
    )
) -> VocabularyResponse:
    """Search the model vocabulary.
    
    Args:
        query: Optional search query for vocabulary words (case-insensitive).
            If not provided, returns first 100 words.
    
    Returns:
        Vocabulary search results with matching words and their indices
    
    Raises:
        HTTPException: If model is not loaded
    """
    try:
        predictor = get_predictor()
        
        if predictor.tokenizer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tokenizer not loaded. Service unavailable."
            )
        
        tokenizer: Tokenizer = predictor.tokenizer
        
        if query:
            logger.info(f"Searching vocabulary for: '{query}'")
            
            # Case-insensitive search
            query_lower = query.lower()
            matches = []
            
            # Search through vocabulary
            for word, index in tokenizer.word_to_index.items():
                if query_lower in word.lower():
                    matches.append({
                        "word": word,
                        "index": index
                    })
                    
                    # Limit results
                    if len(matches) >= MAX_VOCABULARY_SEARCH_RESULTS:
                        break
            
            # Sort by index
            matches.sort(key=lambda x: int(x['index']))  # type: ignore[call-overload]
            
            logger.info(f"Found {len(matches)} matches for '{query}'")
            
            return VocabularyResponse(
                query=query,
                matches=matches,
                total_matches=len(matches)
            )
        
        else:
            logger.info("Listing first 100 vocabulary words")
            
            # Return first 100 words
            matches = []
            for word, index in sorted(tokenizer.word_to_index.items(), key=lambda x: x[1])[:MAX_VOCABULARY_SEARCH_RESULTS]:
                matches.append({
                    "word": word,
                    "index": index
                })
            
            logger.info(f"Returning {len(matches)} vocabulary words")
            
            return VocabularyResponse(
                query=None,
                matches=matches,
                total_matches=len(matches)
            )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error searching vocabulary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
