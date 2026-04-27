"""Main FastAPI application for LSTM text prediction API.

This module creates and configures the FastAPI application with:
- CORS middleware
- Request logging and timing middleware
- All endpoint routers
- Startup and shutdown events
- Model loading
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..model.predictor import Predictor
from ..utils.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    MODEL_ARCHITECTURE_PATH,
    MODEL_WEIGHTS_PATH,
    TOKENIZER_CONFIG_PATH,
)
from ..utils.logger import print_error, print_success, setup_logger
from .endpoints import health, model_info, prediction
from .middleware.logging import RequestLoggingMiddleware
from .middleware.timing import TimingMiddleware

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Global Predictor Instance
# ============================================================================

predictor: Predictor | None = None


# ============================================================================
# Lifespan Context Manager
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    
    Yields:
        None
    """
    # Startup
    logger.info("Starting LSTM Text Prediction API")
    
    global predictor
    
    try:
        # Create predictor instance
        predictor = Predictor()
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        predictor.load_model(
            architecture_path=MODEL_ARCHITECTURE_PATH,
            weights_path=MODEL_WEIGHTS_PATH,
            tokenizer_path=TOKENIZER_CONFIG_PATH
        )
        
        # Set predictor for all endpoints
        prediction.set_predictor(predictor)
        model_info.set_predictor(predictor)
        health.set_predictor(predictor)
        
        # Reset startup time for uptime tracking
        health.reset_startup_time()
        
        print_success(
            f"API started successfully!\n"
            f"Version: {API_VERSION}\n"
            f"Model loaded and ready for predictions",
            title="API Started"
        )
        
        logger.info("API startup complete")
    
    except FileNotFoundError as e:
        error_msg = (
            f"Model files not found: {e}\n\n"
            f"Please ensure the following files exist:\n"
            f"  - {MODEL_ARCHITECTURE_PATH}\n"
            f"  - {MODEL_WEIGHTS_PATH}\n"
            f"  - {TOKENIZER_CONFIG_PATH}\n\n"
            f"Run the training script first to generate these files."
        )
        logger.error(error_msg)
        print_error(error_msg, title="Model Files Not Found")
        sys.exit(1)
    
    except Exception as e:
        error_msg = f"Failed to start API: {e}"
        logger.error(error_msg)
        print_error(error_msg, title="API Startup Failed")
        sys.exit(1)
    
    # Yield control to the application
    yield
    
    # Shutdown
    logger.info("Shutting down LSTM Text Prediction API")
    print_success("API shutdown complete", title="API Shutdown")


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

logger.info("CORS middleware configured")


# ============================================================================
# Custom Middleware
# ============================================================================

# Add timing middleware (should be before logging to track total time)
app.add_middleware(TimingMiddleware)
logger.info("Timing middleware configured")

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)
logger.info("Request logging middleware configured")


# ============================================================================
# Routers
# ============================================================================

# Health and metrics endpoints (no prefix)
app.include_router(health.router)
logger.info("Health router registered")

# Prediction endpoints (/predict)
app.include_router(prediction.router)
logger.info("Prediction router registered")

# Model info endpoints (/model)
app.include_router(model_info.router)
logger.info("Model info router registered")


# ============================================================================
# Application Ready
# ============================================================================

logger.info("FastAPI application configured and ready")
