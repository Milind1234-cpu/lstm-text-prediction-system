"""Request logging middleware for FastAPI.

This middleware logs all incoming requests and outgoing responses with
structured logging including method, path, status code, and processing time.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logger import setup_logger

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Logging Middleware
# ============================================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.
    
    Logs request method, path, timestamp, response status code,
    and processing time for all API requests.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and log details.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
        
        Returns:
            HTTP response
        """
        # Record start time
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Log request body for POST endpoints
        if request.method == "POST":
            try:
                # Read body without consuming it
                body = await request.body()
                if body:
                    # Limit body logging to first 500 characters
                    body_str = body.decode('utf-8')[:500]
                    logger.debug(f"Request body: {body_str}")
            except Exception as e:
                logger.warning(f"Could not read request body: {e}")
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {processing_time:.2f}ms"
            )
            raise
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {processing_time:.2f}ms"
        )
        
        return response
