"""Response time tracking middleware for FastAPI.

This middleware tracks response times per endpoint and aggregates
metrics for average response time calculation.
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logger import setup_logger

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Metrics Storage
# ============================================================================


class MetricsTracker:
    """Tracks API metrics including response times and request counts.
    
    Attributes:
        request_counts: Dictionary mapping endpoint paths to request counts
        response_times: Dictionary mapping endpoint paths to lists of response times
        total_predictions: Total number of predictions generated
        error_counts: Dictionary mapping error types to error counts
    """
    
    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.request_counts: dict[str, int] = defaultdict(int)
        self.response_times: dict[str, list[float]] = defaultdict(list)
        self.total_predictions: int = 0
        self.error_counts: dict[str, int] = defaultdict(int)
    
    def record_request(self, path: str, response_time: float) -> None:
        """Record a request with its response time.
        
        Args:
            path: Request path
            response_time: Response time in milliseconds
        """
        self.request_counts[path] += 1
        self.response_times[path].append(response_time)
    
    def record_prediction(self, count: int = 1) -> None:
        """Record prediction(s) generated.
        
        Args:
            count: Number of predictions generated (default: 1)
        """
        self.total_predictions += count
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence.
        
        Args:
            error_type: Type of error (e.g., 'validation_error', 'internal_error')
        """
        self.error_counts[error_type] += 1
    
    def get_avg_response_times(self) -> dict[str, float]:
        """Get average response times per endpoint.
        
        Returns:
            Dictionary mapping endpoint paths to average response times in milliseconds
        """
        avg_times = {}
        for path, times in self.response_times.items():
            if times:
                avg_times[path] = sum(times) / len(times)
            else:
                avg_times[path] = 0.0
        return avg_times
    
    def get_metrics(self) -> dict:
        """Get all tracked metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        return {
            'total_requests': dict(self.request_counts),
            'avg_response_time': self.get_avg_response_times(),
            'total_predictions': self.total_predictions,
            'errors': dict(self.error_counts),
        }


# Global metrics tracker instance
metrics_tracker = MetricsTracker()


# ============================================================================
# Timing Middleware
# ============================================================================


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking response times per endpoint.
    
    Records response time for each request and aggregates metrics
    for average response time calculation.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and track response time.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
        
        Returns:
            HTTP response with X-Process-Time header
        """
        # Record start time
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time in milliseconds
            processing_time = (time.time() - start_time) * 1000
            
            # Record metrics
            metrics_tracker.record_request(request.url.path, processing_time)
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{processing_time:.2f}ms"
            
            return response
        
        except Exception as e:
            # Record error
            processing_time = (time.time() - start_time) * 1000
            metrics_tracker.record_request(request.url.path, processing_time)
            
            # Determine error type
            error_type = type(e).__name__
            metrics_tracker.record_error(error_type)
            
            raise


# ============================================================================
# Metrics Access Functions
# ============================================================================


def get_metrics_tracker() -> MetricsTracker:
    """Get the global metrics tracker instance.
    
    Returns:
        Global MetricsTracker instance
    """
    return metrics_tracker
