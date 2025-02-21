"""Rate limiting utilities for market data providers."""

import time
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Dict, Optional, Union
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter implementation with token bucket algorithm."""
    
    def __init__(
        self,
        max_requests: int,
        time_window: Union[int, float],
        retry_interval: Union[int, float] = 1.0
    ):
        """Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
            retry_interval: Time to wait between retries when rate limited
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.retry_interval = retry_interval
        self.requests = []
        self._lock = Lock()
    
    def _cleanup_old_requests(self) -> None:
        """Remove requests older than the time window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        self.requests = [ts for ts in self.requests if ts > cutoff]
    
    def _wait_if_needed(self) -> None:
        """Wait if we've exceeded the rate limit."""
        while True:
            with self._lock:
                self._cleanup_old_requests()
                if len(self.requests) < self.max_requests:
                    self.requests.append(datetime.now())
                    break
            time.sleep(self.retry_interval)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Callable: Rate limited function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._wait_if_needed()
            return func(*args, **kwargs)
        return wrapper


class BackoffRateLimiter(RateLimiter):
    """Rate limiter with exponential backoff for failures."""
    
    def __init__(
        self,
        max_requests: int,
        time_window: Union[int, float],
        retry_interval: Union[int, float] = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """Initialize the backoff rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
            retry_interval: Initial time to wait between retries
            max_retries: Maximum number of retries before giving up
            backoff_factor: Factor to multiply retry interval by after each failure
        """
        super().__init__(max_requests, time_window, retry_interval)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._retry_counts: Dict[str, int] = {}
        self._backoff_until: Dict[str, datetime] = {}
    
    def _get_retry_key(self, func: Callable, *args, **kwargs) -> str:
        """Generate a unique key for tracking retries.
        
        Args:
            func: The function being called
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: Unique key for this function call
        """
        # Simple key generation - can be made more sophisticated if needed
        return f"{func.__name__}:{args}:{kwargs}"
    
    def _should_backoff(self, key: str) -> bool:
        """Check if we should still be backing off for this key.
        
        Args:
            key: Retry tracking key
            
        Returns:
            bool: True if we should continue backing off
        """
        backoff_time = self._backoff_until.get(key)
        return backoff_time is not None and datetime.now() < backoff_time
    
    def _get_backoff_time(self, retry_count: int) -> float:
        """Calculate backoff time for a given retry count.
        
        Args:
            retry_count: Current retry count
            
        Returns:
            float: Backoff time in seconds
        """
        return self.retry_interval * (self.backoff_factor ** (retry_count - 1))
    
    def _update_backoff(self, key: str, retry_count: int) -> None:
        """Update backoff time after a failure.
        
        Args:
            key: Retry tracking key
            retry_count: Current retry count
        """
        # Calculate backoff time
        backoff_time = self._get_backoff_time(retry_count)
        self._backoff_until[key] = datetime.now() + timedelta(seconds=backoff_time)
        logger.warning(
            f"Backing off {key} for {backoff_time:.1f}s (attempt {retry_count})"
        )
        
        # Sleep with context for testing
        self._backoff_sleep(backoff_time)
    
    def _backoff_sleep(self, duration: float) -> None:
        """Sleep for backoff with context for testing.
        
        Args:
            duration: Sleep duration in seconds
        """
        time.sleep(duration)
    
    def _cleanup_old_retries(self) -> None:
        """Remove old retry tracking data."""
        now = datetime.now()
        expired_keys = [
            key for key, time in self._backoff_until.items()
            if time < now
        ]
        for key in expired_keys:
            self._retry_counts.pop(key, None)
            self._backoff_until.pop(key, None)
    
    def _call_with_retries(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        key: str,
        retry_count: int = 0
    ):
        """Call function with retries.
        
        Args:
            func: Function to call
            args: Function arguments
            kwargs: Function keyword arguments
            key: Retry tracking key
            retry_count: Current retry count
            
        Returns:
            Any: Function result
            
        Raises:
            Exception: If max retries exceeded
        """
        try:
            # Wait for rate limit
            self._wait_if_needed()
            
            # Call function
            result = func(*args, **kwargs)
            
            # Success - reset retry count
            self._retry_counts.pop(key, None)
            self._backoff_until.pop(key, None)
            
            return result
        
        except Exception as e:
            # Check max retries
            if retry_count >= self.max_retries:
                logger.error(
                    f"Max retries ({self.max_retries}) exceeded for {key}"
                )
                raise
            
            # Update retry count
            retry_count += 1
            self._retry_counts[key] = retry_count
            
            # Update backoff and wait
            self._update_backoff(key, retry_count)
            
            # Log and retry
            logger.warning(
                f"Error in {func.__name__}, retrying "
                f"(attempt {retry_count}): {str(e)}"
            )
            
            # Recursive retry
            return self._call_with_retries(func, args, kwargs, key, retry_count)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function with backoff.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Callable: Rate limited function with backoff
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._get_retry_key(func, *args, **kwargs)
            
            # Clean up old retry data
            self._cleanup_old_retries()
            
            return self._call_with_retries(func, args, kwargs, key)
        
        return wrapper 