"""Tests for rate limiting utilities."""

import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.marketdata.utils.rate_limit import RateLimiter, BackoffRateLimiter


def test_rate_limiter_basic():
    """Test basic rate limiting functionality."""
    # Allow 2 requests per second
    limiter = RateLimiter(max_requests=2, time_window=1.0)
    
    # Create a mock function to decorate
    mock_func = Mock(return_value="test")
    mock_func.__name__ = "mock_func"  # Add name attribute
    limited_func = limiter(mock_func)
    
    # First two calls should be immediate
    start_time = time.time()
    assert limited_func() == "test"
    assert limited_func() == "test"
    elapsed = time.time() - start_time
    assert elapsed < 0.1  # Should be nearly instant
    
    # Third call should wait
    start_time = time.time()
    assert limited_func() == "test"
    elapsed = time.time() - start_time
    assert elapsed >= 1.0  # Should wait for the full window


def test_rate_limiter_cleanup():
    """Test that old requests are cleaned up."""
    limiter = RateLimiter(max_requests=1, time_window=0.1)
    
    # Add an old request
    old_time = datetime.now() - timedelta(seconds=1)
    limiter.requests.append(old_time)
    
    # Cleanup should remove it
    limiter._cleanup_old_requests()
    assert len(limiter.requests) == 0


def test_backoff_rate_limiter_success():
    """Test backoff rate limiter with successful calls."""
    limiter = BackoffRateLimiter(
        max_requests=2,
        time_window=1.0,
        retry_interval=0.1
    )
    
    # Create a mock function that succeeds
    mock_func = Mock(return_value="test")
    mock_func.__name__ = "mock_func"  # Add name attribute
    limited_func = limiter(mock_func)
    
    # Multiple successful calls should not trigger backoff
    for _ in range(3):
        assert limited_func() == "test"
    
    assert len(limiter._retry_counts) == 0
    assert len(limiter._backoff_until) == 0


def test_backoff_rate_limiter_retry():
    """Test backoff rate limiter retry behavior."""
    limiter = BackoffRateLimiter(
        max_requests=2,
        time_window=1.0,
        retry_interval=0.1,
        max_retries=2,
        backoff_factor=2.0
    )
    
    # Create a mock function that fails twice then succeeds
    mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
    mock_func.__name__ = "mock_func"  # Add name attribute
    limited_func = limiter(mock_func)
    
    # Should eventually succeed after retries
    assert limited_func() == "success"
    assert mock_func.call_count == 3
    
    # Retry data should be cleared after success
    assert len(limiter._retry_counts) == 0
    assert len(limiter._backoff_until) == 0


def test_backoff_rate_limiter_max_retries():
    """Test backoff rate limiter max retries behavior."""
    limiter = BackoffRateLimiter(
        max_requests=2,
        time_window=1.0,
        retry_interval=0.1,
        max_retries=2
    )
    
    # Create a mock function that always fails
    error = ValueError("fail")
    mock_func = Mock(side_effect=error)
    mock_func.__name__ = "mock_func"  # Add name attribute
    limited_func = limiter(mock_func)
    
    # Should raise after max retries
    with pytest.raises(ValueError) as exc_info:
        limited_func()
    
    assert exc_info.value == error
    assert mock_func.call_count == 3  # Initial + 2 retries


def test_backoff_rate_limiter_cleanup():
    """Test backoff rate limiter cleanup of old retry data."""
    limiter = BackoffRateLimiter(
        max_requests=2,
        time_window=1.0,
        retry_interval=0.1
    )
    
    # Add some expired retry data
    old_time = datetime.now() - timedelta(seconds=10)
    key = "test_func:():{}:"
    limiter._retry_counts[key] = 1
    limiter._backoff_until[key] = old_time
    
    # Cleanup should remove it
    limiter._cleanup_old_retries()
    assert len(limiter._retry_counts) == 0
    assert len(limiter._backoff_until) == 0


def test_backoff_rate_limiter_increasing_delays():
    """Test that backoff intervals increase with each retry."""
    limiter = BackoffRateLimiter(
        max_requests=2,
        time_window=1.0,
        retry_interval=0.1,
        max_retries=3,
        backoff_factor=2.0
    )
    
    # Mock the backoff sleep to track delays
    backoff_sleeps = []
    original_backoff_sleep = limiter._backoff_sleep
    
    def mock_backoff_sleep(duration):
        backoff_sleeps.append(duration)
        original_backoff_sleep(0)  # Don't actually sleep
    
    limiter._backoff_sleep = mock_backoff_sleep
    
    # Create a mock function that always fails
    mock_func = Mock(side_effect=ValueError("fail"))
    mock_func.__name__ = "mock_func"  # Add name attribute
    limited_func = limiter(mock_func)
    
    # Should try with increasing delays
    with pytest.raises(ValueError):
        limited_func()
    
    # Should have increasing intervals
    assert len(backoff_sleeps) == 3
    assert backoff_sleeps[1] > backoff_sleeps[0]
    assert backoff_sleeps[2] > backoff_sleeps[1] 