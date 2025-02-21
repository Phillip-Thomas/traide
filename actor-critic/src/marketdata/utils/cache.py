"""Caching utilities for market data providers."""

import os
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class Cache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache."""
        pass


class FileCache(Cache):
    """File-based cache implementation."""
    
    def __init__(self, cache_dir: Union[str, Path]):
        """Initialize the file cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path: Path to cache file
        """
        # Use hash of key as filename to avoid filesystem issues
        filename = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / filename
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if cached data is expired.
        
        Args:
            metadata: Cache metadata
            
        Returns:
            bool: True if expired
        """
        if 'expires_at' not in metadata:
            return False
        
        expires_at = datetime.fromisoformat(metadata['expires_at'])
        return datetime.now() > expires_at
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Check expiration
            if self._is_expired(data['metadata']):
                self.delete(key)
                return None
            
            # Handle special types
            if data['metadata']['type'] == 'dataframe':
                return pd.read_json(StringIO(data['value']))
            
            return data['value']
        
        except Exception as e:
            logger.warning(f"Error reading cache for key '{key}': {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        cache_path = self._get_cache_path(key)
        
        try:
            metadata = {
                'created_at': datetime.now().isoformat(),
                'type': 'value'
            }
            
            if ttl is not None:
                metadata['expires_at'] = (
                    datetime.now() + timedelta(seconds=ttl)
                ).isoformat()
            
            # Handle special types
            if isinstance(value, pd.DataFrame):
                metadata['type'] = 'dataframe'
                value = value.to_json()
            
            data = {
                'metadata': metadata,
                'value': value
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        
        except Exception as e:
            logger.warning(f"Error writing cache for key '{key}': {str(e)}")
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)
        try:
            cache_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Error deleting cache for key '{key}': {str(e)}")
    
    def clear(self) -> None:
        """Clear all values from the cache."""
        try:
            for cache_file in self.cache_dir.glob('*'):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")


def cache_result(
    cache: Cache,
    ttl: Optional[int] = None,
    key_prefix: str = ""
) -> Callable:
    """Decorator to cache function results.
    
    Args:
        cache: Cache implementation to use
        ttl: Time to live in seconds (optional)
        key_prefix: Prefix for cache keys
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key '{cache_key}'")
                return cached_value
            
            # Call function and cache result
            logger.debug(f"Cache miss for key '{cache_key}'")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    
    return decorator 