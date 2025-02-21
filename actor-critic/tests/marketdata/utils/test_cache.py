"""Tests for caching utilities."""

import json
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.marketdata.utils.cache import Cache, FileCache, cache_result


class SimpleCache(Cache):
    """Simple in-memory cache for testing."""
    
    def __init__(self):
        self._data = {}
        self._expires = {}
    
    def get(self, key):
        if key not in self._data:
            return None
        
        # Check expiration
        if key in self._expires:
            if datetime.now() > self._expires[key]:
                self.delete(key)
                return None
        
        return self._data.get(key)
    
    def set(self, key, value, ttl=None):
        self._data[key] = value
        if ttl is not None:
            self._expires[key] = datetime.now() + timedelta(seconds=ttl)
    
    def delete(self, key):
        self._data.pop(key, None)
        self._expires.pop(key, None)
    
    def clear(self):
        self._data.clear()
        self._expires.clear()


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary directory for cache files."""
    return tmp_path / "cache"


@pytest.fixture
def file_cache(cache_dir):
    """Create a FileCache instance."""
    return FileCache(cache_dir)


def test_file_cache_basic(file_cache):
    """Test basic cache operations."""
    # Set and get
    file_cache.set("test_key", "test_value")
    assert file_cache.get("test_key") == "test_value"
    
    # Delete
    file_cache.delete("test_key")
    assert file_cache.get("test_key") is None
    
    # Clear
    file_cache.set("key1", "value1")
    file_cache.set("key2", "value2")
    file_cache.clear()
    assert file_cache.get("key1") is None
    assert file_cache.get("key2") is None


def test_file_cache_ttl(file_cache):
    """Test cache TTL functionality."""
    # Set with TTL
    file_cache.set("test_key", "test_value", ttl=1)
    assert file_cache.get("test_key") == "test_value"
    
    # Wait for expiration
    time.sleep(1.1)
    assert file_cache.get("test_key") is None


def test_file_cache_dataframe(file_cache):
    """Test caching pandas DataFrames."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    
    file_cache.set("test_df", df)
    result = file_cache.get("test_df")
    
    pd.testing.assert_frame_equal(df, result)


def test_file_cache_invalid_json(file_cache):
    """Test handling of invalid cache files."""
    # Write invalid JSON
    cache_path = file_cache._get_cache_path("test_key")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("invalid json")
    
    assert file_cache.get("test_key") is None


def test_file_cache_expired_metadata(file_cache):
    """Test handling of expired cache entries."""
    # Write expired cache entry
    cache_path = file_cache._get_cache_path("test_key")
    expired_time = (datetime.now() - timedelta(hours=1)).isoformat()
    data = {
        'metadata': {
            'created_at': expired_time,
            'expires_at': expired_time,
            'type': 'value'
        },
        'value': "test_value"
    }
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    
    assert file_cache.get("test_key") is None


def test_cache_result_decorator():
    """Test the cache_result decorator."""
    cache = SimpleCache()
    mock_func = Mock(return_value="test_value")
    
    @cache_result(cache)
    def cached_func(arg1, arg2=None):
        return mock_func(arg1, arg2)
    
    # First call should miss cache
    result1 = cached_func("a", arg2="b")
    assert result1 == "test_value"
    assert mock_func.call_count == 1
    
    # Second call should hit cache
    result2 = cached_func("a", arg2="b")
    assert result2 == "test_value"
    assert mock_func.call_count == 1  # Still 1
    
    # Different args should miss cache
    result3 = cached_func("c", arg2="d")
    assert result3 == "test_value"
    assert mock_func.call_count == 2


def test_cache_result_ttl():
    """Test the cache_result decorator with TTL."""
    cache = SimpleCache()
    mock_func = Mock(return_value="test_value")
    
    @cache_result(cache, ttl=1)
    def cached_func():
        return mock_func()
    
    # First call
    result1 = cached_func()
    assert result1 == "test_value"
    assert mock_func.call_count == 1
    
    # Wait for TTL
    time.sleep(1.1)
    
    # Should miss cache
    result2 = cached_func()
    assert result2 == "test_value"
    assert mock_func.call_count == 2


def test_cache_result_key_prefix():
    """Test the cache_result decorator with key prefix."""
    cache = SimpleCache()
    
    @cache_result(cache, key_prefix="prefix1")
    def func1():
        return "value1"
    
    @cache_result(cache, key_prefix="prefix2")
    def func2():
        return "value2"
    
    # Different prefixes should not conflict
    func1()
    func2()
    
    assert cache.get("prefix1:func1") == "value1"
    assert cache.get("prefix2:func2") == "value2" 