"""Tests for the base MarketDataProvider class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.marketdata.providers.base import MarketDataProvider


class SimpleProvider(MarketDataProvider):
    """Simple concrete implementation for testing."""
    
    @property
    def supported_timeframes(self):
        return ['1m', '5m', '1h']
    
    def fetch_historical_data(self, *args, **kwargs):
        pass
    
    def get_latest_data(self, *args, **kwargs):
        pass


def test_validate_timeframe():
    """Test timeframe validation."""
    provider = SimpleProvider()
    
    # Valid timeframe should not raise
    provider.validate_timeframe('1m')
    
    # Invalid timeframe should raise ValueError
    with pytest.raises(ValueError):
        provider.validate_timeframe('invalid')


def test_validate_data():
    """Test market data validation."""
    provider = SimpleProvider()
    
    # Valid data
    valid_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    provider.validate_data(valid_data)
    
    # Missing columns
    invalid_data = valid_data.drop(columns=['volume'])
    with pytest.raises(ValueError, match="Missing required columns"):
        provider.validate_data(invalid_data)
    
    # Non-monotonic index
    invalid_data = valid_data.copy()
    invalid_data.iloc[0, invalid_data.columns.get_loc('timestamp')] = pd.Timestamp('2024-01-10')
    with pytest.raises(ValueError, match="monotonically increasing"):
        provider.validate_data(invalid_data)
    
    # Null values
    invalid_data = valid_data.copy()
    invalid_data.iloc[0, invalid_data.columns.get_loc('close')] = np.nan
    with pytest.raises(ValueError, match="contains null values"):
        provider.validate_data(invalid_data)
    
    # Non-positive values
    invalid_data = valid_data.copy()
    invalid_data.iloc[0, invalid_data.columns.get_loc('volume')] = -100
    with pytest.raises(ValueError, match="contains non-positive values"):
        provider.validate_data(invalid_data)
    
    # Invalid OHLC relationships
    invalid_data = valid_data.copy()
    invalid_data.iloc[0, invalid_data.columns.get_loc('high')] = 90.0  # High below low
    with pytest.raises(ValueError, match="Invalid OHLC relationships"):
        provider.validate_data(invalid_data)


def test_standardize_data():
    """Test market data standardization."""
    provider = SimpleProvider()
    
    # Create test data with mixed types
    data = pd.DataFrame({
        'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [100, 101, 102],  # integers
        'high': [105.0, 106.0, 107.0],  # floats
        'low': np.array([95, 96, 97]),  # numpy integers
        'close': np.array([102.0, 103.0, 104.0]),  # numpy floats
        'volume': [1000, 1100, 1200]
    })
    
    # Standardize data
    result = provider.standardize_data(data)
    
    # Check timestamp is datetime index
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == 'timestamp'
    assert result.index.is_monotonic_increasing
    
    # Check numeric columns are float64
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        assert result[col].dtype == np.float64 