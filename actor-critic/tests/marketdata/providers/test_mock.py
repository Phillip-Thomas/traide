"""Tests for the MockMarketDataProvider class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.marketdata.providers.mock import MockMarketDataProvider


@pytest.fixture
def provider():
    """Create a mock provider instance."""
    return MockMarketDataProvider(seed=42)


def test_supported_timeframes(provider):
    """Test supported timeframes property."""
    expected = ['1m', '5m', '15m', '1h', '4h', '1d']
    assert provider.supported_timeframes == expected


def test_timeframe_to_minutes(provider):
    """Test timeframe to minutes conversion."""
    # Test minutes
    assert provider._timeframe_to_minutes('1m') == 1
    assert provider._timeframe_to_minutes('5m') == 5
    assert provider._timeframe_to_minutes('15m') == 15
    
    # Test hours
    assert provider._timeframe_to_minutes('1h') == 60
    assert provider._timeframe_to_minutes('4h') == 240
    
    # Test days
    assert provider._timeframe_to_minutes('1d') == 1440
    
    # Test invalid timeframe
    with pytest.raises(ValueError):
        provider._timeframe_to_minutes('1w')


def test_generate_ohlcv(provider):
    """Test synthetic OHLCV data generation."""
    n_points = 100
    base_price = 100.0
    
    df = provider._generate_ohlcv(n_points, base_price)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'open', 'high', 'low', 'close', 'volume'}
    assert len(df) == n_points
    
    # Check data types
    assert df['open'].dtype == np.float64
    assert df['high'].dtype == np.float64
    assert df['low'].dtype == np.float64
    assert df['close'].dtype == np.float64
    assert df['volume'].dtype == np.int64
    
    # Check OHLC relationships
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['low'] <= df['open']).all()
    assert (df['low'] <= df['close']).all()
    
    # Check volume is positive
    assert (df['volume'] > 0).all()


def test_fetch_historical_data(provider):
    """Test historical data fetching."""
    start_date = '2024-01-01'
    end_date = '2024-01-02'
    symbol = 'AAPL'
    timeframe = '1h'
    
    df = provider.fetch_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'timestamp'
    assert set(df.columns) == {'open', 'high', 'low', 'close', 'volume'}
    
    # Check time range
    assert df.index[0].strftime('%Y-%m-%d') == start_date
    assert df.index[-1].strftime('%Y-%m-%d') == end_date
    
    # Check timeframe
    time_diff = df.index[1] - df.index[0]
    assert time_diff == timedelta(hours=1)
    
    # Convert index to column for validation
    df_validate = df.reset_index()
    provider.validate_data(df_validate)


def test_get_latest_data(provider):
    """Test latest data fetching."""
    symbol = 'AAPL'
    timeframe = '1d'
    
    df = provider.get_latest_data(symbol=symbol, timeframe=timeframe)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'timestamp'
    assert set(df.columns) == {'open', 'high', 'low', 'close', 'volume'}
    
    # Check we only get one data point
    assert len(df) == 1
    
    # Convert index to column for validation
    df_validate = df.reset_index()
    provider.validate_data(df_validate)


def test_reproducibility(provider):
    """Test that data generation is reproducible with same seed."""
    params = {
        'symbol': 'AAPL',
        'start_date': '2024-01-01',
        'end_date': '2024-01-02',
        'timeframe': '1h'
    }
    
    # Generate data twice with same provider (same seed)
    df1 = provider.fetch_historical_data(**params)
    df2 = provider.fetch_historical_data(**params)
    
    # Data should be identical
    pd.testing.assert_frame_equal(df1, df2)
    
    # Generate data with different seed
    different_provider = MockMarketDataProvider(seed=43)
    df3 = different_provider.fetch_historical_data(**params)
    
    # Data should be different
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3) 