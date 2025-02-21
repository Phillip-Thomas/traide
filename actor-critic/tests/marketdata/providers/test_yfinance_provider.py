"""Integration tests for the YFinance provider."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.marketdata.providers.yfinance_provider import YFinanceProvider


@pytest.fixture
def provider():
    """Create a YFinance provider instance."""
    return YFinanceProvider()


def test_supported_timeframes(provider):
    """Test that supported timeframes are correctly reported."""
    timeframes = provider.supported_timeframes
    assert len(timeframes) > 0
    assert '1d' in timeframes
    assert '1h' in timeframes
    assert '1m' in timeframes


def test_fetch_historical_daily_data(provider):
    """Test fetching daily historical data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = provider.fetch_historical_data(
        symbol='AAPL',
        start_date=start_date,
        end_date=end_date,
        timeframe='1d'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns
    assert df.index.name == 'timestamp'
    assert df.index.is_monotonic_increasing


def test_fetch_historical_hourly_data(provider):
    """Test fetching hourly historical data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = provider.fetch_historical_data(
        symbol='AAPL',
        start_date=start_date,
        end_date=end_date,
        timeframe='1h'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == 'timestamp'
    assert df.index.is_monotonic_increasing


def test_fetch_historical_minute_data(provider):
    """Test fetching minute historical data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    df = provider.fetch_historical_data(
        symbol='AAPL',
        start_date=start_date,
        end_date=end_date,
        timeframe='1m'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == 'timestamp'
    assert df.index.is_monotonic_increasing


def test_get_latest_data(provider):
    """Test getting latest data."""
    df = provider.get_latest_data(
        symbol='AAPL',
        timeframe='1d'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns
    assert df.index.name == 'timestamp'


def test_invalid_timeframe(provider):
    """Test that invalid timeframes raise ValueError."""
    with pytest.raises(ValueError):
        provider.fetch_historical_data(
            symbol='AAPL',
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            timeframe='invalid'
        )


def test_invalid_symbol(provider):
    """Test that invalid symbols raise ConnectionError."""
    with pytest.raises(ConnectionError):
        provider.fetch_historical_data(
            symbol='INVALID_SYMBOL_123',
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            timeframe='1d'
        )


def test_extended_hours_data(provider):
    """Test fetching data with extended hours."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = provider.fetch_historical_data(
        symbol='AAPL',
        start_date=start_date,
        end_date=end_date,
        timeframe='1d',
        include_extended_hours=True
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == 'timestamp'


def test_data_validation(provider):
    """Test that data validation works correctly."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = provider.fetch_historical_data(
        symbol='AAPL',
        start_date=start_date,
        end_date=end_date,
        timeframe='1d'
    )
    
    # Check OHLC relationships
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['low'] <= df['open']).all()
    assert (df['low'] <= df['close']).all()
    
    # Check for positive values
    assert (df['open'] > 0).all()
    assert (df['high'] > 0).all()
    assert (df['low'] > 0).all()
    assert (df['close'] > 0).all()
    assert (df['volume'] >= 0).all()
    
    # Check for NaN values
    assert not df['open'].isnull().any()
    assert not df['high'].isnull().any()
    assert not df['low'].isnull().any()
    assert not df['close'].isnull().any()
    assert not df['volume'].isnull().any() 