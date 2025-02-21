"""Tests for OHLCV data aggregation utilities."""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.marketdata.utils.aggregation import (
    ohlcv_agg,
    vwap_agg,
    time_weighted_agg,
    tick_resample,
    volume_resample,
    dollar_resample,
    custom_resample
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = {
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    }
    
    # Ensure OHLC relationships are valid
    df = pd.DataFrame(data, index=dates)
    df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['low'])
    df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['high'])
    
    return df


def test_ohlcv_agg():
    """Test standard OHLCV aggregation functions."""
    df = pd.DataFrame()  # Empty DataFrame is sufficient for this test
    agg_funcs = ohlcv_agg(df)
    
    assert agg_funcs['open'] == 'first'
    assert agg_funcs['high'] == 'max'
    assert agg_funcs['low'] == 'min'
    assert agg_funcs['close'] == 'last'
    assert agg_funcs['volume'] == 'sum'


def test_vwap_agg(sample_data):
    """Test VWAP calculation."""
    vwap = vwap_agg(sample_data)
    
    # Manual VWAP calculation
    expected = (sample_data['close'] * sample_data['volume']).sum() / sample_data['volume'].sum()
    
    assert np.isclose(vwap, expected)
    assert not np.isnan(vwap)


def test_time_weighted_agg(sample_data):
    """Test time-weighted average price calculation."""
    twap = time_weighted_agg(sample_data)
    
    # Verify basic properties
    assert not np.isnan(twap)
    assert twap >= sample_data['close'].min()
    assert twap <= sample_data['close'].max()


def test_tick_resample(sample_data):
    """Test tick-based resampling."""
    ticks = 5
    resampled = tick_resample(sample_data, ticks)
    
    # Check number of bars
    expected_bars = len(sample_data) // ticks + (1 if len(sample_data) % ticks else 0)
    assert len(resampled) == expected_bars
    
    # Check OHLC relationships
    assert (resampled['high'] >= resampled['low']).all()
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all()


def test_volume_resample(sample_data):
    """Test volume-based resampling."""
    volume_threshold = sample_data['volume'].mean() * 5
    resampled = volume_resample(sample_data, volume_threshold)
    
    # Check OHLC relationships
    assert (resampled['high'] >= resampled['low']).all()
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all()
    
    # Check volume aggregation
    assert (resampled['volume'] >= volume_threshold).all()


def test_dollar_resample(sample_data):
    """Test dollar volume-based resampling."""
    dollar_threshold = (sample_data['close'] * sample_data['volume']).mean() * 5
    resampled = dollar_resample(sample_data, dollar_threshold)
    
    # Check OHLC relationships
    assert (resampled['high'] >= resampled['low']).all()
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all()


def test_custom_resample(sample_data):
    """Test custom resampling with additional aggregations."""
    # Define custom aggregation
    def price_range(x):
        return x['high'].max() - x['low'].min()
    
    custom_aggs = {'range': price_range}
    
    resampled = custom_resample(
        sample_data,
        rule='5min',
        custom_aggs=custom_aggs
    )
    
    # Check standard OHLC relationships
    assert (resampled['high'] >= resampled['low']).all()
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all()
    
    # Check custom aggregation
    assert 'range' in resampled.columns
    assert (resampled['range'] >= 0).all()
    assert (resampled['range'] <= resampled['high'] - resampled['low']).all()


def test_custom_resample_with_custom_agg_funcs(sample_data):
    """Test custom resampling with custom aggregation functions."""
    agg_funcs = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'mean'  # Changed from sum to mean
    }
    
    resampled = custom_resample(
        sample_data,
        rule='5min',
        agg_funcs=agg_funcs
    )
    
    # Check that volume is averaged instead of summed
    assert (resampled['volume'] <= sample_data['volume'].max()).all()
    
    # Check OHLC relationships still hold
    assert (resampled['high'] >= resampled['low']).all()
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all() 