"""Tests for data processing utilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.marketdata.utils.processor import DataProcessor


@pytest.fixture
def processor():
    """Create a DataProcessor instance."""
    return DataProcessor()


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    timestamps = pd.date_range(
        start='2024-01-01',
        end='2024-01-02',
        freq='1min'
    )
    
    # Generate realistic OHLCV data
    n_points = len(timestamps)
    closes = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_points)))
    highs = closes * np.exp(np.random.uniform(0, 0.002, n_points))
    lows = closes * np.exp(-np.random.uniform(0, 0.002, n_points))
    opens = lows + (highs - lows) * np.random.uniform(0, 1, n_points)
    volumes = np.random.lognormal(10, 1, n_points).astype(int)
    
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=timestamps)


def test_validate_data(processor, sample_data):
    """Test data validation."""
    # Valid data should not raise
    processor.validate_data(sample_data)
    
    # Test invalid index
    invalid_data = sample_data.reset_index()
    with pytest.raises(ValueError, match="must have DatetimeIndex"):
        processor.validate_data(invalid_data)
    
    # Test non-monotonic index
    invalid_data = sample_data.copy()
    invalid_data.index = invalid_data.index[::-1]
    with pytest.raises(ValueError, match="monotonically increasing"):
        processor.validate_data(invalid_data)
    
    # Test missing columns
    invalid_data = sample_data.drop('volume', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        processor.validate_data(invalid_data)
    
    # Test null values
    invalid_data = sample_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    with pytest.raises(ValueError, match="contains null values"):
        processor.validate_data(invalid_data)
    
    # Test non-positive values
    invalid_data = sample_data.copy()
    invalid_data.loc[invalid_data.index[0], 'volume'] = 0
    with pytest.raises(ValueError, match="contains non-positive values"):
        processor.validate_data(invalid_data)
    
    # Test invalid OHLC relationships
    invalid_data = sample_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = (
        invalid_data.loc[invalid_data.index[0], 'low'] - 1.0
    )
    with pytest.raises(ValueError, match="Invalid OHLC relationships"):
        processor.validate_data(invalid_data)


def test_timeframe_to_offset(processor):
    """Test timeframe to offset conversion."""
    # Test valid timeframes
    assert processor._timeframe_to_offset('1m') == pd.Timedelta(minutes=1)
    assert processor._timeframe_to_offset('1h') == pd.Timedelta(hours=1)
    assert processor._timeframe_to_offset('1d') == pd.Timedelta(days=1)
    assert processor._timeframe_to_offset('1w') == pd.Timedelta(weeks=1)
    assert processor._timeframe_to_offset('1mo') == pd.Timedelta(days=30)
    
    # Test invalid timeframes
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        processor._timeframe_to_offset('1s')  # Invalid unit
    
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        processor._timeframe_to_offset('invalid')


def test_resample_basic(processor, sample_data):
    """Test basic resampling functionality."""
    # Resample to 5-minute bars
    resampled = processor.resample(sample_data, '5m')
    
    # Check basic properties
    assert isinstance(resampled, pd.DataFrame)
    assert isinstance(resampled.index, pd.DatetimeIndex)
    assert set(resampled.columns) == {'open', 'high', 'low', 'close', 'volume', 'vwap'}
    
    # Check that timestamps are 5 minutes apart
    time_diffs = resampled.index[1:] - resampled.index[:-1]
    assert all(diff == pd.Timedelta(minutes=5) for diff in time_diffs)
    
    # Check OHLC relationships
    assert (resampled['high'] >= resampled['low']).all()
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all()


def test_resample_volume_weighted(processor, sample_data):
    """Test volume-weighted resampling."""
    # Resample with volume weighting
    resampled_vw = processor.resample(sample_data, '5m', volume_weighted=True)
    
    # Resample without volume weighting
    resampled_simple = processor.resample(sample_data, '5m', volume_weighted=False)
    
    # Check VWAP column
    assert 'vwap' in resampled_vw.columns
    assert 'vwap' not in resampled_simple.columns
    
    # Check VWAP calculation
    first_group = sample_data.iloc[:5]  # First 5 minutes
    expected_vwap = (
        (first_group['close'] * first_group['volume']).sum() /
        first_group['volume'].sum()
    )
    assert abs(resampled_vw['vwap'].iloc[0] - expected_vwap) < 1e-10


def test_resample_edge_cases(processor, sample_data):
    """Test resampling edge cases."""
    # Single row
    single_row = sample_data.iloc[[0]]
    resampled = processor.resample(single_row, '5m')
    assert len(resampled) == 1
    
    # Empty DataFrame
    empty_df = sample_data.iloc[0:0]
    resampled = processor.resample(empty_df, '5m')
    assert len(resampled) == 0
    
    # Sparse data
    sparse_data = sample_data.iloc[::10]  # Every 10th row
    resampled = processor.resample(sparse_data, '5m')
    assert not resampled.isnull().any().any()  # No NaN values after forward fill 