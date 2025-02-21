"""Unit tests for moving average indicators."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.indicators.moving_averages import (
    MovingAverageParams,
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage,
    HullMovingAverage,
    VolumeWeightedMovingAverage
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = {
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    }
    return pd.DataFrame(data).set_index('timestamp')


@pytest.fixture
def linear_data():
    """Create linear data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = {
        'timestamp': dates,
        'close': np.arange(100),
        'volume': np.ones(100) * 1000
    }
    return pd.DataFrame(data).set_index('timestamp')


def test_moving_average_params():
    """Test moving average parameters."""
    params = MovingAverageParams(period=10, source='high')
    assert params.period == 10
    assert params.source == 'high'


def test_sma_calculation(linear_data):
    """Test SMA calculation."""
    sma = SimpleMovingAverage(MovingAverageParams(period=5))
    result = sma.calculate(linear_data)
    
    # First value should be itself (min_periods=1)
    assert result.values[0] == 0
    
    # After period, should be average of last n values
    assert result.values[4] == 2.0  # (0+1+2+3+4)/5
    assert result.values[5] == 3.0  # (1+2+3+4+5)/5
    
    # Test metadata
    assert result.metadata['period'] == 5
    assert result.metadata['source'] == 'close'


def test_ema_calculation(linear_data):
    """Test EMA calculation."""
    ema = ExponentialMovingAverage(MovingAverageParams(period=5))
    result = ema.calculate(linear_data)
    
    # First value should be itself (min_periods=1)
    assert result.values[0] == 0
    
    # EMA should give more weight to recent values
    assert result.values[5] > 3.0  # Should be higher than SMA
    
    # Test metadata
    assert result.metadata['period'] == 5
    assert result.metadata['source'] == 'close'


def test_wma_calculation(linear_data):
    """Test WMA calculation."""
    wma = WeightedMovingAverage(MovingAverageParams(period=5))
    result = wma.calculate(linear_data)
    
    # First value should be itself (min_periods=1)
    assert result.values[0] == 0
    
    # Calculate expected WMA manually
    weights = np.array([1, 2, 3, 4, 5]) / 15  # sum(1:5) = 15
    expected = np.sum(np.array([1, 2, 3, 4, 5]) * weights)
    assert result.values[5] == pytest.approx(expected)
    
    # Test metadata
    assert result.metadata['period'] == 5
    assert result.metadata['source'] == 'close'


def test_hull_calculation(linear_data):
    """Test Hull MA calculation."""
    hma = HullMovingAverage(MovingAverageParams(period=16))
    result = hma.calculate(linear_data)
    
    # First value should be valid
    assert not np.isnan(result.values[0])
    
    # HMA should be more responsive than regular MA
    # Test by comparing distances from actual values
    wma = WeightedMovingAverage(MovingAverageParams(period=16))
    wma_result = wma.calculate(linear_data)
    
    # Get distances from actual values
    hma_dist = np.abs(result.values - linear_data['close'])
    wma_dist = np.abs(wma_result.values - linear_data['close'])
    
    # HMA should have smaller average distance
    assert hma_dist.mean() < wma_dist.mean()
    
    # Test metadata
    assert result.metadata['period'] == 16
    assert result.metadata['source'] == 'close'


def test_vwma_calculation(linear_data):
    """Test VWMA calculation."""
    vwma = VolumeWeightedMovingAverage(MovingAverageParams(period=5))
    result = vwma.calculate(linear_data)
    
    # First value should be itself (min_periods=1)
    assert result.values[0] == 0
    
    # With constant volume, should be same as SMA
    sma = SimpleMovingAverage(MovingAverageParams(period=5))
    sma_result = sma.calculate(linear_data)
    
    np.testing.assert_array_almost_equal(
        result.values,
        sma_result.values
    )
    
    # Test with varying volume
    linear_data['volume'] = np.arange(100) + 1
    result = vwma.calculate(linear_data)
    
    # VWMA should be different from SMA with varying volume
    assert not np.allclose(result.values, sma_result.values)
    
    # Test metadata
    assert result.metadata['period'] == 5
    assert result.metadata['source'] == 'close'


def test_moving_averages_with_gaps(sample_data):
    """Test moving averages with missing data."""
    # Create gaps in data
    sample_data.loc[sample_data.index[10:15], 'close'] = np.nan
    
    indicators = [
        SimpleMovingAverage(),
        ExponentialMovingAverage(),
        WeightedMovingAverage(),
        HullMovingAverage(),
        VolumeWeightedMovingAverage()
    ]
    
    for indicator in indicators:
        result = indicator.calculate(sample_data)
        
        # Should handle NaN values
        assert np.isnan(result.values[10:15]).all()
        
        # Should recover after gap
        assert not np.isnan(result.values[20:]).any()


def test_moving_averages_with_different_sources(sample_data):
    """Test moving averages with different price sources."""
    sources = ['open', 'high', 'low', 'close']
    
    indicators = [
        SimpleMovingAverage(),
        ExponentialMovingAverage(),
        WeightedMovingAverage(),
        HullMovingAverage(),
        VolumeWeightedMovingAverage()
    ]
    
    for indicator in indicators:
        for source in sources:
            params = MovingAverageParams(source=source)
            indicator.params = params
            result = indicator.calculate(sample_data)
            
            # Should use correct source
            assert result.metadata['source'] == source
            
            # Values should be different for different sources
            other_sources = [s for s in sources if s != source]
            for other_source in other_sources:
                other_params = MovingAverageParams(source=other_source)
                indicator.params = other_params
                other_result = indicator.calculate(sample_data)
                
                assert not np.allclose(
                    result.values,
                    other_result.values
                )


def test_moving_averages_with_different_periods(sample_data):
    """Test moving averages with different periods."""
    periods = [5, 10, 20, 50]
    
    indicators = [
        SimpleMovingAverage(),
        ExponentialMovingAverage(),
        WeightedMovingAverage(),
        HullMovingAverage(),
        VolumeWeightedMovingAverage()
    ]
    
    for indicator in indicators:
        results = []
        for period in periods:
            params = MovingAverageParams(period=period)
            indicator.params = params
            result = indicator.calculate(sample_data)
            results.append(result.values)
            
            # Should use correct period
            assert result.metadata['period'] == period
        
        # Longer periods should be smoother (less variance)
        variances = [r.var() for r in results]
        assert all(v1 > v2 for v1, v2 in zip(variances[:-1], variances[1:])) 