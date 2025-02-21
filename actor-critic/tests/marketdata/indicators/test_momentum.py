"""Unit tests for momentum indicators."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.indicators.momentum import (
    RSIParams,
    RelativeStrengthIndex,
    MACDParams,
    MovingAverageConvergenceDivergence
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
def trend_data():
    """Create trending data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    trend = np.concatenate([
        np.linspace(100, 150, 50),  # Uptrend
        np.linspace(150, 100, 50)   # Downtrend
    ])
    data = {
        'timestamp': dates,
        'close': trend,
        'volume': np.ones(100) * 1000
    }
    return pd.DataFrame(data).set_index('timestamp')


def test_rsi_params():
    """Test RSI parameters."""
    params = RSIParams(period=10, source='high', min_periods=5)
    assert params.period == 10
    assert params.source == 'high'
    assert params.min_periods == 5


def test_rsi_calculation(trend_data):
    """Test RSI calculation."""
    rsi = RelativeStrengthIndex(RSIParams(period=14))
    result = rsi.calculate(trend_data)
    
    # Check result type
    assert isinstance(result.values, pd.Series)
    
    # Check bounds
    assert result.values.min() >= 0
    assert result.values.max() <= 100
    
    # Check metadata
    assert result.metadata['period'] == 14
    assert result.metadata['source'] == 'close'
    
    # Check trend response
    # RSI should be high during uptrend and low during downtrend
    mid_point = len(trend_data) // 2
    uptrend_rsi = result.values[mid_point-5:mid_point].mean()
    downtrend_rsi = result.values[-5:].mean()
    assert uptrend_rsi > 60  # High during uptrend
    assert downtrend_rsi < 40  # Low during downtrend


def test_rsi_with_different_sources(sample_data):
    """Test RSI with different price sources."""
    sources = ['open', 'high', 'low', 'close']
    
    for source in sources:
        params = RSIParams(source=source)
        rsi = RelativeStrengthIndex(params)
        result = rsi.calculate(sample_data)
        
        # Check source in metadata
        assert result.metadata['source'] == source
        
        # Values should be different for different sources
        other_sources = [s for s in sources if s != source]
        for other_source in other_sources:
            other_params = RSIParams(source=other_source)
            other_rsi = RelativeStrengthIndex(other_params)
            other_result = other_rsi.calculate(sample_data)
            
            assert not np.allclose(
                result.values,
                other_result.values
            )


def test_rsi_with_gaps(sample_data):
    """Test RSI with missing data."""
    # Create gaps in data
    sample_data.loc[sample_data.index[10:15], 'close'] = np.nan
    
    rsi = RelativeStrengthIndex()
    result = rsi.calculate(sample_data)
    
    # Should handle NaN values
    assert np.isnan(result.values[10:15]).all()
    
    # Should recover after gap
    assert not np.isnan(result.values[20:]).any()


def test_macd_params():
    """Test MACD parameters."""
    params = MACDParams(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        source='high',
        min_periods=12
    )
    assert params.fast_period == 12
    assert params.slow_period == 26
    assert params.signal_period == 9
    assert params.source == 'high'
    assert params.min_periods == 12


def test_macd_calculation(trend_data):
    """Test MACD calculation."""
    macd = MovingAverageConvergenceDivergence(MACDParams())
    result = macd.calculate(trend_data)
    
    # Check result type
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['macd', 'signal', 'histogram'])
    
    # Check metadata
    assert result.metadata['fast_period'] == 12
    assert result.metadata['slow_period'] == 26
    assert result.metadata['signal_period'] == 9
    assert result.metadata['source'] == 'close'
    
    # Check trend response
    # MACD should be positive during uptrend and negative during downtrend
    mid_point = len(trend_data) // 2
    uptrend_macd = result.values['macd'][mid_point-5:mid_point].mean()
    downtrend_macd = result.values['macd'][-5:].mean()
    assert uptrend_macd > 0  # Positive during uptrend
    assert downtrend_macd < 0  # Negative during downtrend


def test_macd_with_different_sources(sample_data):
    """Test MACD with different price sources."""
    sources = ['open', 'high', 'low', 'close']
    
    for source in sources:
        params = MACDParams(source=source)
        macd = MovingAverageConvergenceDivergence(params)
        result = macd.calculate(sample_data)
        
        # Check source in metadata
        assert result.metadata['source'] == source
        
        # Values should be different for different sources
        other_sources = [s for s in sources if s != source]
        for other_source in other_sources:
            other_params = MACDParams(source=other_source)
            other_macd = MovingAverageConvergenceDivergence(other_params)
            other_result = other_macd.calculate(sample_data)
            
            assert not np.allclose(
                result.values['macd'],
                other_result.values['macd']
            )


def test_macd_with_gaps(sample_data):
    """Test MACD with missing data."""
    # Create gaps in data
    sample_data.loc[sample_data.index[10:15], 'close'] = np.nan
    
    macd = MovingAverageConvergenceDivergence()
    result = macd.calculate(sample_data)
    
    # Should handle NaN values
    assert np.isnan(result.values.loc[sample_data.index[10:15]]).all().all()
    
    # Should recover after gap
    assert not np.isnan(result.values.loc[sample_data.index[20:]]).all().all()


def test_macd_with_different_periods(trend_data):
    """Test MACD with different periods."""
    # Test with shorter periods
    short_params = MACDParams(fast_period=6, slow_period=13, signal_period=4)
    short_macd = MovingAverageConvergenceDivergence(short_params)
    short_result = short_macd.calculate(trend_data)
    
    # Test with longer periods
    long_params = MACDParams(fast_period=24, slow_period=52, signal_period=18)
    long_macd = MovingAverageConvergenceDivergence(long_params)
    long_result = long_macd.calculate(trend_data)
    
    # Shorter periods should be more responsive
    short_std = short_result.values['macd'].std()
    long_std = long_result.values['macd'].std()
    assert short_std > long_std  # More volatile with shorter periods 