"""Unit tests for volatility indicators."""

import numpy as np
import pandas as pd
import pytest

from marketdata.indicators.volatility import (
    BollingerBands,
    BollingerBandsParams,
    AverageTrueRange,
    ATRParams
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    close = 100 + np.random.randn(n).cumsum()
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n)
    volume = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def trend_data():
    """Create trending data for testing."""
    n = 100
    x = np.linspace(0, 4*np.pi, n)
    trend = 100 + 20*np.sin(x) + np.linspace(0, 50, n)
    noise = np.random.randn(n) * 2
    close = trend + noise
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n)
    volume = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_bbands_params():
    """Test Bollinger Bands parameters."""
    params = BollingerBandsParams()
    assert params.period == 20
    assert params.num_std == 2.0
    assert params.source == 'close'
    assert params.min_periods is None
    
    params = BollingerBandsParams(period=10, num_std=1.5, source='high', min_periods=5)
    assert params.period == 10
    assert params.num_std == 1.5
    assert params.source == 'high'
    assert params.min_periods == 5


def test_bbands_calculation(trend_data):
    """Test Bollinger Bands calculation."""
    bbands = BollingerBands()
    result = bbands.calculate(trend_data)
    
    # Check result structure
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['upper', 'middle', 'lower', 'width', 'percent_b'])
    
    # Check band relationships
    assert (result.values['upper'] >= result.values['middle']).all()
    assert (result.values['middle'] >= result.values['lower']).all()
    
    # Check width is non-negative
    assert (result.values['width'] >= 0).all()
    
    # Check %B is between 0 and 1
    assert (result.values['percent_b'] >= 0).all()
    assert (result.values['percent_b'] <= 1).all()
    
    # Check metadata
    assert result.metadata['period'] == 20
    assert result.metadata['num_std'] == 2.0
    assert result.metadata['source'] == 'close'


def test_bbands_with_different_sources(sample_data):
    """Test Bollinger Bands with different price sources."""
    sources = ['open', 'high', 'low', 'close']
    for source in sources:
        bbands = BollingerBands(BollingerBandsParams(source=source))
        result = bbands.calculate(sample_data)
        assert result.metadata['source'] == source


def test_bbands_with_gaps(sample_data):
    """Test Bollinger Bands with missing data."""
    # Create gaps in data
    sample_data.loc[10:15, 'close'] = np.nan
    
    bbands = BollingerBands(BollingerBandsParams(min_periods=5))
    result = bbands.calculate(sample_data)
    
    # Check that gaps are propagated
    assert result.values.loc[10:15, 'middle'].isna().all()
    assert result.values.loc[10:15, 'upper'].isna().all()
    assert result.values.loc[10:15, 'lower'].isna().all()


def test_atr_params():
    """Test ATR parameters."""
    params = ATRParams()
    assert params.period == 14
    assert params.smoothing == 'rma'
    assert params.min_periods is None
    
    params = ATRParams(period=10, smoothing='ema', min_periods=5)
    assert params.period == 10
    assert params.smoothing == 'ema'
    assert params.min_periods == 5


def test_atr_calculation(sample_data):
    """Test ATR calculation."""
    atr = AverageTrueRange()
    result = atr.calculate(sample_data)
    
    # Check result structure
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['tr', 'atr', 'atr_percent'])
    
    # Check values are non-negative
    assert (result.values['tr'] >= 0).all()
    assert (result.values['atr'] >= 0).all()
    assert (result.values['atr_percent'] >= 0).all()
    
    # Check metadata
    assert result.metadata['period'] == 14
    assert result.metadata['smoothing'] == 'rma'


def test_atr_with_different_smoothing(sample_data):
    """Test ATR with different smoothing methods."""
    smoothing_methods = ['sma', 'ema', 'rma']
    for smoothing in smoothing_methods:
        atr = AverageTrueRange(ATRParams(smoothing=smoothing))
        result = atr.calculate(sample_data)
        assert result.metadata['smoothing'] == smoothing


def test_atr_with_gaps(sample_data):
    """Test ATR with missing data."""
    # Create gaps in data
    sample_data.loc[10:15, ['high', 'low', 'close']] = np.nan
    
    atr = AverageTrueRange(ATRParams(min_periods=5))
    result = atr.calculate(sample_data)
    
    # Check that gaps are propagated
    assert result.values.loc[10:15, 'tr'].isna().all()
    assert result.values.loc[10:15, 'atr'].isna().all()
    assert result.values.loc[10:15, 'atr_percent'].isna().all()


def test_atr_true_range_calculation(sample_data):
    """Test ATR True Range calculation."""
    atr = AverageTrueRange()
    result = atr.calculate(sample_data)
    
    # Calculate true range manually for verification
    high_low = sample_data['high'] - sample_data['low']
    high_close = (sample_data['high'] - sample_data['close'].shift()).abs()
    low_close = (sample_data['low'] - sample_data['close'].shift()).abs()
    
    expected_tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    expected_tr.iloc[0] = high_low.iloc[0]  # First value should use high-low
    
    # Compare with calculated values
    pd.testing.assert_series_equal(
        result.values['tr'],
        expected_tr,
        check_names=False
    ) 