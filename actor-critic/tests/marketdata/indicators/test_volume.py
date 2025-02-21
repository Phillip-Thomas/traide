"""Unit tests for volume indicators."""

import numpy as np
import pandas as pd
import pytest

from marketdata.indicators.volume import (
    OnBalanceVolume,
    OBVParams,
    VolumeProfile,
    VolumeProfileParams
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


def test_obv_params():
    """Test OBV parameters."""
    params = OBVParams()
    assert params.source == 'close'
    assert params.signal_period == 20
    assert params.min_periods is None
    
    params = OBVParams(source='high', signal_period=10, min_periods=5)
    assert params.source == 'high'
    assert params.signal_period == 10
    assert params.min_periods == 5


def test_obv_calculation(sample_data):
    """Test OBV calculation."""
    obv = OnBalanceVolume()
    result = obv.calculate(sample_data)
    
    # Check result structure
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['obv', 'signal', 'histogram'])
    
    # Check OBV calculation
    price_changes = sample_data['close'].diff()
    expected_obv = pd.Series(0, index=sample_data.index)
    expected_obv[1:] = np.where(
        price_changes[1:] > 0,
        sample_data['volume'][1:],
        np.where(
            price_changes[1:] < 0,
            -sample_data['volume'][1:],
            0
        )
    ).cumsum()
    
    pd.testing.assert_series_equal(
        result.values['obv'],
        expected_obv,
        check_names=False
    )
    
    # Check metadata
    assert result.metadata['source'] == 'close'
    assert result.metadata['signal_period'] == 20


def test_obv_with_different_sources(sample_data):
    """Test OBV with different price sources."""
    sources = ['open', 'high', 'low', 'close']
    for source in sources:
        obv = OnBalanceVolume(OBVParams(source=source))
        result = obv.calculate(sample_data)
        assert result.metadata['source'] == source


def test_obv_with_gaps(sample_data):
    """Test OBV with missing data."""
    # Create gaps in data
    sample_data.loc[10:15, ['close', 'volume']] = np.nan
    
    obv = OnBalanceVolume(OBVParams(min_periods=5))
    result = obv.calculate(sample_data)
    
    # Check that gaps are propagated
    assert result.values.loc[10:15, 'obv'].isna().all()
    assert result.values.loc[10:15, 'signal'].isna().all()
    assert result.values.loc[10:15, 'histogram'].isna().all()


def test_volume_profile_params():
    """Test Volume Profile parameters."""
    params = VolumeProfileParams()
    assert params.price_source == 'close'
    assert params.n_bins == 24
    assert params.window is None
    assert params.min_periods is None
    
    params = VolumeProfileParams(
        price_source='high',
        n_bins=12,
        window=50,
        min_periods=10
    )
    assert params.price_source == 'high'
    assert params.n_bins == 12
    assert params.window == 50
    assert params.min_periods == 10


def test_volume_profile_calculation(sample_data):
    """Test Volume Profile calculation."""
    vp = VolumeProfile()
    result = vp.calculate(sample_data)
    
    # Check result structure
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['poc', 'vah', 'val', 'volume_total'])
    
    # Check bin columns exist
    assert all(f'vol_bin_{i}' in result.values.columns for i in range(vp.params.n_bins))
    assert all(f'price_bin_{i}' in result.values.columns for i in range(vp.params.n_bins))
    
    # Check value area relationships
    assert (result.values['vah'] >= result.values['poc']).all()
    assert (result.values['poc'] >= result.values['val']).all()
    
    # Check metadata
    assert result.metadata['price_source'] == 'close'
    assert result.metadata['n_bins'] == 24
    assert result.metadata['window'] is None


def test_volume_profile_with_window(sample_data):
    """Test Volume Profile with rolling window."""
    window = 20
    vp = VolumeProfile(VolumeProfileParams(window=window))
    result = vp.calculate(sample_data)
    
    # Check that initial values are NaN
    assert result.values.iloc[:window-1].isna().all().all()
    
    # Check that later values are calculated
    assert not result.values.iloc[window:].isna().all().all()
    
    # Check metadata
    assert result.metadata['window'] == window


def test_volume_profile_with_different_sources(sample_data):
    """Test Volume Profile with different price sources."""
    sources = ['open', 'high', 'low', 'close']
    for source in sources:
        vp = VolumeProfile(VolumeProfileParams(price_source=source))
        result = vp.calculate(sample_data)
        assert result.metadata['price_source'] == source


def test_volume_profile_with_gaps(sample_data):
    """Test Volume Profile with missing data."""
    # Create gaps in data
    sample_data.loc[10:15, ['close', 'volume']] = np.nan
    
    vp = VolumeProfile(VolumeProfileParams(min_periods=5))
    result = vp.calculate(sample_data)
    
    # Check that gaps are propagated
    assert result.values.loc[10:15].isna().all().all()


def test_volume_profile_bin_distribution(sample_data):
    """Test Volume Profile bin distribution."""
    n_bins = 10
    vp = VolumeProfile(VolumeProfileParams(n_bins=n_bins))
    result = vp.calculate(sample_data)
    
    # Check number of bins
    assert sum(1 for col in result.values.columns if col.startswith('vol_bin_')) == n_bins
    assert sum(1 for col in result.values.columns if col.startswith('price_bin_')) == n_bins
    
    # Check bin relationships
    for i in range(n_bins-1):
        assert (result.values[f'price_bin_{i}'] >= result.values[f'price_bin_{i+1}']).all()
    
    # Check volume distribution sums to total
    vol_bins = [result.values[f'vol_bin_{i}'] for i in range(n_bins)]
    total_vol = pd.concat(vol_bins, axis=1).sum(axis=1)
    pd.testing.assert_series_equal(
        total_vol,
        result.values['volume_total'],
        check_names=False
    ) 