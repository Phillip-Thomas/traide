"""Tests for volume-based technical indicators."""

import pytest
import numpy as np
import pandas as pd

from src.marketdata.utils.volume import (
    OnBalanceVolume,
    VolumeProfile,
    VolumeWeightedAveragePrice
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducibility
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


def test_obv_basic(sample_data):
    """Test basic OBV calculation."""
    obv = OnBalanceVolume()
    result = obv.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'obv' in result.columns
    assert len(result) == len(sample_data)
    assert not result['obv'].isna().any()


def test_obv_with_smoothing(sample_data):
    """Test OBV with smoothing window."""
    smooth_window = 5
    obv = OnBalanceVolume(smooth_window=smooth_window)
    result = obv.calculate(sample_data)
    
    assert 'obv' in result.columns
    assert 'obv_smooth' in result.columns
    assert not result['obv_smooth'].isna().all()
    
    # First value should be equal (no smoothing possible)
    assert np.isclose(result['obv'].iloc[0], result['obv_smooth'].iloc[0])


def test_volume_profile_basic(sample_data):
    """Test basic Volume Profile calculation."""
    vp = VolumeProfile(num_bins=10)
    result = vp.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'poc' in result.columns
    assert 'value_area_low' in result.columns
    assert 'value_area_high' in result.columns
    assert 'vol_profile_0' in result.columns
    
    # Check value area relationships
    assert (result['value_area_high'] >= result['value_area_low']).all()
    assert (result['poc'] >= result['value_area_low']).all()
    assert (result['poc'] <= result['value_area_high']).all()


def test_volume_profile_with_window(sample_data):
    """Test Volume Profile with rolling window."""
    window = 20
    vp = VolumeProfile(num_bins=10, window=window)
    result = vp.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    
    # First row should have valid values (using first point)
    assert not result.iloc[0].isna().any()
    
    # Check value area relationships
    assert (result['value_area_high'] >= result['value_area_low']).all()
    assert (result['poc'] >= result['value_area_low']).all()
    assert (result['poc'] <= result['value_area_high']).all()


def test_volume_profile_price_types(sample_data):
    """Test Volume Profile with different price types."""
    # Test close price
    vp_close = VolumeProfile(price_type='close')
    result_close = vp_close.calculate(sample_data)
    
    # Test typical price
    vp_typical = VolumeProfile(price_type='typical')
    result_typical = vp_typical.calculate(sample_data)
    
    # Test HLC3
    vp_hlc3 = VolumeProfile(price_type='hlc3')
    result_hlc3 = vp_hlc3.calculate(sample_data)
    
    # All methods should produce valid results
    assert not result_close['poc'].isna().any()
    assert not result_typical['poc'].isna().any()
    assert not result_hlc3['poc'].isna().any()


def test_volume_profile_invalid_price_type():
    """Test Volume Profile with invalid price type."""
    with pytest.raises(ValueError):
        VolumeProfile(price_type='invalid')


def test_vwap_basic(sample_data):
    """Test basic VWAP calculation."""
    vwap = VolumeWeightedAveragePrice()
    result = vwap.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'vwap' in result.columns
    assert 'cum_volume' in result.columns
    assert not result['vwap'].isna().any()
    assert not result['cum_volume'].isna().any()
    
    # VWAP should be within price range
    assert (result['vwap'] >= sample_data['low']).all()
    assert (result['vwap'] <= sample_data['high']).all()


def test_vwap_anchors(sample_data):
    """Test VWAP with different anchors."""
    # Test daily anchor
    vwap_day = VolumeWeightedAveragePrice(anchor='day')
    result_day = vwap_day.calculate(sample_data)
    
    # Test weekly anchor
    vwap_week = VolumeWeightedAveragePrice(anchor='week')
    result_week = vwap_week.calculate(sample_data)
    
    # Test monthly anchor
    vwap_month = VolumeWeightedAveragePrice(anchor='month')
    result_month = vwap_month.calculate(sample_data)
    
    # All methods should produce valid results
    assert not result_day['vwap'].isna().any()
    assert not result_week['vwap'].isna().any()
    assert not result_month['vwap'].isna().any()


def test_vwap_price_types(sample_data):
    """Test VWAP with different price types."""
    # Test typical price
    vwap_typical = VolumeWeightedAveragePrice(price_type='typical')
    result_typical = vwap_typical.calculate(sample_data)
    
    # Test close price
    vwap_close = VolumeWeightedAveragePrice(price_type='close')
    result_close = vwap_close.calculate(sample_data)
    
    # Test HLC3
    vwap_hlc3 = VolumeWeightedAveragePrice(price_type='hlc3')
    result_hlc3 = vwap_hlc3.calculate(sample_data)
    
    # All methods should produce valid results
    assert not result_typical['vwap'].isna().any()
    assert not result_close['vwap'].isna().any()
    assert not result_hlc3['vwap'].isna().any()


def test_vwap_invalid_params():
    """Test VWAP with invalid parameters."""
    # Test invalid anchor
    with pytest.raises(ValueError):
        VolumeWeightedAveragePrice(anchor='invalid')
    
    # Test invalid price type
    with pytest.raises(ValueError):
        VolumeWeightedAveragePrice(price_type='invalid') 