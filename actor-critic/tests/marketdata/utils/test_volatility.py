"""Tests for volatility-based technical indicators."""

import pytest
import numpy as np
import pandas as pd

from src.marketdata.utils.volatility import (
    BollingerBands,
    AverageTrueRange,
    Volatility
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1d')
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


def test_bollinger_bands_basic(sample_data):
    """Test basic Bollinger Bands calculation."""
    bb = BollingerBands(window=20, num_std=2.0)
    result = bb.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'bb_middle' in result.columns
    assert 'bb_upper' in result.columns
    assert 'bb_lower' in result.columns
    assert 'bb_width' in result.columns
    assert 'bb_percent' in result.columns
    
    # Check relationships (ignoring NaN values)
    valid_mask = ~result['bb_middle'].isna()
    assert (result['bb_upper'][valid_mask] >= result['bb_middle'][valid_mask]).all()
    assert (result['bb_middle'][valid_mask] >= result['bb_lower'][valid_mask]).all()
    assert (result['bb_width'][valid_mask] >= 0).all()
    assert ((result['bb_percent'][valid_mask] >= 0) & (result['bb_percent'][valid_mask] <= 1)).all()


def test_bollinger_bands_custom_params(sample_data):
    """Test Bollinger Bands with custom parameters."""
    bb = BollingerBands(window=10, num_std=3.0, min_periods=5)
    result = bb.calculate(sample_data, price_col='close')
    
    # First 4 rows should be NaN (min_periods=5)
    assert result.iloc[:4].isna().all().all()
    
    # Check band width increases with higher num_std
    bb2 = BollingerBands(window=10, num_std=2.0)
    result2 = bb2.calculate(sample_data)
    
    # Compare only valid values
    valid_mask = ~(result['bb_width'].isna() | result2['bb_width'].isna())
    assert (result['bb_width'][valid_mask] >= result2['bb_width'][valid_mask]).all()


def test_atr_basic(sample_data):
    """Test basic ATR calculation."""
    atr = AverageTrueRange(window=14)
    result = atr.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'tr' in result.columns
    assert 'atr' in result.columns
    assert 'atr_percent' in result.columns
    
    # Check relationships (ignoring NaN values)
    valid_tr = ~result['tr'].isna()
    valid_atr = ~result['atr'].isna()
    valid_atr_pct = ~result['atr_percent'].isna()
    
    assert (result['tr'][valid_tr] >= 0).all()
    assert (result['atr'][valid_atr] >= 0).all()
    assert (result['atr_percent'][valid_atr_pct] >= 0).all()


def test_atr_smoothing_methods(sample_data):
    """Test different ATR smoothing methods."""
    # Test SMA
    atr_sma = AverageTrueRange(window=14, smoothing='sma')
    result_sma = atr_sma.calculate(sample_data)
    
    # Test EMA
    atr_ema = AverageTrueRange(window=14, smoothing='ema')
    result_ema = atr_ema.calculate(sample_data)
    
    # Test RMA
    atr_rma = AverageTrueRange(window=14, smoothing='rma')
    result_rma = atr_rma.calculate(sample_data)
    
    # All methods should produce valid results
    assert not result_sma['atr'].isna().all()
    assert not result_ema['atr'].isna().all()
    assert not result_rma['atr'].isna().all()


def test_atr_invalid_smoothing():
    """Test ATR with invalid smoothing method."""
    with pytest.raises(ValueError):
        AverageTrueRange(smoothing='invalid')


def test_volatility_basic(sample_data):
    """Test basic volatility calculation."""
    vol = Volatility(window=20)
    result = vol.calculate(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'volatility' in result.columns
    assert 'parkinson_volatility' in result.columns
    assert 'returns' in result.columns
    
    # Check relationships (ignoring NaN values)
    valid_vol = ~result['volatility'].isna()
    valid_park = ~result['parkinson_volatility'].isna()
    
    assert (result['volatility'][valid_vol] >= 0).all()
    assert (result['parkinson_volatility'][valid_park] >= 0).all()


def test_volatility_price_types(sample_data):
    """Test volatility with different price types."""
    # Test close price
    vol_close = Volatility(window=20, price_type='close')
    result_close = vol_close.calculate(sample_data)
    
    # Test typical price
    vol_typical = Volatility(window=20, price_type='typical')
    result_typical = vol_typical.calculate(sample_data)
    
    # Test HLC3
    vol_hlc3 = Volatility(window=20, price_type='hlc3')
    result_hlc3 = vol_hlc3.calculate(sample_data)
    
    # All methods should produce valid results
    assert not result_close['volatility'].isna().all()
    assert not result_typical['volatility'].isna().all()
    assert not result_hlc3['volatility'].isna().all()


def test_volatility_invalid_price_type():
    """Test volatility with invalid price type."""
    with pytest.raises(ValueError):
        Volatility(price_type='invalid')


def test_volatility_trading_periods(sample_data):
    """Test volatility with different trading periods."""
    # Daily volatility (252 trading days)
    vol_daily = Volatility(trading_periods=252)
    result_daily = vol_daily.calculate(sample_data)
    
    # Weekly volatility (52 weeks)
    vol_weekly = Volatility(trading_periods=52)
    result_weekly = vol_weekly.calculate(sample_data)
    
    # Daily volatility should be higher than weekly (ignoring NaN values)
    valid_mask = ~(result_daily['volatility'].isna() | result_weekly['volatility'].isna())
    assert (result_daily['volatility'][valid_mask] > result_weekly['volatility'][valid_mask]).all()


def test_indicator_min_periods(sample_data):
    """Test min_periods parameter for all indicators."""
    min_periods = 5
    
    # Bollinger Bands
    bb = BollingerBands(window=10, min_periods=min_periods)
    bb_result = bb.calculate(sample_data)
    assert bb_result.iloc[:min_periods-1].isna().all().all()
    assert not bb_result.iloc[min_periods:].isna().all().all()
    
    # ATR
    atr = AverageTrueRange(window=10, min_periods=min_periods)
    atr_result = atr.calculate(sample_data)
    assert atr_result.iloc[:min_periods-1].isna().all().all()
    assert not atr_result.iloc[min_periods:].isna().all().all()
    
    # Volatility
    vol = Volatility(window=10, min_periods=min_periods)
    vol_result = vol.calculate(sample_data)
    assert vol_result.iloc[:min_periods-1].isna().all().all()
    assert not vol_result.iloc[min_periods:].isna().all().all() 