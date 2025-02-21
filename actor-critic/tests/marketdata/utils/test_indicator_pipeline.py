"""Integration tests for indicator pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.marketdata.utils.indicators import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    RelativeStrengthIndex
)
from src.marketdata.utils.volatility import (
    BollingerBands,
    AverageTrueRange,
    Volatility
)
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


def test_trend_indicators_pipeline(sample_data):
    """Test pipeline of trend indicators."""
    # Create indicators
    sma = SimpleMovingAverage(window=20)
    ema = ExponentialMovingAverage(window=20)
    rsi = RelativeStrengthIndex(window=14)
    
    # Calculate indicators
    sma_result = sma.calculate(sample_data)
    ema_result = ema.calculate(sample_data)
    rsi_result = rsi.calculate(sample_data)
    
    # Combine results
    result = pd.concat([
        sma_result,
        ema_result,
        rsi_result
    ], axis=1)
    
    # Check relationships (ignoring NaN values)
    valid_sma = ~result['simple_ma'].isna()
    valid_ema = ~result['exp_ma'].isna()
    valid_rsi = ~result['rsi'].isna()
    
    assert (result['simple_ma'][valid_sma] >= sample_data['low'][valid_sma]).all()
    assert (result['simple_ma'][valid_sma] <= sample_data['high'][valid_sma]).all()
    assert (result['exp_ma'][valid_ema] >= sample_data['low'][valid_ema]).all()
    assert (result['exp_ma'][valid_ema] <= sample_data['high'][valid_ema]).all()
    assert (result['rsi'][valid_rsi] >= 0).all()
    assert (result['rsi'][valid_rsi] <= 100).all()


def test_volatility_indicators_pipeline(sample_data):
    """Test pipeline of volatility indicators."""
    # Create indicators
    bb = BollingerBands(window=20)
    atr = AverageTrueRange(window=14)
    vol = Volatility(window=20)
    
    # Calculate indicators
    bb_result = bb.calculate(sample_data)
    atr_result = atr.calculate(sample_data)
    vol_result = vol.calculate(sample_data)
    
    # Combine results
    result = pd.concat([
        bb_result,
        atr_result,
        vol_result
    ], axis=1)
    
    # Check relationships (ignoring NaN values)
    valid_bb = ~result['bb_middle'].isna()
    valid_atr = ~result['atr'].isna()
    valid_vol = ~result['volatility'].isna()
    
    assert (result['bb_upper'][valid_bb] >= result['bb_middle'][valid_bb]).all()
    assert (result['bb_middle'][valid_bb] >= result['bb_lower'][valid_bb]).all()
    assert (result['atr'][valid_atr] >= 0).all()
    assert (result['volatility'][valid_vol] >= 0).all()


def test_volume_indicators_pipeline(sample_data):
    """Test pipeline of volume indicators."""
    # Create indicators
    obv = OnBalanceVolume(smooth_window=20)
    vp = VolumeProfile(num_bins=10)
    vwap = VolumeWeightedAveragePrice()
    
    # Calculate indicators
    obv_result = obv.calculate(sample_data)
    vp_result = vp.calculate(sample_data)
    vwap_result = vwap.calculate(sample_data)
    
    # Combine results
    result = pd.concat([
        obv_result,
        vp_result,
        vwap_result
    ], axis=1)
    
    # Check relationships
    assert not result['obv'].isna().any()
    assert not result['poc'].isna().any()
    assert not result['vwap'].isna().any()
    assert (result['vwap'] >= sample_data['low']).all()
    assert (result['vwap'] <= sample_data['high']).all()


def test_full_indicator_pipeline(sample_data):
    """Test full pipeline with all indicators."""
    # Create indicators
    indicators = [
        SimpleMovingAverage(window=20),
        ExponentialMovingAverage(window=20),
        RelativeStrengthIndex(window=14),
        BollingerBands(window=20),
        AverageTrueRange(window=14),
        Volatility(window=20),
        OnBalanceVolume(smooth_window=20),
        VolumeProfile(num_bins=10),
        VolumeWeightedAveragePrice()
    ]
    
    # Calculate all indicators
    results = []
    for indicator in indicators:
        result = indicator.calculate(sample_data)
        results.append(result)
    
    # Combine all results
    combined = pd.concat(results, axis=1)
    
    # Basic validation
    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == len(sample_data)
    assert not combined.isna().all().any()  # No column should be all NaN


def test_indicator_pipeline_edge_cases(sample_data):
    """Test indicator pipeline with edge cases."""
    # Create a copy with some edge cases
    edge_data = sample_data.copy()
    
    # Add some edge cases
    edge_data.iloc[0] = np.nan  # First row NaN
    edge_data.iloc[-1] = 0  # Last row zero
    edge_data.iloc[50] = float('inf')  # Middle row infinity
    
    # Create indicators with small windows to test edge handling
    indicators = [
        SimpleMovingAverage(window=3),
        ExponentialMovingAverage(window=3),
        RelativeStrengthIndex(window=3),
        BollingerBands(window=3),
        AverageTrueRange(window=3),
        Volatility(window=3),
        OnBalanceVolume(smooth_window=3),
        VolumeProfile(num_bins=5),
        VolumeWeightedAveragePrice()
    ]
    
    # Calculate all indicators
    results = []
    for indicator in indicators:
        try:
            result = indicator.calculate(edge_data)
            results.append(result)
        except Exception as e:
            pytest.fail(f"Indicator {indicator.__class__.__name__} failed: {str(e)}")
    
    # Combine all results
    combined = pd.concat(results, axis=1)
    
    # Check that we have some valid results after the edge cases
    assert not combined.iloc[10:].isna().all().all()


def test_indicator_pipeline_performance(sample_data):
    """Test performance of indicator pipeline with larger dataset."""
    # Create larger dataset by extending the index
    large_data = sample_data.copy()
    for i in range(9):  # Add 9 more copies
        new_data = sample_data.copy()
        new_data.index = new_data.index + pd.Timedelta(days=i+1)
        large_data = pd.concat([large_data, new_data])
    
    large_data = large_data.sort_index()  # Ensure index is sorted
    
    # Create indicators
    indicators = [
        SimpleMovingAverage(window=20),
        ExponentialMovingAverage(window=20),
        RelativeStrengthIndex(window=14),
        BollingerBands(window=20),
        AverageTrueRange(window=14),
        Volatility(window=20),
        OnBalanceVolume(smooth_window=20),
        VolumeProfile(num_bins=10),
        VolumeWeightedAveragePrice()
    ]
    
    # Time the calculations
    import time
    start_time = time.time()
    
    # Calculate all indicators
    results = []
    for indicator in indicators:
        result = indicator.calculate(large_data)
        results.append(result)
    
    # Combine all results
    combined = pd.concat(results, axis=1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Check performance (should be reasonable for 1000 rows)
    assert execution_time < 5.0  # Should complete within 5 seconds 