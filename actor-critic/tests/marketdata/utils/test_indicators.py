"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.marketdata.utils.indicators import (
    Indicator,
    MovingAverage,
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage,
    RelativeStrengthIndex,
    MovingAverageConvergenceDivergence
)


class SimpleIndicator(Indicator):
    """Simple indicator for testing."""
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['close']
    
    def _get_required_columns(self) -> List[str]:
        return ['close']


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


@pytest.fixture
def rsi_data():
    """Create sample data for RSI testing."""
    timestamps = pd.date_range(
        start='2024-01-01',
        end='2024-01-02',
        freq='1min'
    )
    
    # Create a price series with known characteristics
    n_points = len(timestamps)
    closes = np.zeros(n_points)
    
    # First segment: Uptrend
    closes[:100] = np.linspace(100, 150, 100)
    
    # Second segment: Downtrend
    closes[100:200] = np.linspace(150, 80, 100)
    
    # Third segment: Sideways with volatility
    base = np.linspace(80, 85, 100)
    noise = np.random.normal(0, 2, 100)
    closes[200:300] = base + noise
    
    # Rest of the data: Random walk
    random_changes = np.random.normal(0, 1, n_points - 300)
    closes[300:] = 85 + np.cumsum(random_changes)
    
    return pd.DataFrame({
        'close': closes
    }, index=timestamps)


@pytest.fixture
def macd_data():
    """Create sample data for MACD testing."""
    timestamps = pd.date_range(
        start='2024-01-01',
        end='2024-01-02',
        freq='1min'
    )
    
    # Create a price series with known characteristics
    n_points = len(timestamps)
    closes = np.zeros(n_points)
    
    # First segment: Strong uptrend
    closes[:200] = np.linspace(100, 200, 200)
    
    # Second segment: Sideways consolidation
    base = np.linspace(200, 205, 200)
    noise = np.random.normal(0, 2, 200)
    closes[200:400] = base + noise
    
    # Third segment: Sharp downtrend
    closes[400:600] = np.linspace(205, 150, 200)
    
    # Fourth segment: Slow recovery
    closes[600:] = np.linspace(150, 170, n_points - 600)
    
    return pd.DataFrame({
        'close': closes
    }, index=timestamps)


def test_indicator_validation():
    """Test indicator parameter validation."""
    # Valid parameters
    indicator = SimpleIndicator(window=10)
    assert indicator.window == 10
    
    # Invalid window type
    with pytest.raises(ValueError, match="must be an integer"):
        SimpleIndicator(window=10.5)
    
    # Invalid window value
    with pytest.raises(ValueError, match="must be positive"):
        SimpleIndicator(window=0)


def test_moving_average_validation():
    """Test moving average parameter validation."""
    # Valid parameters
    ma = SimpleMovingAverage(window=10, column='close', min_periods=5)
    assert ma.window == 10
    assert ma.column == 'close'
    assert ma.min_periods == 5
    
    # Invalid column type
    with pytest.raises(ValueError, match="Column must be a string"):
        SimpleMovingAverage(column=123)
    
    # Invalid min_periods type
    with pytest.raises(ValueError, match="min_periods must be an integer"):
        SimpleMovingAverage(min_periods=5.5)
    
    # Invalid min_periods value
    with pytest.raises(ValueError, match="min_periods must be positive"):
        SimpleMovingAverage(min_periods=0)
    
    # min_periods > window
    with pytest.raises(ValueError, match="cannot be greater than window"):
        SimpleMovingAverage(window=5, min_periods=10)


def test_data_validation(sample_data):
    """Test input data validation."""
    indicator = SimpleIndicator()
    
    # Valid data
    indicator._validate_data(sample_data)
    
    # Invalid index type
    invalid_data = sample_data.reset_index()
    with pytest.raises(ValueError, match="must have DatetimeIndex"):
        indicator._validate_data(invalid_data)
    
    # Non-monotonic index
    invalid_data = sample_data.copy()
    invalid_data.index = invalid_data.index[::-1]
    with pytest.raises(ValueError, match="monotonically increasing"):
        indicator._validate_data(invalid_data)
    
    # Missing columns
    invalid_data = sample_data.drop('close', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        indicator._validate_data(invalid_data)


def test_simple_moving_average(sample_data):
    """Test Simple Moving Average calculation."""
    window = 5
    sma = SimpleMovingAverage(window=window)
    result = sma.calculate(sample_data)
    
    # Check basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert result.index.equals(sample_data.index)
    
    # Check calculation
    expected = sample_data['close'].rolling(window=window).mean()
    pd.testing.assert_series_equal(result, expected)
    
    # Check first values are NaN
    assert result.iloc[:window-1].isna().all()
    assert not result.iloc[window:].isna().any()


def test_exponential_moving_average(sample_data):
    """Test Exponential Moving Average calculation."""
    window = 5
    ema = ExponentialMovingAverage(window=window)
    result = ema.calculate(sample_data)
    
    # Check basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert result.index.equals(sample_data.index)
    
    # Check calculation
    expected = sample_data['close'].ewm(
        span=window,
        min_periods=window,
        adjust=False
    ).mean()
    pd.testing.assert_series_equal(result, expected)
    
    # Check first values are NaN
    assert result.iloc[:window-1].isna().all()
    assert not result.iloc[window:].isna().any()


def test_weighted_moving_average(sample_data):
    """Test Weighted Moving Average calculation."""
    window = 5
    wma = WeightedMovingAverage(window=window)
    result = wma.calculate(sample_data)
    
    # Check basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert result.index.equals(sample_data.index)
    
    # Check calculation for a specific point
    i = window + 5  # Some point after the initial window
    weights = np.arange(1, window + 1) / np.arange(1, window + 1).sum()
    window_values = sample_data['close'].iloc[i-window+1:i+1].values
    expected_value = np.sum(window_values * weights)
    assert abs(result.iloc[i] - expected_value) < 1e-10
    
    # Check first values are NaN
    assert result.iloc[:window-1].isna().all()
    assert not result.iloc[window:].isna().any()


def test_moving_average_min_periods(sample_data):
    """Test moving averages with custom min_periods."""
    window = 5
    min_periods = 3
    
    # Test SMA
    sma = SimpleMovingAverage(window=window, min_periods=min_periods)
    result = sma.calculate(sample_data)
    assert result.iloc[:min_periods-1].isna().all()
    assert not result.iloc[min_periods:].isna().any()
    
    # Test EMA
    ema = ExponentialMovingAverage(window=window, min_periods=min_periods)
    result = ema.calculate(sample_data)
    assert result.iloc[:min_periods-1].isna().all()
    assert not result.iloc[min_periods:].isna().any()
    
    # Test WMA
    wma = WeightedMovingAverage(window=window, min_periods=min_periods)
    result = wma.calculate(sample_data)
    assert result.iloc[:min_periods-1].isna().all()
    assert not result.iloc[min_periods:].isna().any()


def test_rsi_validation():
    """Test RSI parameter validation."""
    # Valid parameters
    rsi = RelativeStrengthIndex(window=14, column='close', min_periods=7)
    assert rsi.window == 14
    assert rsi.column == 'close'
    assert rsi.min_periods == 7
    
    # Invalid column type
    with pytest.raises(ValueError, match="Column must be a string"):
        RelativeStrengthIndex(column=123)
    
    # Invalid min_periods type
    with pytest.raises(ValueError, match="min_periods must be an integer"):
        RelativeStrengthIndex(min_periods=5.5)
    
    # Invalid min_periods value
    with pytest.raises(ValueError, match="min_periods must be positive"):
        RelativeStrengthIndex(min_periods=0)
    
    # min_periods > window
    with pytest.raises(ValueError, match="cannot be greater than window"):
        RelativeStrengthIndex(window=5, min_periods=10)


def test_rsi_calculation(rsi_data):
    """Test RSI calculation."""
    window = 14
    rsi = RelativeStrengthIndex(window=window)
    result = rsi.calculate(rsi_data)
    
    # Check basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(rsi_data)
    assert result.index.equals(rsi_data.index)
    assert result.name == f'RSI_{window}'
    
    # Check value bounds for non-NaN values
    valid_values = result.dropna()
    assert (valid_values >= 0).all() and (valid_values <= 100).all()
    
    # Check NaN handling
    assert result.iloc[:window-1].isna().all()
    assert not result.iloc[window:].isna().any()
    
    # Check trend detection
    # Uptrend should have high RSI
    uptrend_rsi = result[50:100].mean()
    assert uptrend_rsi > 70
    
    # Downtrend should have low RSI
    downtrend_rsi = result[150:200].mean()
    assert downtrend_rsi < 30
    
    # Sideways should have moderate RSI
    sideways_rsi = result[250:300].mean()
    assert 40 < sideways_rsi < 60


def test_rsi_min_periods(rsi_data):
    """Test RSI with custom min_periods."""
    window = 14
    min_periods = 7
    rsi = RelativeStrengthIndex(window=window, min_periods=min_periods)
    result = rsi.calculate(rsi_data)
    
    # Check NaN handling with min_periods
    assert result.iloc[:min_periods-1].isna().all()
    assert not result.iloc[min_periods:].isna().any()
    
    # Values should still be bounded
    valid_values = result.iloc[min_periods:]
    assert (valid_values >= 0).all() and (valid_values <= 100).all()


def test_macd_validation():
    """Test MACD parameter validation."""
    # Valid parameters
    macd = MovingAverageConvergenceDivergence(
        fast_window=12,
        slow_window=26,
        signal_window=9,
        column='close',
        min_periods=12
    )
    assert macd.fast_window == 12
    assert macd.slow_window == 26
    assert macd.signal_window == 9
    assert macd.column == 'close'
    assert macd.min_periods == 12
    
    # Invalid window types
    with pytest.raises(ValueError, match="must be integers"):
        MovingAverageConvergenceDivergence(fast_window=12.5)
    
    with pytest.raises(ValueError, match="must be integers"):
        MovingAverageConvergenceDivergence(slow_window=26.5)
    
    with pytest.raises(ValueError, match="must be integers"):
        MovingAverageConvergenceDivergence(signal_window=9.5)
    
    # Invalid window values
    with pytest.raises(ValueError, match="must be positive"):
        MovingAverageConvergenceDivergence(fast_window=0)
    
    with pytest.raises(ValueError, match="must be positive"):
        MovingAverageConvergenceDivergence(slow_window=0)
    
    with pytest.raises(ValueError, match="must be positive"):
        MovingAverageConvergenceDivergence(signal_window=0)
    
    # Fast window >= slow window
    with pytest.raises(ValueError, match="must be smaller than slow window"):
        MovingAverageConvergenceDivergence(fast_window=26, slow_window=26)
    
    # Invalid column type
    with pytest.raises(ValueError, match="Column must be a string"):
        MovingAverageConvergenceDivergence(column=123)
    
    # Invalid min_periods type
    with pytest.raises(ValueError, match="min_periods must be an integer"):
        MovingAverageConvergenceDivergence(min_periods=5.5)
    
    # Invalid min_periods value
    with pytest.raises(ValueError, match="min_periods must be positive"):
        MovingAverageConvergenceDivergence(min_periods=0)
    
    # min_periods > slow_window
    with pytest.raises(ValueError, match="cannot be greater than slow window"):
        MovingAverageConvergenceDivergence(
            slow_window=26,
            min_periods=30
        )


def test_macd_calculation(macd_data):
    """Test MACD calculation."""
    macd = MovingAverageConvergenceDivergence()
    result = macd.calculate(macd_data)
    
    # Check basic properties
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(macd_data)
    assert result.index.equals(macd_data.index)
    
    # Check column names
    expected_columns = [
        'MACD_12_26',
        'Signal_9',
        'Histogram'
    ]
    assert all(col in result.columns for col in expected_columns)
    
    # Check NaN handling for each component
    assert result.iloc[:macd.min_periods-1].isna().all().all()
    
    # Check that values after min_periods are not NaN
    valid_data = result.iloc[macd.min_periods+macd.signal_window-1:]
    assert not valid_data.isna().any().any()
    
    # Check trend detection
    # Strong uptrend should have positive MACD and histogram
    uptrend_start = macd_data.index[100]
    uptrend_end = macd_data.index[150]
    uptrend_mask = (result.index >= uptrend_start) & (result.index <= uptrend_end)
    assert (result.loc[uptrend_mask, 'MACD_12_26'] > 0).all()
    assert (result.loc[uptrend_mask, 'Histogram'] > 0).mean() > 0.8  # 80% positive
    
    # Sideways should have MACD close to zero
    sideways_start = macd_data.index[300]
    sideways_end = macd_data.index[350]
    sideways_mask = (result.index >= sideways_start) & (result.index <= sideways_end)
    assert abs(result.loc[sideways_mask, 'MACD_12_26'].mean()) < 1.0
    
    # Sharp downtrend should have negative MACD and histogram
    downtrend_start = macd_data.index[450]
    downtrend_end = macd_data.index[500]
    downtrend_mask = (result.index >= downtrend_start) & (result.index <= downtrend_end)
    assert (result.loc[downtrend_mask, 'MACD_12_26'] < 0).all()
    assert (result.loc[downtrend_mask, 'Histogram'] < 0).mean() > 0.8  # 80% negative


def test_macd_min_periods(macd_data):
    """Test MACD with custom min_periods."""
    min_periods = 15
    macd = MovingAverageConvergenceDivergence(min_periods=min_periods)
    result = macd.calculate(macd_data)
    
    # Check NaN handling with min_periods for each component
    assert result.iloc[:min_periods-1].isna().all().all()
    
    # Check that values after min_periods + signal_window are not NaN
    valid_data = result.iloc[min_periods+macd.signal_window-1:]
    assert not valid_data.isna().any().any() 