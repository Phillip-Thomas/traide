"""Unit tests for base indicator framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.marketdata.indicators.base import (
    BaseIndicator,
    IndicatorParams,
    IndicatorStyle,
    IndicatorResult,
    IndicatorRegistry,
    register_indicator
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


@dataclass
class MockParams(IndicatorParams):
    """Mock indicator parameters."""
    period: int = 14
    source: str = 'close'


@register_indicator('mock')
class MockIndicator(BaseIndicator):
    """Mock indicator for testing."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MockParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate mock values."""
        params = self.params
        values = data[params.source].rolling(params.period).mean()
        return IndicatorResult(values=values)


def test_indicator_params():
    """Test indicator parameters."""
    params = MockParams(period=20, source='high')
    assert params.period == 20
    assert params.source == 'high'


def test_indicator_style():
    """Test indicator style."""
    style = IndicatorStyle(
        color=(1.0, 0.0, 0.0),
        line_width=2.0,
        line_style='dashed',
        show_points=True,
        point_size=6.0,
        z_index=1,
        visible=True,
        label='Test'
    )
    assert style.color == (1.0, 0.0, 0.0)
    assert style.line_width == 2.0
    assert style.line_style == 'dashed'
    assert style.show_points
    assert style.point_size == 6.0
    assert style.z_index == 1
    assert style.visible
    assert style.label == 'Test'


def test_indicator_result(sample_data):
    """Test indicator result."""
    values = sample_data['close'].rolling(14).mean()
    result = IndicatorResult(values=values)
    assert isinstance(result.values, pd.Series)
    assert isinstance(result.metadata, dict)
    assert isinstance(result.timestamp, pd.Timestamp)


def test_indicator_initialization():
    """Test indicator initialization."""
    # Default initialization
    indicator = MockIndicator()
    assert isinstance(indicator.params, MockParams)
    assert isinstance(indicator.style, IndicatorStyle)
    assert indicator.params.period == 14
    assert indicator.params.source == 'close'
    
    # Custom initialization
    params = MockParams(period=20, source='high')
    style = IndicatorStyle(color=(1.0, 0.0, 0.0))
    indicator = MockIndicator(params, style)
    assert indicator.params.period == 20
    assert indicator.params.source == 'high'
    assert indicator.style.color == (1.0, 0.0, 0.0)


def test_indicator_calculation(sample_data):
    """Test indicator calculation."""
    indicator = MockIndicator()
    result = indicator.calculate(sample_data)
    
    assert isinstance(result, IndicatorResult)
    assert len(result.values) == len(sample_data)
    assert isinstance(result.values, pd.Series)
    assert result.values.equals(
        sample_data['close'].rolling(14).mean()
    )


def test_indicator_caching(sample_data):
    """Test indicator result caching."""
    indicator = MockIndicator()
    
    # First calculation
    result1 = indicator.calculate(sample_data)
    cache_size1 = len(indicator._cache)
    
    # Second calculation (should use cache)
    result2 = indicator.calculate(sample_data)
    cache_size2 = len(indicator._cache)
    
    assert cache_size1 == cache_size2
    assert result1.values.equals(result2.values)
    assert result1.timestamp == result2.timestamp


def test_indicator_cache_invalidation(sample_data):
    """Test cache invalidation."""
    indicator = MockIndicator()
    
    # Calculate with original data
    result1 = indicator.calculate(sample_data)
    
    # Modify data
    modified_data = sample_data.copy()
    modified_data['close'] = modified_data['close'] * 1.1
    
    # Calculate with modified data
    result2 = indicator.calculate(modified_data)
    
    assert not result1.values.equals(result2.values)


def test_indicator_cache_cleanup(sample_data):
    """Test cache cleanup."""
    indicator = MockIndicator()
    
    # Generate multiple cache entries
    for i in range(15):
        data = sample_data.copy()
        data['close'] = data['close'] * (1 + i * 0.1)
        indicator.calculate(data)
    
    # Check cache size limit
    assert len(indicator._cache) <= 10


def test_indicator_registry():
    """Test indicator registry."""
    # Check registration
    assert 'mock' in IndicatorRegistry.list_indicators()
    
    # Create indicator
    indicator = IndicatorRegistry.create('mock')
    assert isinstance(indicator, MockIndicator)
    
    # Create with params
    params = MockParams(period=20)
    indicator = IndicatorRegistry.create('mock', params)
    assert indicator.params.period == 20
    
    # Test invalid indicator
    with pytest.raises(ValueError):
        IndicatorRegistry.create('invalid')


def test_register_decorator():
    """Test register decorator."""
    @register_indicator('test')
    class TestIndicator(MockIndicator):
        pass
    
    assert 'test' in IndicatorRegistry.list_indicators()
    indicator = IndicatorRegistry.create('test')
    assert isinstance(indicator, TestIndicator)


def test_invalid_registration():
    """Test invalid indicator registration."""
    class InvalidIndicator:
        pass
    
    with pytest.raises(ValueError):
        IndicatorRegistry.register('invalid', InvalidIndicator) 