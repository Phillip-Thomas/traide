"""Unit tests for histogram overlay."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.visualization.overlays.histogram import (
    HistogramStyle,
    HistogramOverlay
)
from src.marketdata.visualization.components import (
    BaseChart,
    ChartDimensions,
    ChartScales
)
from src.marketdata.indicators import MovingAverageConvergenceDivergence as MACD
from tests.marketdata.visualization.overlays.test_base import MockRenderer


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
    return pd.DataFrame(data)


@pytest.fixture
def chart():
    """Create a chart for testing."""
    dimensions = ChartDimensions(
        width=800,
        height=600,
        margin_top=20,
        margin_right=50,
        margin_bottom=30,
        margin_left=50
    )
    scales = ChartScales()
    return BaseChart(dimensions, scales)


@pytest.fixture
def renderer():
    """Create a mock renderer for testing."""
    return MockRenderer()


def test_histogram_style():
    """Test histogram style configuration."""
    style = HistogramStyle(
        color=(1.0, 0.0, 0.0),
        line_width=2.0,
        line_style='dashed',
        opacity=0.8,
        show_points=True,
        point_size=6.0,
        z_index=1,
        visible=True,
        label='Test',
        value_column='histogram',
        positive_color=(0.0, 1.0, 0.0),
        negative_color=(1.0, 0.0, 0.0),
        bar_width=0.8,
        zero_line_color=(0.5, 0.5, 0.5),
        zero_line_width=1.0,
        zero_line_style='dotted'
    )
    
    assert style.color == (1.0, 0.0, 0.0)
    assert style.line_width == 2.0
    assert style.line_style == 'dashed'
    assert style.opacity == 0.8
    assert style.show_points
    assert style.point_size == 6.0
    assert style.z_index == 1
    assert style.visible
    assert style.label == 'Test'
    assert style.value_column == 'histogram'
    assert style.positive_color == (0.0, 1.0, 0.0)
    assert style.negative_color == (1.0, 0.0, 0.0)
    assert style.bar_width == 0.8
    assert style.zero_line_color == (0.5, 0.5, 0.5)
    assert style.zero_line_width == 1.0
    assert style.zero_line_style == 'dotted'


def test_histogram_style_default_colors():
    """Test histogram style default colors."""
    style = HistogramStyle()
    assert style.positive_color == (0.0, 0.8, 0.0)  # Green
    assert style.negative_color == (0.8, 0.0, 0.0)  # Red
    assert style.zero_line_color == (0.5, 0.5, 0.5)  # Gray


def test_histogram_overlay_initialization():
    """Test histogram overlay initialization."""
    indicator = MACD()
    style = HistogramStyle(color=(1.0, 0.0, 0.0))
    overlay = HistogramOverlay(indicator, style)
    
    assert overlay.indicator == indicator
    assert overlay.style == style
    assert overlay.visible
    assert overlay.z_index == 0
    assert overlay._result is None


def test_histogram_overlay_get_values_series():
    """Test histogram overlay value retrieval from Series."""
    indicator = MACD()
    style = HistogramStyle()
    overlay = HistogramOverlay(indicator, style)
    
    # Create test result Series
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.Series(np.random.uniform(-1, 1, 10), index=dates)
    
    overlay._result = result
    values = overlay._get_values()
    
    assert isinstance(values, pd.Series)
    assert len(values) == len(result)
    assert (values == result).all()


def test_histogram_overlay_get_values_dataframe():
    """Test histogram overlay value retrieval from DataFrame."""
    indicator = MACD()
    style = HistogramStyle(value_column='histogram')
    overlay = HistogramOverlay(indicator, style)
    
    # Create test result DataFrame
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.DataFrame({
        'histogram': np.random.uniform(-1, 1, 10),
        'other': np.random.uniform(-1, 1, 10)
    }, index=dates)
    
    overlay._result = result
    values = overlay._get_values()
    
    assert isinstance(values, pd.Series)
    assert len(values) == len(result)
    assert (values == result['histogram']).all()


def test_histogram_overlay_get_values_missing_column():
    """Test histogram overlay value retrieval with missing column."""
    indicator = MACD()
    style = HistogramStyle(value_column='missing')
    overlay = HistogramOverlay(indicator, style)
    
    # Create test result DataFrame without required column
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.DataFrame({
        'histogram': np.random.uniform(-1, 1, 10)
    }, index=dates)
    
    overlay._result = result
    with pytest.raises(ValueError, match="Result DataFrame missing required column"):
        overlay._get_values()


def test_histogram_overlay_render_bars(chart, renderer):
    """Test histogram overlay bar rendering."""
    indicator = MACD()
    style = HistogramStyle(
        positive_color=(0.0, 1.0, 0.0),
        negative_color=(1.0, 0.0, 0.0),
        bar_width=0.8
    )
    overlay = HistogramOverlay(indicator, style)
    
    # Create test data with both positive and negative values
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([1.0, -1.0, 0.5, -0.5, -0.1, 1.5, -1.5, 0.8, -0.8, 1.2], index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check for rectangle calls
    rect_calls = [call for call in renderer.calls if call[0] == 'draw_rect']
    assert len(rect_calls) == 10  # One bar per data point
    
    # Check that bars use correct colors
    positive_bars = [call for call in rect_calls if call[5] == style.positive_color]
    negative_bars = [call for call in rect_calls if call[5] == style.negative_color]
    assert len(positive_bars) == 5  # Values >= 0
    assert len(negative_bars) == 5  # Values < 0


def test_histogram_overlay_render_zero_line(chart, renderer):
    """Test histogram overlay zero line rendering."""
    indicator = MACD()
    style = HistogramStyle(
        zero_line_color=(0.5, 0.5, 0.5),
        zero_line_width=2.0,
        zero_line_style='dashed'
    )
    overlay = HistogramOverlay(indicator, style)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series(np.random.uniform(-1, 1, 10), index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check for zero line
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) == 1  # One zero line
    zero_line = line_calls[0]
    assert zero_line[3] == style.zero_line_color  # Check color
    assert zero_line[4] == style.zero_line_width  # Check width
    assert zero_line[5] == style.zero_line_style  # Check style


def test_histogram_overlay_render_with_gaps(chart, renderer):
    """Test histogram overlay rendering with gaps in data."""
    indicator = MACD()
    style = HistogramStyle()
    overlay = HistogramOverlay(indicator, style)
    
    # Create data with gaps
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([1.0, np.nan, 0.5, -0.5, np.nan, np.nan, -1.5, 0.8, -0.8, 1.2], index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that we have the correct number of bars
    # (should skip NaN values)
    rect_calls = [call for call in renderer.calls if call[0] == 'draw_rect']
    assert len(rect_calls) == 7  # Number of non-NaN values


def test_histogram_overlay_render_empty_data(chart, renderer):
    """Test histogram overlay rendering with empty data."""
    indicator = MACD()
    style = HistogramStyle()
    overlay = HistogramOverlay(indicator, style)
    
    # Create empty data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([np.nan] * 10, index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that no drawing calls were made
    assert len(renderer.calls) == 0


def test_histogram_overlay_render_no_zero_line(chart, renderer):
    """Test histogram overlay rendering without zero line."""
    indicator = MACD()
    style = HistogramStyle(zero_line_width=0.0)
    overlay = HistogramOverlay(indicator, style)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series(np.random.uniform(-1, 1, 10), index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that no zero line was drawn
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) == 0 