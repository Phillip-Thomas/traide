"""Unit tests for oscillator overlay."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.visualization.components.base_chart import BaseChart, ChartDimensions, ChartScales
from src.marketdata.visualization.overlays.oscillator import (
    OscillatorLevel,
    OscillatorStyle,
    OscillatorOverlay
)
from src.marketdata.visualization.overlays.base import OverlayStyle
from src.marketdata.indicators import RelativeStrengthIndex as RSI
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
def chart(sample_data):
    """Create a chart for testing."""
    dimensions = ChartDimensions(
        width=800,
        height=600,
        margin_top=20,
        margin_right=20,
        margin_bottom=30,
        margin_left=50
    )
    scales = ChartScales(
        x_type='linear',
        y_type='linear'
    )
    chart = BaseChart(dimensions, scales)
    chart.set_data(sample_data)  # This will initialize the domains
    return chart


@pytest.fixture
def renderer():
    """Create a mock renderer for testing."""
    return MockRenderer()


def test_oscillator_level():
    """Test oscillator level configuration."""
    level = OscillatorLevel(
        value=70,
        color=(1.0, 0.0, 0.0),
        width=2.0,
        style='dashed',
        label='Overbought'
    )
    
    assert level.value == 70
    assert level.color == (1.0, 0.0, 0.0)
    assert level.width == 2.0
    assert level.style == 'dashed'
    assert level.label == 'Overbought'


def test_oscillator_style():
    """Test oscillator style configuration."""
    style = OscillatorStyle(
        color=(1.0, 0.0, 0.0),
        line_width=2.0,
        line_style='dashed',
        opacity=0.8,
        show_points=True,
        point_size=6.0,
        z_index=1,
        visible=True,
        label='Test',
        value_column='value',
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0),
        smooth=True,
        levels=[
            OscillatorLevel(value=80, label='High'),
            OscillatorLevel(value=20, label='Low')
        ]
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
    assert style.value_column == 'value'
    assert style.fill_opacity == 0.3
    assert style.fill_color == (0.8, 0.0, 0.0)
    assert style.smooth
    assert len(style.levels) == 2
    assert style.levels[0].value == 80
    assert style.levels[1].value == 20


def test_oscillator_style_default_levels():
    """Test oscillator style default levels."""
    style = OscillatorStyle()
    assert len(style.levels) == 2
    assert style.levels[0].value == 70  # Overbought
    assert style.levels[1].value == 30  # Oversold


def test_oscillator_style_default_fill_color():
    """Test oscillator style default fill color."""
    style = OscillatorStyle(color=(1.0, 0.0, 0.0))
    assert style.fill_color == (1.0, 0.0, 0.0)


def test_oscillator_overlay_initialization():
    """Test oscillator overlay initialization."""
    indicator = RSI()
    style = OscillatorStyle(color=(1.0, 0.0, 0.0))
    overlay = OscillatorOverlay(indicator, style)
    
    assert overlay.indicator == indicator
    assert overlay.style == style
    assert overlay.visible
    assert overlay.z_index == 0
    assert overlay._result is None


def test_oscillator_overlay_get_values_series():
    """Test oscillator overlay value retrieval from Series."""
    indicator = RSI()
    style = OscillatorStyle()
    overlay = OscillatorOverlay(indicator, style)
    
    # Create test result Series
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.Series(np.random.uniform(0, 100, 10), index=dates)
    
    overlay._result = result
    values = overlay._get_values()
    
    assert isinstance(values, pd.Series)
    assert len(values) == len(result)
    assert (values == result).all()


def test_oscillator_overlay_get_values_dataframe():
    """Test oscillator overlay value retrieval from DataFrame."""
    indicator = RSI()
    style = OscillatorStyle(value_column='value')
    overlay = OscillatorOverlay(indicator, style)
    
    # Create test result DataFrame
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.DataFrame({
        'value': np.random.uniform(0, 100, 10),
        'other': np.random.uniform(0, 100, 10)
    }, index=dates)
    
    overlay._result = result
    values = overlay._get_values()
    
    assert isinstance(values, pd.Series)
    assert len(values) == len(result)
    assert (values == result['value']).all()


def test_oscillator_overlay_get_values_missing_column():
    """Test oscillator overlay value retrieval with missing column."""
    indicator = RSI()
    style = OscillatorStyle(value_column='missing')
    overlay = OscillatorOverlay(indicator, style)
    
    # Create test result DataFrame without required column
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.DataFrame({
        'value': np.random.uniform(0, 100, 10)
    }, index=dates)
    
    overlay._result = result
    with pytest.raises(ValueError, match="Result DataFrame missing required column"):
        overlay._get_values()


def test_oscillator_overlay_render_smooth(chart, renderer, sample_data):
    """Test oscillator overlay smooth rendering."""
    indicator = RSI()
    style = OscillatorStyle(
        color=(1.0, 0.0, 0.0),
        smooth=True,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0)
    )
    overlay = OscillatorOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for bezier curve calls
    bezier_calls = [call for call in renderer.calls if call[0] == 'draw_bezier']
    assert len(bezier_calls) > 0
    
    # Check for fill polygon calls
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) > 0


def test_oscillator_overlay_render_straight(chart, renderer, sample_data):
    """Test oscillator overlay straight line rendering."""
    indicator = RSI()
    style = OscillatorStyle(
        color=(1.0, 0.0, 0.0),
        smooth=False,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0)
    )
    overlay = OscillatorOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for line calls
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) > 0
    
    # Check for fill polygon calls
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) > 0


def test_oscillator_overlay_render_points(chart, renderer, sample_data):
    """Test oscillator overlay point rendering."""
    indicator = RSI()
    style = OscillatorStyle(
        color=(1.0, 0.0, 0.0),
        show_points=True,
        point_size=6.0
    )
    overlay = OscillatorOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for circle calls
    circle_calls = [call for call in renderer.calls if call[0] == 'draw_circle']
    assert len(circle_calls) > 0


def test_oscillator_overlay_render_levels(chart, renderer):
    """Test oscillator overlay level rendering."""
    indicator = RSI()
    style = OscillatorStyle(
        levels=[
            OscillatorLevel(value=80, label='High'),
            OscillatorLevel(value=20, label='Low')
        ]
    )
    overlay = OscillatorOverlay(indicator, style)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series(np.random.uniform(0, 100, 10), index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check for level lines
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) >= 2  # At least one line per level
    
    # Check for level labels
    text_calls = [call for call in renderer.calls if call[0] == 'draw_text']
    assert len(text_calls) == 2  # One label per level


def test_oscillator_overlay_render_with_gaps(chart, renderer):
    """Test oscillator overlay rendering with gaps in data."""
    indicator = RSI()
    style = OscillatorStyle()
    overlay = OscillatorOverlay(indicator, style)
    
    # Create data with gaps
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([70, np.nan, 65, 60, np.nan, np.nan, 40, 35, 30, 25], index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that we have the correct number of line segments
    # (should be broken into multiple segments due to gaps)
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) > 0


def test_oscillator_overlay_render_empty_data(chart, renderer):
    """Test oscillator overlay rendering with empty data."""
    indicator = RSI()
    style = OscillatorStyle()
    overlay = OscillatorOverlay(indicator, style)
    
    # Create empty data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([np.nan] * 10, index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that no drawing calls were made
    assert len(renderer.calls) == 0


def test_oscillator_overlay_render_no_fill(chart, renderer, sample_data):
    """Test oscillator overlay rendering without fill."""
    indicator = RSI()
    style = OscillatorStyle(
        color=(1.0, 0.0, 0.0),
        fill_opacity=0.0
    )
    overlay = OscillatorOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check that no fill polygons were drawn
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) == 0 