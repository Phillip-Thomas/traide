"""Unit tests for line overlay."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.visualization.components.base_chart import ChartDimensions, ChartScales
from src.marketdata.visualization.overlays.line import (
    LineOverlay,
    LineStyle
)
from src.marketdata.visualization.overlays.base import OverlayStyle
from src.marketdata.visualization.components.base_chart import BaseChart
from src.marketdata.indicators import SimpleMovingAverage
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


def test_line_style():
    """Test line style configuration."""
    style = LineStyle(
        color=(1.0, 0.0, 0.0),
        line_width=2.0,
        line_style='dashed',
        opacity=0.8,
        show_points=True,
        point_size=6.0,
        z_index=1,
        visible=True,
        label='Test',
        smooth=True,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0),
        value_column='close'
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
    assert style.smooth
    assert style.fill_opacity == 0.3
    assert style.fill_color == (0.8, 0.0, 0.0)
    assert style.value_column == 'close'


def test_line_overlay_initialization():
    """Test line overlay initialization."""
    indicator = SimpleMovingAverage()
    style = LineStyle(color=(1.0, 0.0, 0.0))
    overlay = LineOverlay(indicator, style)
    
    assert overlay.indicator == indicator
    assert overlay.style == style
    assert overlay.visible
    assert overlay.z_index == 0
    assert overlay._result is None


def test_line_overlay_get_values(sample_data):
    """Test line overlay value retrieval."""
    indicator = SimpleMovingAverage()
    style = LineStyle(color=(1.0, 0.0, 0.0), value_column='close')
    overlay = LineOverlay(indicator, style)
    
    overlay.update(sample_data)
    values = overlay._get_values()
    
    assert isinstance(values, pd.Series)
    assert len(values) == len(sample_data)
    assert not values.isna().all()


def test_line_overlay_render_smooth(chart, renderer, sample_data):
    """Test line overlay smooth rendering."""
    indicator = SimpleMovingAverage()
    style = LineStyle(
        color=(1.0, 0.0, 0.0),
        smooth=True,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0)
    )
    overlay = LineOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for bezier curve calls
    bezier_calls = [call for call in renderer.calls if call[0] == 'draw_bezier']
    assert len(bezier_calls) > 0
    
    # Check for fill polygon calls
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) > 0


def test_line_overlay_render_straight(chart, renderer, sample_data):
    """Test line overlay straight line rendering."""
    indicator = SimpleMovingAverage()
    style = LineStyle(
        color=(1.0, 0.0, 0.0),
        smooth=False,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0)
    )
    overlay = LineOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for line calls
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) > 0
    
    # Check for fill polygon calls
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) > 0


def test_line_overlay_render_points(chart, renderer, sample_data):
    """Test line overlay point rendering."""
    indicator = SimpleMovingAverage()
    style = LineStyle(
        color=(1.0, 0.0, 0.0),
        show_points=True,
        point_size=6.0
    )
    overlay = LineOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for circle calls
    circle_calls = [call for call in renderer.calls if call[0] == 'draw_circle']
    assert len(circle_calls) > 0


def test_line_overlay_render_no_fill(chart, renderer, sample_data):
    """Test line overlay rendering without fill."""
    indicator = SimpleMovingAverage()
    style = LineStyle(
        color=(1.0, 0.0, 0.0),
        fill_opacity=0.0
    )
    overlay = LineOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check that no fill polygons were drawn
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) == 0


def test_line_overlay_render_with_gaps(chart, renderer):
    """Test line overlay rendering with gaps in data."""
    # Create data with gaps
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0], index=dates)
    
    indicator = SimpleMovingAverage()
    style = LineStyle(color=(1.0, 0.0, 0.0))
    overlay = LineOverlay(indicator, style)
    
    overlay._result = data  # Directly set result for testing
    overlay.render(chart, renderer)
    
    # Check that we have the correct number of line segments
    # (should be broken into multiple segments due to gaps)
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) > 0


def test_line_overlay_render_empty_data(chart, renderer):
    """Test line overlay rendering with empty data."""
    indicator = SimpleMovingAverage()
    style = LineStyle(color=(1.0, 0.0, 0.0))
    overlay = LineOverlay(indicator, style)
    
    # Create empty data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.Series([np.nan] * 10, index=dates)
    
    overlay._result = data  # Directly set result for testing
    overlay.render(chart, renderer)
    
    # Check that no drawing calls were made
    assert len(renderer.calls) == 0 