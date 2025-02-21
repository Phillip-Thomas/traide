"""Unit tests for band overlay."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.visualization.components.base_chart import ChartDimensions, ChartScales
from src.marketdata.visualization.overlays.band import (
    BandOverlay,
    BandStyle
)
from src.marketdata.visualization.overlays.base import OverlayStyle
from src.marketdata.visualization.components.base_chart import BaseChart
from src.marketdata.indicators import BollingerBands
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
    return pd.DataFrame(data).set_index('timestamp')


@pytest.fixture
def chart():
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
        y_type='linear',
        x_min=0,
        x_max=100,
        y_min=0,
        y_max=100
    )
    return BaseChart(dimensions, scales)


@pytest.fixture
def renderer():
    """Create a mock renderer for testing."""
    return MockRenderer()


def test_band_style():
    """Test band style configuration."""
    style = BandStyle(
        color=(1.0, 0.0, 0.0),
        line_width=2.0,
        line_style='dashed',
        opacity=0.8,
        show_points=True,
        point_size=6.0,
        z_index=1,
        visible=True,
        label='Test',
        upper_column='upper_band',
        middle_column='middle_band',
        lower_column='lower_band',
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0),
        smooth=True
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
    assert style.upper_column == 'upper_band'
    assert style.middle_column == 'middle_band'
    assert style.lower_column == 'lower_band'
    assert style.fill_opacity == 0.3
    assert style.fill_color == (0.8, 0.0, 0.0)
    assert style.smooth


def test_band_style_default_fill_color():
    """Test band style default fill color."""
    style = BandStyle(color=(1.0, 0.0, 0.0))
    assert style.fill_color == (1.0, 0.0, 0.0)


def test_band_overlay_initialization():
    """Test band overlay initialization."""
    indicator = BollingerBands()
    style = BandStyle(color=(1.0, 0.0, 0.0))
    overlay = BandOverlay(indicator, style)
    
    assert overlay.indicator == indicator
    assert overlay.style == style
    assert overlay.visible
    assert overlay.z_index == 0
    assert overlay._result is None


def test_band_overlay_get_values(sample_data):
    """Test band overlay value retrieval."""
    indicator = BollingerBands()
    style = BandStyle(
        upper_column='upper',
        middle_column='middle',
        lower_column='lower'
    )
    overlay = BandOverlay(indicator, style)
    
    # Create test result DataFrame
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.DataFrame({
        'upper': np.random.uniform(110, 120, 10),
        'middle': np.random.uniform(100, 110, 10),
        'lower': np.random.uniform(90, 100, 10)
    }, index=dates)
    
    overlay._result = result
    values = overlay._get_values()
    
    assert isinstance(values, pd.DataFrame)
    assert len(values) == len(result)
    assert all(col in values.columns for col in ['upper', 'middle', 'lower'])


def test_band_overlay_get_values_missing_columns():
    """Test band overlay value retrieval with missing columns."""
    indicator = BollingerBands()
    style = BandStyle()
    overlay = BandOverlay(indicator, style)
    
    # Create test result DataFrame with missing columns
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    result = pd.DataFrame({
        'upper': np.random.uniform(110, 120, 10),
        'middle': np.random.uniform(100, 110, 10)
    }, index=dates)
    
    overlay._result = result
    with pytest.raises(ValueError, match="Result DataFrame missing required columns"):
        overlay._get_values()


def test_band_overlay_render_smooth(chart, renderer, sample_data):
    """Test band overlay smooth rendering."""
    indicator = BollingerBands()
    style = BandStyle(
        color=(1.0, 0.0, 0.0),
        smooth=True,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0)
    )
    overlay = BandOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for bezier curve calls
    bezier_calls = [call for call in renderer.calls if call[0] == 'draw_bezier']
    assert len(bezier_calls) > 0
    
    # Check for fill polygon calls
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) > 0


def test_band_overlay_render_straight(chart, renderer, sample_data):
    """Test band overlay straight line rendering."""
    indicator = BollingerBands()
    style = BandStyle(
        color=(1.0, 0.0, 0.0),
        smooth=False,
        fill_opacity=0.3,
        fill_color=(0.8, 0.0, 0.0)
    )
    overlay = BandOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for line calls
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) > 0
    
    # Check for fill polygon calls
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) > 0


def test_band_overlay_render_points(chart, renderer, sample_data):
    """Test band overlay point rendering."""
    indicator = BollingerBands()
    style = BandStyle(
        color=(1.0, 0.0, 0.0),
        show_points=True,
        point_size=6.0
    )
    overlay = BandOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check for circle calls
    circle_calls = [call for call in renderer.calls if call[0] == 'draw_circle']
    assert len(circle_calls) > 0


def test_band_overlay_render_no_fill(chart, renderer, sample_data):
    """Test band overlay rendering without fill."""
    indicator = BollingerBands()
    style = BandStyle(
        color=(1.0, 0.0, 0.0),
        fill_opacity=0.0
    )
    overlay = BandOverlay(indicator, style)
    
    overlay.update(sample_data)
    overlay.render(chart, renderer)
    
    # Check that no fill polygons were drawn
    polygon_calls = [call for call in renderer.calls if call[0] == 'draw_polygon']
    assert len(polygon_calls) == 0


def test_band_overlay_render_with_gaps(chart, renderer):
    """Test band overlay rendering with gaps in data."""
    # Create data with gaps
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.DataFrame({
        'upper': [110, np.nan, 112, 113, np.nan, np.nan, 116, 117, 118, 119],
        'middle': [100, np.nan, 102, 103, np.nan, np.nan, 106, 107, 108, 109],
        'lower': [90, np.nan, 92, 93, np.nan, np.nan, 96, 97, 98, 99]
    }, index=dates)
    
    indicator = BollingerBands()
    style = BandStyle(color=(1.0, 0.0, 0.0))
    overlay = BandOverlay(indicator, style)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that we have the correct number of line segments
    # (should be broken into multiple segments due to gaps)
    line_calls = [call for call in renderer.calls if call[0] == 'draw_line']
    assert len(line_calls) > 0


def test_band_overlay_render_empty_data(chart, renderer):
    """Test band overlay rendering with empty data."""
    indicator = BollingerBands()
    style = BandStyle(color=(1.0, 0.0, 0.0))
    overlay = BandOverlay(indicator, style)
    
    # Create empty data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
    data = pd.DataFrame({
        'upper': [np.nan] * 10,
        'middle': [np.nan] * 10,
        'lower': [np.nan] * 10
    }, index=dates)
    
    overlay._result = data
    overlay.render(chart, renderer)
    
    # Check that no drawing calls were made 