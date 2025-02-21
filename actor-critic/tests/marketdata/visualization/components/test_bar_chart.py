"""Unit tests for bar chart component."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cairo

from src.marketdata.visualization.components.bar_chart import (
    BarChart,
    BarStyle
)
from src.marketdata.visualization.components.base_chart import (
    BaseChart,
    ChartDimensions,
    ChartScales
)
from src.marketdata.visualization.utils.renderer import CairoRenderer


@pytest.fixture
def surface():
    """Create a Cairo surface for testing."""
    return cairo.ImageSurface(cairo.FORMAT_ARGB32, 800, 600)


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
    
    # Ensure OHLC relationships are valid
    df = pd.DataFrame(data)
    df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['low'])
    df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['high'])
    
    return df


@pytest.fixture
def chart():
    """Create a bar chart instance."""
    dimensions = ChartDimensions(
        width=800,
        height=600,
        margin_top=20,
        margin_right=50,
        margin_bottom=30,
        margin_left=50
    )
    scales = ChartScales(
        x_scale_type='time',
        y_scale_type='linear',
        price_format='.2f',
        time_format='%Y-%m-%d %H:%M',
        show_grid=True,
        show_crosshair=True
    )
    style = BarStyle(
        up_color=(0.0, 0.8, 0.0),
        down_color=(0.8, 0.0, 0.0),
        neutral_color=(0.5, 0.5, 0.5),
        bar_width=8.0,
        bar_spacing=2.0,
        opacity=0.8
    )
    return BarChart(dimensions, scales, style)


@pytest.fixture
def renderer(surface):
    """Create a renderer for testing."""
    return CairoRenderer(surface)


def test_chart_initialization(chart):
    """Test bar chart initialization."""
    assert chart.dimensions.width == 800
    assert chart.dimensions.height == 600
    assert chart.dimensions.inner_width == 700
    assert chart.dimensions.inner_height == 550
    assert chart.scales.x_scale_type == 'time'
    assert chart.scales.y_scale_type == 'linear'
    assert chart.style.up_color == (0.0, 0.8, 0.0)
    assert chart.style.down_color == (0.8, 0.0, 0.0)
    assert chart.style.neutral_color == (0.5, 0.5, 0.5)
    assert chart.style.bar_width == 8.0
    assert chart.style.bar_spacing == 2.0
    assert chart.style.opacity == 0.8
    assert chart.value_column == 'volume'
    assert chart.color_by == 'close'


def test_render_empty_chart(chart, renderer):
    """Test rendering an empty chart."""
    chart.render(renderer)
    # No errors should be raised


def test_render_with_data(chart, renderer, sample_data):
    """Test rendering a chart with data."""
    chart.set_data(sample_data)
    chart.render(renderer)
    # No errors should be raised


def test_render_with_different_timeframes(chart, renderer):
    """Test rendering with different timeframes."""
    # Daily data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='1D')
    daily_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 30),
        'high': np.random.uniform(110, 120, 30),
        'low': np.random.uniform(90, 100, 30),
        'close': np.random.uniform(100, 110, 30),
        'volume': np.random.uniform(1000, 2000, 30)
    })
    chart.set_data(daily_data)
    chart.render(renderer)
    
    # Hourly data
    dates = pd.date_range(start='2024-01-01', periods=24, freq='1H')
    hourly_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 24),
        'high': np.random.uniform(110, 120, 24),
        'low': np.random.uniform(90, 100, 24),
        'close': np.random.uniform(100, 110, 24),
        'volume': np.random.uniform(1000, 2000, 24)
    })
    chart.set_data(hourly_data)
    chart.render(renderer)
    
    # No errors should be raised


def test_render_with_zoom(chart, renderer, sample_data):
    """Test rendering with zoom."""
    chart.set_data(sample_data)
    
    # Zoom in
    chart.handle_zoom(1.0, (400, 300))
    chart.render(renderer)
    
    # Zoom out
    chart.handle_zoom(-1.0, (400, 300))
    chart.render(renderer)
    
    # No errors should be raised


def test_render_with_pan(chart, renderer, sample_data):
    """Test rendering with pan."""
    chart.set_data(sample_data)
    
    # Pan chart
    chart.handle_pan_start((400, 300))
    chart.handle_pan_move((420, 320))
    chart.render(renderer)
    
    chart.handle_pan_end()
    chart.render(renderer)
    
    # No errors should be raised


def test_render_with_crosshair(chart, renderer, sample_data):
    """Test rendering with crosshair."""
    chart.set_data(sample_data)
    
    # Move mouse to trigger crosshair
    chart._last_mouse_pos = (400, 300)
    chart.render(renderer)
    
    # Move mouse outside chart area
    chart._last_mouse_pos = (0, 0)
    chart.render(renderer)
    
    # No errors should be raised


def test_render_with_grid(chart, renderer, sample_data):
    """Test rendering with and without grid."""
    chart.set_data(sample_data)
    
    # With grid
    chart.scales.show_grid = True
    chart.render(renderer)
    
    # Without grid
    chart.scales.show_grid = False
    chart.render(renderer)
    
    # No errors should be raised


def test_render_with_different_styles(chart, renderer, sample_data):
    """Test rendering with different styles."""
    chart.set_data(sample_data)
    
    # Change colors
    chart.style.up_color = (1.0, 0.0, 0.0)
    chart.style.down_color = (0.0, 0.0, 1.0)
    chart.style.neutral_color = (0.0, 0.0, 0.0)
    chart.render(renderer)
    
    # Change bar properties
    chart.style.bar_width = 12.0
    chart.style.bar_spacing = 4.0
    chart.style.opacity = 0.5
    chart.render(renderer)
    
    # No errors should be raised


def test_render_with_different_value_columns(chart, renderer, sample_data):
    """Test rendering with different value columns."""
    chart.set_data(sample_data)
    
    # Test different value columns
    for column in ['volume', 'close', 'high', 'low']:
        chart.value_column = column
        chart.render(renderer)
    
    # No errors should be raised


def test_render_with_different_color_by(chart, renderer, sample_data):
    """Test rendering with different color_by settings."""
    chart.set_data(sample_data)
    
    # Test different color_by columns
    for column in [None, 'close', 'volume']:
        chart.color_by = column
        chart.render(renderer)
    
    # No errors should be raised


def test_value_formatting(chart, renderer, sample_data):
    """Test value formatting in axes and crosshair."""
    # Create data with different value ranges
    sample_data['small_values'] = sample_data['volume'] / 1000  # Values < 1000
    sample_data['large_values'] = sample_data['volume'] * 1000  # Values > 1M
    
    chart.set_data(sample_data)
    
    # Test with small values
    chart.value_column = 'small_values'
    chart.render(renderer)
    
    # Test with large values
    chart.value_column = 'large_values'
    chart.render(renderer)
    
    # No errors should be raised


def test_bar_width_calculation(chart, renderer):
    """Test bar width calculation with different data densities."""
    # Test with sparse data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1D')
    sparse_data = pd.DataFrame({
        'timestamp': dates,
        'volume': np.random.uniform(1000, 2000, 10),
        'close': np.random.uniform(100, 110, 10),
        'open': np.random.uniform(100, 110, 10),
        'high': np.random.uniform(110, 120, 10),
        'low': np.random.uniform(90, 100, 10)
    })
    chart.set_data(sparse_data)
    chart.render(renderer)
    
    # Test with dense data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    dense_data = pd.DataFrame({
        'timestamp': dates,
        'volume': np.random.uniform(1000, 2000, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(110, 120, 1000),
        'low': np.random.uniform(90, 100, 1000)
    })
    chart.set_data(dense_data)
    chart.render(renderer)
    
    # No errors should be raised 