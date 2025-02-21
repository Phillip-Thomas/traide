"""Unit tests for candlestick chart component."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cairo

from src.marketdata.visualization.components.candlestick_chart import (
    CandlestickChart,
    CandlestickStyle
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
    """Create a candlestick chart instance."""
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
    style = CandlestickStyle(
        up_color=(0.0, 0.8, 0.0),
        down_color=(0.8, 0.0, 0.0),
        wick_width=1.0,
        body_width=8.0,
        hollow_up=False,
        hollow_down=False
    )
    return CandlestickChart(dimensions, scales, style)


@pytest.fixture
def renderer(surface):
    """Create a renderer for testing."""
    return CairoRenderer(surface)


def test_chart_initialization(chart):
    """Test candlestick chart initialization."""
    assert chart.dimensions.width == 800
    assert chart.dimensions.height == 600
    assert chart.dimensions.inner_width == 700
    assert chart.dimensions.inner_height == 550
    assert chart.scales.x_scale_type == 'time'
    assert chart.scales.y_scale_type == 'linear'
    assert chart.style.up_color == (0.0, 0.8, 0.0)
    assert chart.style.down_color == (0.8, 0.0, 0.0)
    assert chart.style.wick_width == 1.0
    assert chart.style.body_width == 8.0
    assert not chart.style.hollow_up
    assert not chart.style.hollow_down


def test_render_empty_chart(chart, renderer):
    """Test rendering an empty chart."""
    chart.render(renderer)
    # No errors should be raised


def test_render_with_data(chart, renderer, sample_data):
    """Test rendering a chart with data."""
    chart.set_data(sample_data)
    chart.render(renderer)
    # No errors should be raised


def test_render_with_hollow_candles(chart, renderer, sample_data):
    """Test rendering with hollow candles."""
    chart.style.hollow_up = True
    chart.style.hollow_down = False
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
    chart.style.up_color = (0.0, 1.0, 0.0)
    chart.style.down_color = (1.0, 0.0, 0.0)
    chart.render(renderer)
    
    # Change widths
    chart.style.wick_width = 2.0
    chart.style.body_width = 12.0
    chart.render(renderer)
    
    # Change hollow settings
    chart.style.hollow_up = True
    chart.style.hollow_down = True
    chart.render(renderer)
    
    # No errors should be raised 