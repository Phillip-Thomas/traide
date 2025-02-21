"""Unit tests for base chart component."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.visualization.components.base_chart import (
    BaseChart,
    ChartDimensions,
    ChartScales
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
    
    # Ensure OHLC relationships are valid
    df = pd.DataFrame(data)
    df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['low'])
    df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['high'])
    
    return df


@pytest.fixture
def chart():
    """Create a base chart instance."""
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
    return BaseChart(dimensions, scales)


def test_chart_initialization(chart):
    """Test chart initialization."""
    assert chart.dimensions.width == 800
    assert chart.dimensions.height == 600
    assert chart.dimensions.inner_width == 700
    assert chart.dimensions.inner_height == 550
    assert chart.scales.x_scale_type == 'time'
    assert chart.scales.y_scale_type == 'linear'
    assert chart._data is None
    assert chart._x_domain is None
    assert chart._y_domain is None
    assert chart._zoom_level == 1.0
    assert chart._pan_offset == (0.0, 0.0)


def test_set_data(chart, sample_data):
    """Test setting chart data."""
    chart.set_data(sample_data)
    
    assert chart._data is not None
    assert len(chart._data) == len(sample_data)
    assert chart._x_domain is not None
    assert chart._y_domain is not None
    assert chart._x_domain[0] == sample_data['timestamp'].min()
    assert chart._x_domain[1] == sample_data['timestamp'].max()


def test_set_data_validation(chart):
    """Test data validation when setting data."""
    invalid_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10),
        'close': np.random.uniform(100, 110, 10)
    })
    
    with pytest.raises(ValueError, match='Missing required columns'):
        chart.set_data(invalid_data)


def test_zoom(chart, sample_data):
    """Test chart zooming."""
    chart.set_data(sample_data)
    initial_domain = chart._x_domain
    
    # Zoom in
    chart.handle_zoom(1.0, (400, 300))
    assert chart._zoom_level > 1.0
    
    # Zoom out
    chart.handle_zoom(-1.0, (400, 300))
    assert chart._zoom_level == pytest.approx(1.0, rel=1e-2)


def test_pan(chart, sample_data):
    """Test chart panning."""
    chart.set_data(sample_data)
    initial_domain = chart._x_domain
    
    # Start pan
    chart.handle_pan_start((400, 300))
    assert chart._is_panning
    assert chart._last_mouse_pos == (400, 300)
    
    # Move pan
    chart.handle_pan_move((420, 320))
    assert chart._pan_offset == (20, 20)
    
    # End pan
    chart.handle_pan_end()
    assert not chart._is_panning
    assert chart._last_mouse_pos is None


def test_coordinate_conversion(chart, sample_data):
    """Test coordinate conversion methods."""
    chart.set_data(sample_data)
    
    # Test data to pixel conversion
    timestamp = sample_data['timestamp'].iloc[0]
    price = sample_data['close'].iloc[0]
    x, y = chart.to_pixel_coords(timestamp, price)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert chart.dimensions.margin_left <= x <= chart.dimensions.width - chart.dimensions.margin_right
    assert chart.dimensions.margin_top <= y <= chart.dimensions.height - chart.dimensions.margin_bottom
    
    # Test pixel to data conversion
    t, p = chart.from_pixel_coords(x, y)
    assert isinstance(t, datetime)
    assert isinstance(p, float)
    assert abs((t - timestamp).total_seconds()) < 1  # Within 1 second
    assert abs(p - price) < 0.01  # Within 0.01


def test_visible_data(chart, sample_data):
    """Test getting visible data."""
    chart.set_data(sample_data)
    
    # Initial view should show all data
    visible = chart.get_visible_data()
    assert len(visible) == len(sample_data)
    
    # Pan to show only part of the data
    chart.handle_pan_start((400, 300))
    chart.handle_pan_move((200, 300))
    
    visible = chart.get_visible_data()
    assert len(visible) <= len(sample_data)


def test_log_scale(chart, sample_data):
    """Test logarithmic price scale."""
    chart.scales.y_scale_type = 'log'
    chart.set_data(sample_data)
    
    # Test coordinate conversion with log scale
    timestamp = sample_data['timestamp'].iloc[0]
    price = sample_data['close'].iloc[0]
    x, y = chart.to_pixel_coords(timestamp, price)
    t, p = chart.from_pixel_coords(x, y)
    
    assert abs((t - timestamp).total_seconds()) < 1  # Within 1 second
    assert abs(p - price) / price < 0.01  # Within 1% 