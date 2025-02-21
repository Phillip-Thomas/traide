"""Unit tests for base overlay system."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.marketdata.visualization.overlays.base import (
    OverlayStyle,
    IndicatorOverlay,
    OverlayManager
)
from src.marketdata.visualization.components import (
    BaseChart,
    ChartDimensions,
    ChartScales
)
from src.marketdata.visualization.utils import Renderer
from src.marketdata.indicators import SimpleMovingAverage


class MockRenderer(Renderer):
    """Mock renderer for testing."""
    
    def __init__(self):
        self.calls = []
    
    def clear(self) -> None:
        self.calls.append(('clear',))
    
    def draw_line(self, start, end, color, width=1.0, dash=None) -> None:
        self.calls.append(('draw_line', start, end, color, width, dash))
    
    def draw_rect(self, x, y, width, height, color, fill=True, line_width=1.0) -> None:
        self.calls.append(('draw_rect', x, y, width, height, color, fill, line_width))
    
    def draw_text(self, text, pos, color, font_size=12.0, align='left', baseline='top') -> None:
        self.calls.append(('draw_text', text, pos, color, font_size, align, baseline))
    
    def draw_polygon(self, points, color, fill=True, line_width=1.0) -> None:
        self.calls.append(('draw_polygon', points, color, fill, line_width))
    
    def draw_circle(self, center, radius, color, fill=True, line_width=1.0) -> None:
        self.calls.append(('draw_circle', center, radius, color, fill, line_width))
    
    def draw_bezier(self, start, control1, control2, end, color, width=1.0) -> None:
        self.calls.append(('draw_bezier', start, control1, control2, end, color, width))


class MockOverlay(IndicatorOverlay):
    """Mock overlay for testing."""
    
    def _render_indicator(self, chart, renderer):
        renderer.draw_line(
            start=(0, 0),
            end=(100, 100),
            color=self.style.color,
            width=self.style.line_width
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
    return pd.DataFrame(data)


@pytest.fixture
def chart(sample_data):
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
    chart = BaseChart(dimensions, scales)
    chart.set_data(sample_data)  # This will initialize the domains
    return chart


@pytest.fixture
def renderer():
    """Create a mock renderer for testing."""
    return MockRenderer()


def test_overlay_style():
    """Test overlay style configuration."""
    style = OverlayStyle(
        color=(1.0, 0.0, 0.0),
        line_width=2.0,
        line_style='dashed',
        opacity=0.8,
        show_points=True,
        point_size=6.0,
        z_index=1,
        visible=True,
        label='Test'
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


def test_overlay_initialization():
    """Test overlay initialization."""
    indicator = SimpleMovingAverage()
    style = OverlayStyle(color=(1.0, 0.0, 0.0))
    overlay = MockOverlay(indicator, style)
    
    assert overlay.indicator == indicator
    assert overlay.style == style
    assert overlay.visible
    assert overlay.z_index == 0
    assert overlay._result is None


def test_overlay_visibility(chart, renderer):
    """Test overlay visibility control."""
    overlay = MockOverlay(SimpleMovingAverage())
    
    # Test visible
    overlay.visible = True
    overlay.render(chart, renderer)
    assert len(renderer.calls) > 0
    
    # Test invisible
    renderer.calls.clear()
    overlay.visible = False
    overlay.render(chart, renderer)
    assert len(renderer.calls) == 0


def test_overlay_z_index():
    """Test overlay z-index control."""
    overlay = MockOverlay(SimpleMovingAverage())
    
    assert overlay.z_index == 0
    overlay.z_index = 2
    assert overlay.z_index == 2


def test_overlay_update(sample_data):
    """Test overlay update with new data."""
    indicator = SimpleMovingAverage()
    overlay = MockOverlay(indicator)
    
    assert overlay._result is None
    overlay.update(sample_data)
    assert overlay._result is not None


def test_overlay_manager():
    """Test overlay manager functionality."""
    manager = OverlayManager()
    
    # Test add overlay
    overlay1 = MockOverlay(SimpleMovingAverage())
    overlay2 = MockOverlay(SimpleMovingAverage())
    overlay2.z_index = 1
    
    manager.add_overlay(overlay1)
    manager.add_overlay(overlay2)
    
    assert len(manager.overlays) == 2
    assert manager.overlays[0].z_index < manager.overlays[1].z_index
    
    # Test remove overlay
    manager.remove_overlay(overlay1)
    assert len(manager.overlays) == 1
    assert manager.overlays[0] == overlay2
    
    # Test clear overlays
    manager.clear_overlays()
    assert len(manager.overlays) == 0


def test_overlay_manager_update(sample_data):
    """Test overlay manager update functionality."""
    manager = OverlayManager()
    overlay = MockOverlay(SimpleMovingAverage())
    manager.add_overlay(overlay)
    
    assert overlay._result is None
    manager.update_overlays(sample_data)
    assert overlay._result is not None


def test_overlay_manager_render(chart, renderer):
    """Test overlay manager render functionality."""
    manager = OverlayManager()
    overlay1 = MockOverlay(SimpleMovingAverage())
    overlay2 = MockOverlay(SimpleMovingAverage())
    
    manager.add_overlay(overlay1)
    manager.add_overlay(overlay2)
    
    manager.render_overlays(chart, renderer)
    assert len(renderer.calls) == 2  # One call per overlay


def test_overlay_manager_z_index_sorting():
    """Test overlay manager z-index sorting."""
    manager = OverlayManager()
    
    overlay1 = MockOverlay(SimpleMovingAverage())
    overlay2 = MockOverlay(SimpleMovingAverage())
    overlay3 = MockOverlay(SimpleMovingAverage())
    
    overlay1.z_index = 2
    overlay2.z_index = 0
    overlay3.z_index = 1
    
    manager.add_overlay(overlay1)
    manager.add_overlay(overlay2)
    manager.add_overlay(overlay3)
    
    assert manager.overlays[0].z_index == 0
    assert manager.overlays[1].z_index == 1
    assert manager.overlays[2].z_index == 2 