"""Base chart component for market data visualization."""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.scales import LinearScale, LogScale, TimeScale
from ..utils.renderer import Renderer
from ..overlays.base import OverlayManager


@dataclass
class ChartDimensions:
    """Chart dimensions configuration."""
    width: float = 800.0
    height: float = 600.0
    margin_top: float = 20.0
    margin_right: float = 50.0
    margin_bottom: float = 30.0
    margin_left: float = 50.0
    
    @property
    def inner_width(self) -> float:
        """Get inner width (excluding margins)."""
        return self.width - self.margin_left - self.margin_right
    
    @property
    def inner_height(self) -> float:
        """Get inner height (excluding margins)."""
        return self.height - self.margin_top - self.margin_bottom


@dataclass
class ChartScales:
    """Chart scaling configuration."""
    x_scale_type: str = 'time'  # 'time' or 'linear'
    y_scale_type: str = 'linear'  # 'linear' or 'log'
    price_format: str = '.2f'
    time_format: str = '%Y-%m-%d %H:%M'
    show_grid: bool = True
    show_crosshair: bool = True
    x_min: Optional[Union[float, datetime]] = None
    x_max: Optional[Union[float, datetime]] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_type: str = 'linear'  # 'linear' or 'time'
    y_type: str = 'linear'  # 'linear' or 'log'
    x: Optional[Union[LinearScale, TimeScale]] = None
    y: Optional[Union[LinearScale, LogScale]] = None
    
    def update_scales(
        self,
        x_domain: Tuple[Union[float, datetime], Union[float, datetime]],
        y_domain: Tuple[float, float],
        dimensions: ChartDimensions
    ) -> None:
        """Update scale objects based on domains and dimensions.
        
        Args:
            x_domain: X-axis domain (min, max)
            y_domain: Y-axis domain (min, max)
            dimensions: Chart dimensions
        """
        x_range = (dimensions.margin_left, dimensions.width - dimensions.margin_right)
        y_range = (dimensions.height - dimensions.margin_bottom, dimensions.margin_top)
        
        # Create x scale
        if self.x_scale_type == 'time':
            self.x = TimeScale(x_domain, x_range)
        else:
            self.x = LinearScale(x_domain, x_range)
        
        # Create y scale
        if self.y_scale_type == 'log':
            self.y = LogScale(y_domain, y_range)
        else:
            self.y = LinearScale(y_domain, y_range)


class BaseChart:
    """Base chart component for market data visualization."""
    
    def __init__(
        self,
        dimensions: ChartDimensions,
        scales: Optional[ChartScales] = None
    ):
        """Initialize the chart.
        
        Args:
            dimensions: Chart dimensions configuration
            scales: Chart scales configuration
        """
        self.dimensions = dimensions
        self.scales = scales or ChartScales()
        
        # Data state
        self._data: Optional[pd.DataFrame] = None
        self._x_domain: Optional[Tuple[datetime, datetime]] = None
        self._y_domain: Optional[Tuple[float, float]] = None
        
        # View state
        self._zoom_level: float = 1.0
        self._pan_offset: Tuple[float, float] = (0.0, 0.0)
        
        # Interaction state
        self._is_panning: bool = False
        self._last_mouse_pos: Optional[Tuple[float, float]] = None
        
        # Overlay manager
        self.overlay_manager = OverlayManager()
        
        # Subcharts
        self._subcharts: List[Tuple[BaseChart, float]] = []
    
    def add_subchart(self, chart: 'BaseChart', height_ratio: float = 0.2) -> None:
        """Add a subchart.
        
        Args:
            chart: The subchart to add
            height_ratio: Height ratio of the subchart (0.0 to 1.0)
        """
        if height_ratio <= 0.0 or height_ratio >= 1.0:
            raise ValueError("Height ratio must be between 0.0 and 1.0")
        
        # Add to subcharts list
        self._subcharts.append((chart, height_ratio))
        
        # Update dimensions
        total_ratio = sum(ratio for _, ratio in self._subcharts)
        if total_ratio >= 1.0:
            raise ValueError("Total height ratio of subcharts cannot exceed 1.0")
        
        main_height = self.dimensions.height * (1.0 - total_ratio)
        subchart_height = self.dimensions.height * height_ratio
        
        # Update main chart dimensions
        self.dimensions.height = main_height
        
        # Update subchart dimensions
        chart.dimensions.height = subchart_height
        chart.dimensions.margin_top = 10
        chart.dimensions.margin_bottom = 20
        chart.dimensions.margin_left = self.dimensions.margin_left
        chart.dimensions.margin_right = self.dimensions.margin_right
        
        # Share x-scale
        chart.scales.x = self.scales.x
        chart.scales.x_scale_type = self.scales.x_scale_type
        chart.scales.time_format = self.scales.time_format
        chart.scales.show_grid = self.scales.show_grid
        chart.scales.show_crosshair = self.scales.show_crosshair
    
    def set_data(self, df: pd.DataFrame) -> None:
        """Set the chart data.
        
        Args:
            df: DataFrame with OHLCV data
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Store data
        self._data = df
        
        # Update domains
        self._update_domains()
    
    def _update_domains(self) -> None:
        """Update data domains and scales."""
        if self._data is None:
            return
        
        # Update x domain (time)
        self._x_domain = (
            self._data['timestamp'].min(),
            self._data['timestamp'].max()
        )
        
        # Update y domain (price)
        self._y_domain = (
            self._data['low'].min(),
            self._data['high'].max()
        )
        
        # Update scales
        self.scales.update_scales(self._x_domain, self._y_domain, self.dimensions)
    
    def handle_zoom(self, delta: float, center: Tuple[float, float]) -> None:
        """Handle zoom event.
        
        Args:
            delta: Zoom delta (positive for zoom in, negative for zoom out)
            center: Zoom center point (x, y) in pixel coordinates
        """
        if self._data is None:
            return
        
        # Calculate new zoom level
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_zoom = self._zoom_level * zoom_factor
        
        # Limit zoom range
        new_zoom = np.clip(new_zoom, 0.1, 10.0)
        
        # Update zoom level
        self._zoom_level = new_zoom
        
        # Update view
        self._update_view()
    
    def handle_pan_start(self, pos: Tuple[float, float]) -> None:
        """Handle pan start event.
        
        Args:
            pos: Starting position (x, y) in pixel coordinates
        """
        self._is_panning = True
        self._last_mouse_pos = pos
    
    def handle_pan_move(self, pos: Tuple[float, float]) -> None:
        """Handle pan move event.
        
        Args:
            pos: Current position (x, y) in pixel coordinates
        """
        if not self._is_panning or self._last_mouse_pos is None:
            return
        
        # Calculate pan delta
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        
        # Update pan offset
        self._pan_offset = (
            self._pan_offset[0] + dx,
            self._pan_offset[1] + dy
        )
        
        # Update last position
        self._last_mouse_pos = pos
        
        # Update view
        self._update_view()
    
    def handle_pan_end(self) -> None:
        """Handle pan end event."""
        self._is_panning = False
        self._last_mouse_pos = None
    
    def _update_view(self) -> None:
        """Update view based on current zoom and pan state."""
        if self._data is None:
            return
        
        # Calculate visible data range
        visible_width = self.dimensions.inner_width / self._zoom_level
        visible_height = self.dimensions.inner_height / self._zoom_level
        
        # Update domains based on pan and zoom
        if self._x_domain is not None:
            x_range = self._x_domain[1] - self._x_domain[0]
            x_offset = pd.Timedelta(seconds=x_range.total_seconds() * self._pan_offset[0] / visible_width)
            self._x_domain = (
                self._x_domain[0] + x_offset,
                self._x_domain[1] + x_offset
            )
        
        if self._y_domain is not None:
            y_range = self._y_domain[1] - self._y_domain[0]
            y_offset = y_range * self._pan_offset[1] / visible_height
            self._y_domain = (
                self._y_domain[0] + y_offset,
                self._y_domain[1] + y_offset
            )
    
    def get_visible_data(self) -> Optional[pd.DataFrame]:
        """Get data visible in current view.
        
        Returns:
            pd.DataFrame: Visible data subset
        """
        if self._data is None or self._x_domain is None:
            return None
        
        mask = (
            (self._data['timestamp'] >= self._x_domain[0]) &
            (self._data['timestamp'] <= self._x_domain[1])
        )
        return self._data[mask].copy()
    
    def to_pixel_coords(self, timestamp: datetime, price: float) -> Tuple[float, float]:
        """Convert data coordinates to pixel coordinates.
        
        Args:
            timestamp: Data timestamp
            price: Price value
            
        Returns:
            Tuple[float, float]: Pixel coordinates (x, y)
        """
        if self.scales.x is None or self.scales.y is None:
            return (0.0, 0.0)
        
        x = self.scales.x.transform(timestamp)
        y = self.scales.y.transform(price)
        
        return (x, y)
    
    def from_pixel_coords(self, x: float, y: float) -> Tuple[datetime, float]:
        """Convert pixel coordinates to data coordinates.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            
        Returns:
            Tuple[datetime, float]: Data coordinates (timestamp, price)
        """
        if self.scales.x is None or self.scales.y is None:
            return (datetime.now(), 0.0)
        
        timestamp = self.scales.x.invert(x)
        price = self.scales.y.invert(y)
        
        return (timestamp, price)
    
    def render(self, renderer: Renderer) -> None:
        """Render the chart.
        
        Args:
            renderer: Renderer implementation
        """
        # Clear background
        renderer.clear()
        
        # Draw grid if enabled
        if self.scales.show_grid:
            self._draw_grid(renderer)
        
        # Draw axes
        self._draw_axes(renderer)
        
        # Draw chart content
        self._render_content(renderer)
        
        # Draw overlays
        self.overlay_manager.render_overlays(self, renderer)
        
        # Draw crosshair if enabled
        if self.scales.show_crosshair:
            self._draw_crosshair(renderer)
        
        # Draw subcharts
        y_offset = self.dimensions.height + self.dimensions.margin_bottom
        for subchart, _ in self._subcharts:
            # Save current transform
            renderer.save()
            
            # Translate to subchart position
            renderer.translate(0, y_offset)
            
            # Render subchart
            subchart.render(renderer)
            
            # Restore transform
            renderer.restore()
            
            # Update offset for next subchart
            y_offset += subchart.dimensions.height + subchart.dimensions.margin_bottom
    
    def _render_content(self, renderer: Renderer) -> None:
        """Render the chart content.
        
        Args:
            renderer: Renderer implementation
        """
        pass 