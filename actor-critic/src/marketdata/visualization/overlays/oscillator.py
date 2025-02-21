"""Oscillator overlay for technical indicators."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, TYPE_CHECKING
from datetime import datetime

import pandas as pd
import numpy as np

from ..overlays.base import OverlayStyle, IndicatorOverlay
from ..utils.renderer import Renderer

if TYPE_CHECKING:
    from ..components.base_chart import BaseChart


@dataclass
class OscillatorLevel:
    """Configuration for an oscillator level line."""
    
    value: float
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    width: float = 1.0
    style: Optional[str] = 'dashed'
    label: Optional[str] = None


@dataclass
class OscillatorStyle(OverlayStyle):
    """Style configuration for oscillator overlays."""
    
    # Oscillator-specific style options
    value_column: str = 'value'
    smooth: bool = False
    fill_opacity: float = 0.1
    fill_color: Optional[Tuple[float, float, float]] = None
    levels: List[OscillatorLevel] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default values."""
        super().__post_init__()
        if self.fill_color is None:
            # Default fill color is the line color with reduced opacity
            self.fill_color = self.color
        
        # Add default levels if none provided
        if not self.levels:
            self.levels = [
                OscillatorLevel(
                    value=70,
                    color=(0.8, 0.0, 0.0),  # Red
                    label='Overbought'
                ),
                OscillatorLevel(
                    value=30,
                    color=(0.0, 0.8, 0.0),  # Green
                    label='Oversold'
                )
            ]


class OscillatorOverlay(IndicatorOverlay):
    """Overlay for oscillator-based indicators like RSI."""
    
    def __init__(self, indicator, style: Optional[OscillatorStyle] = None):
        """Initialize the oscillator overlay.
        
        Args:
            indicator: The indicator to overlay
            style: Optional style configuration
        """
        if style is None:
            style = OscillatorStyle()
        super().__init__(indicator, style)
    
    def _get_values(self) -> pd.Series:
        """Get the values to plot.
        
        Returns:
            Series with oscillator values
        """
        if isinstance(self._result, pd.Series):
            return self._result
        elif isinstance(self._result, pd.DataFrame):
            if self.style.value_column not in self._result.columns:
                raise ValueError(f"Result DataFrame missing required column: {self.style.value_column}")
            return self._result[self.style.value_column]
        else:
            raise ValueError("Oscillator overlay requires Series or DataFrame result")
    
    def _draw_levels(self, chart: 'BaseChart', renderer: Renderer, x_range: Tuple[float, float]):
        """Draw oscillator level lines.
        
        Args:
            chart: The chart to draw on
            renderer: The renderer to use
            x_range: Tuple of (min_x, max_x) coordinates
        """
        for level in self.style.levels:
            y = chart.scales.y.transform(level.value)
            
            # Convert dash style to pattern
            dash_pattern = {
                'solid': None,
                'dashed': [6.0, 4.0],
                'dotted': [2.0, 2.0]
            }.get(level.style, None)
            
            # Draw level line
            renderer.draw_line(
                start=(x_range[0], y),
                end=(x_range[1], y),
                color=level.color,
                width=level.width,
                dash=dash_pattern
            )
            
            # Draw level label if provided
            if level.label:
                renderer.draw_text(
                    text=level.label,
                    pos=(x_range[1] + 5, y),
                    color=level.color,
                    align='left',
                    baseline='middle'
                )
    
    def _draw_fill(
        self,
        chart: 'BaseChart',
        renderer: Renderer,
        points: List[Tuple[float, float]]
    ) -> None:
        """Draw fill below the line.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
            points: List of points to draw
        """
        style = self.style
        if style.fill_opacity <= 0:
            return
        
        # Create polygon points
        polygon_points = points.copy()
        
        # Add bottom corners
        bottom_y = chart.dimensions.height - chart.dimensions.margin_bottom
        polygon_points.append((points[-1][0], bottom_y))
        polygon_points.append((points[0][0], bottom_y))
        
        # Draw fill
        fill_color = style.fill_color or style.color
        renderer.draw_polygon(
            points=polygon_points,
            color=(*fill_color, style.fill_opacity),
            fill=True
        )
    
    def _draw_line(
        self,
        renderer: Renderer,
        points: List[Tuple[float, float]]
    ) -> None:
        """Draw straight line segments.
        
        Args:
            renderer: The renderer to use
            points: List of points to draw
        """
        style = self.style
        
        for i in range(len(points) - 1):
            renderer.draw_line(
                start=points[i],
                end=points[i + 1],
                color=style.color,
                width=style.line_width
            )
    
    def _draw_smooth_line(
        self,
        renderer: Renderer,
        points: List[Tuple[float, float]]
    ) -> None:
        """Draw smooth line using bezier curves.
        
        Args:
            renderer: The renderer to use
            points: List of points to draw
        """
        style = self.style
        
        for i in range(len(points) - 1):
            # Calculate control points
            if i == 0:
                # First segment
                control1 = points[i]
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                control2 = (
                    points[i][0] + dx * 0.5,
                    points[i][1] + dy * 0.5
                )
            elif i == len(points) - 2:
                # Last segment
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                control1 = (
                    points[i][0] + dx * 0.5,
                    points[i][1] + dy * 0.5
                )
                control2 = points[i + 1]
            else:
                # Middle segments
                dx1 = points[i + 1][0] - points[i - 1][0]
                dy1 = points[i + 1][1] - points[i - 1][1]
                dx2 = points[i + 2][0] - points[i][0]
                dy2 = points[i + 2][1] - points[i][1]
                
                control1 = (
                    points[i][0] + dx1 * 0.25,
                    points[i][1] + dy1 * 0.25
                )
                control2 = (
                    points[i + 1][0] - dx2 * 0.25,
                    points[i + 1][1] - dy2 * 0.25
                )
            
            renderer.draw_bezier(
                start=points[i],
                control1=control1,
                control2=control2,
                end=points[i + 1],
                color=style.color,
                width=style.line_width
            )
    
    def _draw_points(
        self,
        renderer: Renderer,
        points: List[Tuple[float, float]]
    ) -> None:
        """Draw points.
        
        Args:
            renderer: The renderer to use
            points: List of points to draw
        """
        style = self.style
        
        for point in points:
            renderer.draw_circle(
                center=point,
                radius=style.point_size / 2,
                color=style.color,
                fill=True
            )
    
    def _render_indicator(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render the oscillator overlay.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        if self._result is None or len(self._result) == 0:
            return
        
        values = self._get_values()
        
        # Get visible data range
        visible_data = chart.get_visible_data()
        if visible_data is None:
            return
        
        visible_values = values[visible_data.index]
        
        # Convert to pixel coordinates
        points = []
        for idx, val in zip(visible_values.index, visible_values):
            if pd.isna(val):
                continue
                
            # Convert index to datetime if it's not already
            if isinstance(idx, (int, np.integer)):
                timestamp = visible_values.index[idx]
            else:
                timestamp = pd.Timestamp(idx)
                
            point = chart.to_pixel_coords(timestamp, val)
            points.append(point)
        
        if len(points) < 2:
            return
        
        # Get x range for level lines
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        
        # Draw levels
        self._draw_levels(chart, renderer, (min_x, max_x))
        
        # Draw fill
        self._draw_fill(chart, renderer, points)
        
        # Draw line
        draw_line = self._draw_smooth_line if self.style.smooth else self._draw_line
        draw_line(renderer, points)
        
        # Draw points if requested
        if self.style.show_points:
            self._draw_points(renderer, points) 