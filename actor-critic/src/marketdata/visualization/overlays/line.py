"""Line-based indicator overlays."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd

from .base import IndicatorOverlay, OverlayStyle
from ..utils.renderer import Renderer

if TYPE_CHECKING:
    from ..components.base_chart import BaseChart


@dataclass
class LineStyle(OverlayStyle):
    """Line overlay style configuration."""
    smooth: bool = False  # Use bezier curves for smoothing
    fill_opacity: float = 0.0  # 0.0 for no fill
    fill_color: Optional[Tuple[float, float, float]] = None
    value_column: str = 'close'  # For indicators with multiple columns


class LineOverlay(IndicatorOverlay):
    """Line-based indicator overlay."""
    
    def __init__(
        self,
        indicator: 'BaseIndicator',
        style: Optional[LineStyle] = None
    ):
        """Initialize the line overlay.
        
        Args:
            indicator: The indicator to overlay
            style: Line style configuration
        """
        super().__init__(indicator, style or LineStyle())
    
    def _render_indicator(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render the indicator line.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        style = self.style
        values = self._get_values()
        
        if values is None or len(values) == 0:
            return
        
        # Get visible data range
        visible_data = chart.get_visible_data()
        if visible_data is None:
            visible_values = values  # Use all values if no visible range is specified
        else:
            # Get values within the visible range
            visible_values = values[visible_data.index]
        
        # Convert to pixel coordinates
        points = [
            chart.to_pixel_coords(pd.Timestamp(idx), val)
            for idx, val in zip(visible_values.index, visible_values)
            if not pd.isna(val)
        ]
        
        if len(points) < 2:
            return
        
        # Draw fill if requested
        if style.fill_opacity > 0:
            self._draw_fill(chart, renderer, points)
        
        # Draw line
        if style.smooth:
            self._draw_smooth_line(renderer, points)
        else:
            self._draw_line(renderer, points)
        
        # Draw points if requested
        if style.show_points:
            self._draw_points(renderer, points)
    
    def _get_values(self) -> Optional[pd.Series]:
        """Get values to plot.
        
        Returns:
            pd.Series: Values to plot
        """
        if self._result is None:
            return None
        
        if isinstance(self._result, pd.Series):
            return self._result
        elif isinstance(self._result, pd.DataFrame):
            return self._result[self.style.value_column]
        else:
            return None
    
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
        fill_color = style.fill_color or style.color
        
        # Create polygon points
        polygon_points = points.copy()
        
        # Add bottom corners
        bottom_y = chart.dimensions.height - chart.dimensions.margin_bottom
        polygon_points.append((points[-1][0], bottom_y))
        polygon_points.append((points[0][0], bottom_y))
        
        # Draw fill
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