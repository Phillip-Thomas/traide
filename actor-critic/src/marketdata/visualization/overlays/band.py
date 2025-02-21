"""Band overlay for technical indicators."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd

from .base import IndicatorOverlay, OverlayStyle
from ..utils.renderer import Renderer
from ..utils.scales import LinearScale, TimeScale

if TYPE_CHECKING:
    from ..components.base_chart import BaseChart


@dataclass
class BandStyle(OverlayStyle):
    """Style configuration for band overlays."""
    
    # Band-specific style options
    upper_column: str = 'upper'
    middle_column: str = 'middle'
    lower_column: str = 'lower'
    smooth: bool = False
    fill_opacity: float = 0.1
    fill_color: Optional[Tuple[float, float, float]] = None
    show_middle: bool = True
    middle_style: Optional[str] = 'dashed'


class BandOverlay(IndicatorOverlay):
    """Overlay for band-based indicators like Bollinger Bands."""
    
    def __init__(self, indicator, style: Optional[BandStyle] = None):
        """Initialize the band overlay.
        
        Args:
            indicator: The indicator to overlay
            style: Optional style configuration
        """
        if style is None:
            style = BandStyle()
        super().__init__(indicator, style)
    
    def _get_values(self) -> pd.DataFrame:
        """Get the values to plot.
        
        Returns:
            DataFrame with band values
        """
        if not isinstance(self._result, pd.DataFrame):
            raise ValueError("Band overlay requires DataFrame result")
        
        style = self.style
        required_columns = [style.upper_column, style.lower_column]
        if style.show_middle:
            required_columns.append(style.middle_column)
        
        missing_columns = [col for col in required_columns if col not in self._result.columns]
        if missing_columns:
            raise ValueError(f"Result DataFrame missing required columns: {missing_columns}")
        
        return self._result[required_columns]
    
    def _draw_fill(
        self,
        chart: 'BaseChart',
        renderer: Renderer,
        upper_points: List[Tuple[float, float]],
        lower_points: List[Tuple[float, float]]
    ) -> None:
        """Draw fill between bands.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
            upper_points: List of upper band points
            lower_points: List of lower band points
        """
        style = self.style
        if style.fill_opacity <= 0:
            return
        
        # Create polygon points
        polygon_points = upper_points.copy()
        polygon_points.extend(reversed(lower_points))
        
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
        points: List[Tuple[float, float]],
        color: Optional[Tuple[float, float, float]] = None,
        width: Optional[float] = None,
        dash: Optional[str] = None
    ) -> None:
        """Draw straight line segments.
        
        Args:
            renderer: The renderer to use
            points: List of points to draw
            color: Optional line color
            width: Optional line width
            dash: Optional dash pattern
        """
        style = self.style
        color = color or style.color
        width = width or style.line_width
        
        for i in range(len(points) - 1):
            renderer.draw_line(
                start=points[i],
                end=points[i + 1],
                color=color,
                width=width,
                dash=dash
            )
    
    def _draw_smooth_line(
        self,
        renderer: Renderer,
        points: List[Tuple[float, float]],
        color: Optional[Tuple[float, float, float]] = None,
        width: Optional[float] = None,
        dash: Optional[str] = None
    ) -> None:
        """Draw smooth line using bezier curves.
        
        Args:
            renderer: The renderer to use
            points: List of points to draw
            color: Optional line color
            width: Optional line width
            dash: Optional dash pattern
        """
        style = self.style
        color = color or style.color
        width = width or style.line_width
        
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
                color=color,
                width=width
            )
    
    def _draw_points(
        self,
        renderer: Renderer,
        points: List[Tuple[float, float]],
        color: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Draw points.
        
        Args:
            renderer: The renderer to use
            points: List of points to draw
            color: Optional point color
        """
        style = self.style
        color = color or style.color
        
        for point in points:
            renderer.draw_circle(
                center=point,
                radius=style.point_size / 2,
                color=color,
                fill=True
            )
    
    def _render_indicator(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render the band overlay.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        if self._result is None or len(self._result) == 0:
            return
        
        values = self._get_values()
        style = self.style
        
        # Get visible data range
        visible_data = chart.get_visible_data()
        if visible_data is None:
            return
        
        visible_values = values.loc[visible_data.index]
        
        # Convert to pixel coordinates
        upper_points = [
            chart.to_pixel_coords(pd.Timestamp(idx), val)
            for idx, val in zip(visible_values.index, visible_values[style.upper_column])
            if not pd.isna(val)
        ]
        
        lower_points = [
            chart.to_pixel_coords(pd.Timestamp(idx), val)
            for idx, val in zip(visible_values.index, visible_values[style.lower_column])
            if not pd.isna(val)
        ]
        
        if style.show_middle:
            middle_points = [
                chart.to_pixel_coords(pd.Timestamp(idx), val)
                for idx, val in zip(visible_values.index, visible_values[style.middle_column])
                if not pd.isna(val)
            ]
        else:
            middle_points = []
        
        if len(upper_points) < 2 or len(lower_points) < 2:
            return
        
        # Draw fill between bands
        self._draw_fill(chart, renderer, upper_points, lower_points)
        
        # Draw lines
        draw_line = self._draw_smooth_line if style.smooth else self._draw_line
        
        # Upper band
        draw_line(renderer, upper_points)
        
        # Lower band
        draw_line(renderer, lower_points)
        
        # Middle line
        if style.show_middle and middle_points:
            # Convert dash style to pattern
            dash_pattern = {
                'solid': None,
                'dashed': [6.0, 4.0],
                'dotted': [2.0, 2.0]
            }.get(style.middle_style, None)
            
            draw_line(renderer, middle_points, dash=dash_pattern)
        
        # Draw points if requested
        if style.show_points:
            self._draw_points(renderer, upper_points)
            self._draw_points(renderer, lower_points)
            if style.show_middle and middle_points:
                self._draw_points(renderer, middle_points) 