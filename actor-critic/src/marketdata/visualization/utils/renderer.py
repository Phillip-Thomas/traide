"""Chart rendering utilities."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import cairo


class Renderer(ABC):
    """Abstract base class for chart renderers."""
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the rendering surface."""
        pass
    
    @abstractmethod
    def draw_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        color: Tuple[float, float, float],
        width: float = 1.0,
        dash: Optional[List[float]] = None
    ) -> None:
        """Draw a line.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            color: Line color (r, g, b)
            width: Line width
            dash: Optional dash pattern
        """
        pass
    
    @abstractmethod
    def draw_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Union[Tuple[float, float, float], Tuple[float, float, float, float]],
        fill: bool = True,
        line_width: float = 1.0
    ) -> None:
        """Draw a rectangle.
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Rectangle width
            height: Rectangle height
            color: Fill/stroke color (r, g, b) or (r, g, b, a)
            fill: Whether to fill the rectangle
            line_width: Stroke width if not filled
        """
        pass
    
    @abstractmethod
    def draw_text(
        self,
        text: str,
        pos: Tuple[float, float],
        color: Tuple[float, float, float],
        font_size: float = 12.0,
        align: str = 'left',
        baseline: str = 'top'
    ) -> None:
        """Draw text.
        
        Args:
            text: Text to draw
            pos: Position (x, y)
            color: Text color (r, g, b)
            font_size: Font size in points
            align: Text alignment ('left', 'center', 'right')
            baseline: Baseline alignment ('top', 'middle', 'bottom')
        """
        pass
    
    @abstractmethod
    def draw_polygon(
        self,
        points: List[Tuple[float, float]],
        color: Tuple[float, float, float, float],
        fill: bool = True,
        line_width: float = 1.0
    ) -> None:
        """Draw a polygon.
        
        Args:
            points: List of points [(x, y), ...]
            color: Fill/stroke color (r, g, b, a)
            fill: Whether to fill the polygon
            line_width: Stroke width if not filled
        """
        pass
    
    @abstractmethod
    def draw_circle(
        self,
        center: Tuple[float, float],
        radius: float,
        color: Tuple[float, float, float],
        fill: bool = True,
        line_width: float = 1.0
    ) -> None:
        """Draw a circle.
        
        Args:
            center: Center point (x, y)
            radius: Circle radius
            color: Fill/stroke color (r, g, b)
            fill: Whether to fill the circle
            line_width: Stroke width if not filled
        """
        pass
    
    @abstractmethod
    def draw_bezier(
        self,
        start: Tuple[float, float],
        control1: Tuple[float, float],
        control2: Tuple[float, float],
        end: Tuple[float, float],
        color: Tuple[float, float, float],
        width: float = 1.0
    ) -> None:
        """Draw a cubic bezier curve.
        
        Args:
            start: Start point (x, y)
            control1: First control point (x, y)
            control2: Second control point (x, y)
            end: End point (x, y)
            color: Line color (r, g, b)
            width: Line width
        """
        pass


class CairoRenderer(Renderer):
    """Cairo-based renderer implementation."""
    
    def __init__(self, surface: cairo.Surface):
        """Initialize the renderer.
        
        Args:
            surface: Cairo surface to render on
        """
        self.surface = surface
        self.context = cairo.Context(surface)
        
        # Set default font
        self.context.select_font_face(
            'Arial',
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
    
    def clear(self) -> None:
        """Clear the rendering surface."""
        self.context.save()
        self.context.set_source_rgb(1, 1, 1)  # White
        self.context.paint()
        self.context.restore()
    
    def draw_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        color: Tuple[float, float, float],
        width: float = 1.0,
        dash: Optional[List[float]] = None
    ) -> None:
        """Draw a line."""
        self.context.save()
        
        self.context.set_source_rgb(*color)
        self.context.set_line_width(width)
        
        if dash:
            self.context.set_dash(dash)
        
        self.context.move_to(*start)
        self.context.line_to(*end)
        self.context.stroke()
        
        self.context.restore()
    
    def draw_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Union[Tuple[float, float, float], Tuple[float, float, float, float]],
        fill: bool = True,
        line_width: float = 1.0
    ) -> None:
        """Draw a rectangle."""
        self.context.save()
        
        if len(color) == 3:
            self.context.set_source_rgb(*color)
        else:
            self.context.set_source_rgba(*color)
            
        self.context.set_line_width(line_width)
        
        self.context.rectangle(x, y, width, height)
        if fill:
            self.context.fill()
        else:
            self.context.stroke()
        
        self.context.restore()
    
    def draw_text(
        self,
        text: str,
        pos: Tuple[float, float],
        color: Tuple[float, float, float],
        font_size: float = 12.0,
        align: str = 'left',
        baseline: str = 'top'
    ) -> None:
        """Draw text."""
        self.context.save()
        
        self.context.set_source_rgb(*color)
        self.context.set_font_size(font_size)
        
        # Get text extents
        extents = self.context.text_extents(text)
        
        # Calculate position adjustments
        x_offset = {
            'left': 0,
            'center': -extents.width / 2,
            'right': -extents.width
        }.get(align, 0)
        
        y_offset = {
            'top': extents.height,
            'middle': extents.height / 2,
            'bottom': 0
        }.get(baseline, extents.height)
        
        # Draw text
        self.context.move_to(pos[0] + x_offset, pos[1] + y_offset)
        self.context.show_text(text)
        
        self.context.restore()
    
    def draw_polygon(
        self,
        points: List[Tuple[float, float]],
        color: Tuple[float, float, float, float],
        fill: bool = True,
        line_width: float = 1.0
    ) -> None:
        """Draw a polygon."""
        if not points:
            return
        
        self.context.save()
        
        self.context.set_source_rgba(*color)
        self.context.set_line_width(line_width)
        
        self.context.move_to(*points[0])
        for point in points[1:]:
            self.context.line_to(*point)
        self.context.close_path()
        
        if fill:
            self.context.fill()
        else:
            self.context.stroke()
        
        self.context.restore()
    
    def draw_circle(
        self,
        center: Tuple[float, float],
        radius: float,
        color: Tuple[float, float, float],
        fill: bool = True,
        line_width: float = 1.0
    ) -> None:
        """Draw a circle."""
        self.context.save()
        
        self.context.set_source_rgb(*color)
        self.context.set_line_width(line_width)
        
        self.context.arc(center[0], center[1], radius, 0, 2 * np.pi)
        if fill:
            self.context.fill()
        else:
            self.context.stroke()
        
        self.context.restore()
    
    def draw_bezier(
        self,
        start: Tuple[float, float],
        control1: Tuple[float, float],
        control2: Tuple[float, float],
        end: Tuple[float, float],
        color: Tuple[float, float, float],
        width: float = 1.0
    ) -> None:
        """Draw a cubic bezier curve."""
        self.context.save()
        
        self.context.set_source_rgb(*color)
        self.context.set_line_width(width)
        
        self.context.move_to(*start)
        self.context.curve_to(
            control1[0], control1[1],
            control2[0], control2[1],
            end[0], end[1]
        )
        self.context.stroke()
        
        self.context.restore()
    
    def save(self) -> None:
        """Save the current graphics state."""
        self.context.save()
    
    def restore(self) -> None:
        """Restore the last saved graphics state."""
        self.context.restore()
    
    def translate(self, dx: float, dy: float) -> None:
        """Translate the coordinate system.
        
        Args:
            dx: X translation
            dy: Y translation
        """
        self.context.translate(dx, dy)


class ChartRenderer:
    """High-level chart renderer."""
    
    def __init__(
        self,
        chart: 'BaseChart',
        renderer: Renderer
    ):
        """Initialize the chart renderer.
        
        Args:
            chart: The chart to render
            renderer: The renderer to use
        """
        from ..components.base_chart import BaseChart  # Import here to avoid circular dependency
        
        if not isinstance(chart, BaseChart):
            raise TypeError("Chart must be an instance of BaseChart")
        
        self.chart = chart
        self.renderer = renderer
    
    def render(self) -> None:
        """Render the chart."""
        # Clear the surface
        self.renderer.clear()
        
        # Draw chart components
        self._draw_grid()
        self._draw_axes()
        self._draw_data()
        
        if self.chart.scales.show_crosshair:
            self._draw_crosshair()
    
    def _draw_grid(self) -> None:
        """Draw grid lines."""
        if not self.chart.scales.show_grid:
            return
        
        # Draw vertical grid lines
        x_ticks = self.chart.scales.x.ticks()
        for x in x_ticks:
            self.renderer.draw_line(
                start=(x, self.chart.dimensions.margin_top),
                end=(x, self.chart.dimensions.height - self.chart.dimensions.margin_bottom),
                color=(0.9, 0.9, 0.9),
                width=1.0,
                dash=[2, 2]
            )
        
        # Draw horizontal grid lines
        y_ticks = self.chart.scales.y.ticks()
        for y in y_ticks:
            self.renderer.draw_line(
                start=(self.chart.dimensions.margin_left, y),
                end=(self.chart.dimensions.width - self.chart.dimensions.margin_right, y),
                color=(0.9, 0.9, 0.9),
                width=1.0,
                dash=[2, 2]
            )
    
    def _draw_axes(self) -> None:
        """Draw chart axes."""
        # Draw x-axis line
        self.renderer.draw_line(
            start=(
                self.chart.dimensions.margin_left,
                self.chart.dimensions.height - self.chart.dimensions.margin_bottom
            ),
            end=(
                self.chart.dimensions.width - self.chart.dimensions.margin_right,
                self.chart.dimensions.height - self.chart.dimensions.margin_bottom
            ),
            color=(0.2, 0.2, 0.2),
            width=1.0
        )
        
        # Draw y-axis line
        self.renderer.draw_line(
            start=(
                self.chart.dimensions.margin_left,
                self.chart.dimensions.margin_top
            ),
            end=(
                self.chart.dimensions.margin_left,
                self.chart.dimensions.height - self.chart.dimensions.margin_bottom
            ),
            color=(0.2, 0.2, 0.2),
            width=1.0
        )
        
        # Draw x-axis labels
        x_ticks = self.chart.scales.x.ticks()
        for x in x_ticks:
            value = self.chart.scales.x.invert(x)
            if isinstance(value, datetime):
                label = value.strftime(self.chart.scales.time_format)
            else:
                label = f"{value:{self.chart.scales.price_format}}"
            
            self.renderer.draw_text(
                text=label,
                pos=(
                    x,
                    self.chart.dimensions.height - self.chart.dimensions.margin_bottom + 15
                ),
                color=(0.2, 0.2, 0.2),
                font_size=10,
                align='center',
                baseline='top'
            )
        
        # Draw y-axis labels
        y_ticks = self.chart.scales.y.ticks()
        for y in y_ticks:
            value = self.chart.scales.y.invert(y)
            label = f"{value:{self.chart.scales.price_format}}"
            
            self.renderer.draw_text(
                text=label,
                pos=(
                    self.chart.dimensions.margin_left - 5,
                    y
                ),
                color=(0.2, 0.2, 0.2),
                font_size=10,
                align='right',
                baseline='middle'
            )
    
    def _draw_data(self) -> None:
        """Draw chart data."""
        pass  # Implemented by subclasses
    
    def _draw_crosshair(self) -> None:
        """Draw crosshair."""
        pass  # Implemented by subclasses 