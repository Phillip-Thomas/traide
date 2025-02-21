"""Area chart component."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .base_chart import BaseChart, ChartDimensions, ChartScales
from ..utils.renderer import Renderer


@dataclass
class AreaStyle:
    """Area chart style configuration."""
    fill_color: Tuple[float, float, float] = (0.2, 0.6, 1.0)  # Blue
    fill_opacity: float = 0.2
    line_color: Optional[Tuple[float, float, float]] = None  # None to use fill_color
    line_width: float = 2.0
    smooth: bool = False
    show_points: bool = False
    point_size: float = 4.0
    use_gradient: bool = False
    gradient_stops: Optional[List[Tuple[float, Tuple[float, float, float, float]]]] = None


class AreaChart(BaseChart):
    """Area chart component."""
    
    def __init__(
        self,
        dimensions: ChartDimensions,
        scales: Optional[ChartScales] = None,
        style: Optional[AreaStyle] = None,
        value_column: str = 'close',
        base_value: float = 0.0
    ):
        """Initialize the area chart.
        
        Args:
            dimensions: Chart dimensions configuration
            scales: Chart scales configuration
            style: Area style configuration
            value_column: Column to plot
            base_value: Base value for area (usually 0)
        """
        super().__init__(dimensions, scales)
        self.style = style or AreaStyle()
        self.value_column = value_column
        self.base_value = base_value
        
        # Set line color if not specified
        if self.style.line_color is None:
            self.style.line_color = self.style.fill_color
    
    def render(self, renderer: Renderer) -> None:
        """Render the area chart.
        
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
        
        # Draw area
        self._draw_area(renderer)
        
        # Draw crosshair if enabled
        if self.scales.show_crosshair:
            self._draw_crosshair(renderer)
    
    def _draw_grid(self, renderer: Renderer) -> None:
        """Draw chart grid.
        
        Args:
            renderer: Renderer implementation
        """
        # Get visible data range
        visible_data = self.get_visible_data()
        if visible_data is None:
            return
        
        # Draw vertical grid lines (time)
        time_range = visible_data['timestamp'].max() - visible_data['timestamp'].min()
        if time_range.total_seconds() > 86400:  # More than a day
            freq = 'D'
        elif time_range.total_seconds() > 3600:  # More than an hour
            freq = 'H'
        else:
            freq = '15min'
        
        grid_times = pd.date_range(
            visible_data['timestamp'].min(),
            visible_data['timestamp'].max(),
            freq=freq
        )
        
        for t in grid_times:
            x, _ = self.to_pixel_coords(t, 0)
            renderer.draw_line(
                (x, self.dimensions.margin_top),
                (x, self.dimensions.height - self.dimensions.margin_bottom),
                (0.9, 0.9, 0.9),  # Light gray
                width=1.0,
                dash=[5, 5]
            )
        
        # Draw horizontal grid lines (value)
        value_range = visible_data[self.value_column].max() - visible_data[self.value_column].min()
        num_lines = 5
        value_step = value_range / num_lines
        
        for i in range(num_lines + 1):
            value = visible_data[self.value_column].min() + i * value_step
            _, y = self.to_pixel_coords(visible_data['timestamp'].iloc[0], value)
            renderer.draw_line(
                (self.dimensions.margin_left, y),
                (self.dimensions.width - self.dimensions.margin_right, y),
                (0.9, 0.9, 0.9),  # Light gray
                width=1.0,
                dash=[5, 5]
            )
    
    def _draw_axes(self, renderer: Renderer) -> None:
        """Draw chart axes.
        
        Args:
            renderer: Renderer implementation
        """
        # Get visible data range
        visible_data = self.get_visible_data()
        if visible_data is None:
            return
        
        # Draw time axis
        time_range = visible_data['timestamp'].max() - visible_data['timestamp'].min()
        if time_range.total_seconds() > 86400:  # More than a day
            time_format = '%Y-%m-%d'
        else:
            time_format = '%H:%M'
        
        for t in pd.date_range(
            visible_data['timestamp'].min(),
            visible_data['timestamp'].max(),
            periods=5
        ):
            x, y = self.to_pixel_coords(t, visible_data[self.value_column].min())
            renderer.draw_text(
                t.strftime(time_format),
                (x, y + 5),
                (0.2, 0.2, 0.2),  # Dark gray
                align='center',
                baseline='top'
            )
        
        # Draw value axis
        value_range = visible_data[self.value_column].max() - visible_data[self.value_column].min()
        num_labels = 5
        value_step = value_range / num_labels
        
        for i in range(num_labels + 1):
            value = visible_data[self.value_column].min() + i * value_step
            x, y = self.to_pixel_coords(visible_data['timestamp'].iloc[0], value)
            renderer.draw_text(
                f'{value:.2f}',
                (self.dimensions.margin_left - 5, y),
                (0.2, 0.2, 0.2),  # Dark gray
                align='right',
                baseline='middle'
            )
    
    def _draw_area(self, renderer: Renderer) -> None:
        """Draw the area.
        
        Args:
            renderer: Renderer implementation
        """
        # Get visible data
        visible_data = self.get_visible_data()
        if visible_data is None or len(visible_data) < 2:
            return
        
        # Get coordinates for line points
        points = [
            self.to_pixel_coords(row['timestamp'], row[self.value_column])
            for _, row in visible_data.iterrows()
        ]
        
        # Get base line points
        base_points = [
            self.to_pixel_coords(row['timestamp'], self.base_value)
            for _, row in visible_data.iterrows()
        ]
        
        # Create polygon for fill
        fill_points = points.copy()
        fill_points.extend(base_points[::-1])  # Add base points in reverse
        
        # Draw fill
        if self.style.use_gradient and self.style.gradient_stops:
            # Create gradient fill
            gradient_color = self.style.gradient_stops[0][1]  # Use first stop color
            renderer.draw_polygon(
                fill_points,
                gradient_color,
                fill=True
            )
        else:
            # Solid fill
            fill_color = self.style.fill_color + (self.style.fill_opacity,)
            renderer.draw_polygon(
                fill_points,
                fill_color,
                fill=True
            )
        
        # Draw line
        if self.style.smooth:
            # Use cubic interpolation for smooth line
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            
            # Calculate control points for cubic Bezier curves
            control_points = self._calculate_bezier_control_points(x_points, y_points)
            
            # Draw smooth curve
            for i in range(len(points) - 1):
                renderer.draw_bezier(
                    points[i],
                    (control_points[0][i], control_points[1][i]),
                    (control_points[2][i], control_points[3][i]),
                    points[i + 1],
                    self.style.line_color,
                    width=self.style.line_width
                )
        else:
            # Draw straight line segments
            for i in range(len(points) - 1):
                renderer.draw_line(
                    points[i],
                    points[i + 1],
                    self.style.line_color,
                    width=self.style.line_width
                )
        
        # Draw points if enabled
        if self.style.show_points:
            for point in points:
                renderer.draw_circle(
                    point,
                    self.style.point_size,
                    self.style.line_color,
                    fill=True
                )
    
    def _calculate_bezier_control_points(
        self,
        x_points: List[float],
        y_points: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate control points for smooth curve.
        
        Args:
            x_points: List of x coordinates
            y_points: List of y coordinates
            
        Returns:
            Tuple of arrays containing control point coordinates
        """
        # Convert to numpy arrays
        x = np.array(x_points)
        y = np.array(y_points)
        
        # Calculate first control points
        n = len(x) - 1
        
        # Calculate first control points
        p1x = np.zeros(n)
        p1y = np.zeros(n)
        p2x = np.zeros(n)
        p2y = np.zeros(n)
        
        # Calculate first control points
        for i in range(n):
            # First control point
            if i == 0:
                p1x[i] = x[i] + (x[i+1] - x[i]) / 3
                p1y[i] = y[i] + (y[i+1] - y[i]) / 3
            else:
                p1x[i] = 2 * x[i] / 3 + x[i+1] / 3
                p1y[i] = 2 * y[i] / 3 + y[i+1] / 3
            
            # Second control point
            if i == n - 1:
                p2x[i] = x[i] + 2 * (x[i+1] - x[i]) / 3
                p2y[i] = y[i] + 2 * (y[i+1] - y[i]) / 3
            else:
                p2x[i] = x[i] / 3 + 2 * x[i+1] / 3
                p2y[i] = y[i] / 3 + 2 * y[i+1] / 3
        
        return p1x, p1y, p2x, p2y
    
    def _draw_crosshair(self, renderer: Renderer) -> None:
        """Draw chart crosshair.
        
        Args:
            renderer: Renderer implementation
        """
        if self._last_mouse_pos is None:
            return
        
        x, y = self._last_mouse_pos
        
        # Draw vertical line
        if (x >= self.dimensions.margin_left and
            x <= self.dimensions.width - self.dimensions.margin_right):
            renderer.draw_line(
                (x, self.dimensions.margin_top),
                (x, self.dimensions.height - self.dimensions.margin_bottom),
                (0.5, 0.5, 0.5),  # Gray
                width=1.0,
                dash=[2, 2]
            )
        
        # Draw horizontal line
        if (y >= self.dimensions.margin_top and
            y <= self.dimensions.height - self.dimensions.margin_bottom):
            renderer.draw_line(
                (self.dimensions.margin_left, y),
                (self.dimensions.width - self.dimensions.margin_right, y),
                (0.5, 0.5, 0.5),  # Gray
                width=1.0,
                dash=[2, 2]
            )
        
        # Draw coordinates
        timestamp, value = self.from_pixel_coords(x, y)
        coord_text = f'{timestamp.strftime(self.scales.time_format)} - {value:.2f}'
        
        renderer.draw_text(
            coord_text,
            (x + 5, y - 5),
            (0.2, 0.2, 0.2),  # Dark gray
            align='left',
            baseline='bottom'
        ) 