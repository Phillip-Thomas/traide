"""Bar chart component."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .base_chart import BaseChart, ChartDimensions, ChartScales
from ..utils.renderer import Renderer


@dataclass
class BarStyle:
    """Bar chart style configuration."""
    up_color: Tuple[float, float, float] = (0.0, 0.8, 0.0)  # Green
    down_color: Tuple[float, float, float] = (0.8, 0.0, 0.0)  # Red
    neutral_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # Gray
    bar_width: float = 8.0
    bar_spacing: float = 2.0
    opacity: float = 0.8


class BarChart(BaseChart):
    """Bar chart component."""
    
    def __init__(
        self,
        dimensions: ChartDimensions,
        scales: Optional[ChartScales] = None,
        style: Optional[BarStyle] = None,
        value_column: str = 'volume',
        color_by: Optional[str] = 'close'
    ):
        """Initialize the bar chart.
        
        Args:
            dimensions: Chart dimensions configuration
            scales: Chart scales configuration
            style: Bar style configuration
            value_column: Column to plot as bars
            color_by: Column to determine bar color (e.g., 'close' for price direction)
        """
        super().__init__(dimensions, scales)
        self.style = style or BarStyle()
        self.value_column = value_column
        self.color_by = color_by
    
    def render(self, renderer: Renderer) -> None:
        """Render the bar chart.
        
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
        
        # Draw bars
        self._draw_bars(renderer)
        
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
        value_range = visible_data[self.value_column].max()
        num_lines = 5
        value_step = value_range / num_lines
        
        for i in range(num_lines + 1):
            value = i * value_step
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
            x, y = self.to_pixel_coords(t, 0)
            renderer.draw_text(
                t.strftime(time_format),
                (x, y + 5),
                (0.2, 0.2, 0.2),  # Dark gray
                align='center',
                baseline='top'
            )
        
        # Draw value axis
        value_range = visible_data[self.value_column].max()
        num_labels = 5
        value_step = value_range / num_labels
        
        for i in range(num_labels + 1):
            value = i * value_step
            x, y = self.to_pixel_coords(visible_data['timestamp'].iloc[0], value)
            
            # Format large numbers
            if value >= 1_000_000:
                label = f'{value/1_000_000:.1f}M'
            elif value >= 1_000:
                label = f'{value/1_000:.1f}K'
            else:
                label = f'{value:.0f}'
            
            renderer.draw_text(
                label,
                (self.dimensions.margin_left - 5, y),
                (0.2, 0.2, 0.2),  # Dark gray
                align='right',
                baseline='middle'
            )
    
    def _draw_bars(self, renderer: Renderer) -> None:
        """Draw bars.
        
        Args:
            renderer: Renderer implementation
        """
        # Get visible data
        visible_data = self.get_visible_data()
        if visible_data is None:
            return
        
        # Calculate bar width based on data density
        time_range = visible_data['timestamp'].max() - visible_data['timestamp'].min()
        num_bars = len(visible_data)
        if num_bars > 1:
            bar_spacing = time_range.total_seconds() / (num_bars - 1)
            x_range = self.dimensions.inner_width
            pixels_per_bar = x_range / num_bars
            bar_width = min(self.style.bar_width, pixels_per_bar - self.style.bar_spacing)
        else:
            bar_width = self.style.bar_width
        
        # Get zero y-coordinate
        _, zero_y = self.to_pixel_coords(visible_data['timestamp'].iloc[0], 0)
        
        # Draw each bar
        for _, row in visible_data.iterrows():
            # Calculate bar coordinates
            center_x, _ = self.to_pixel_coords(row['timestamp'], 0)
            _, value_y = self.to_pixel_coords(row['timestamp'], row[self.value_column])
            
            # Calculate bar rectangle
            bar_left = center_x - bar_width / 2
            bar_top = min(value_y, zero_y)
            bar_height = abs(value_y - zero_y)
            
            # Determine color
            if self.color_by is None:
                color = self.style.neutral_color
            else:
                current_value = row[self.color_by]
                prev_value = self._data[self.color_by].shift(1).iloc[self._data.index.get_loc(row.name)]
                if pd.isna(prev_value):
                    prev_value = current_value
                
                if current_value > prev_value:
                    color = self.style.up_color
                elif current_value < prev_value:
                    color = self.style.down_color
                else:
                    color = self.style.neutral_color
            
            # Draw bar
            renderer.draw_rect(
                bar_left,
                bar_top,
                bar_width,
                bar_height,
                color + (self.style.opacity,),
                fill=True
            )
    
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
        
        # Format value
        if value >= 1_000_000:
            value_str = f'{value/1_000_000:.1f}M'
        elif value >= 1_000:
            value_str = f'{value/1_000:.1f}K'
        else:
            value_str = f'{value:.0f}'
        
        coord_text = f'{timestamp.strftime(self.scales.time_format)} - {value_str}'
        
        renderer.draw_text(
            coord_text,
            (x + 5, y - 5),
            (0.2, 0.2, 0.2),  # Dark gray
            align='left',
            baseline='bottom'
        ) 