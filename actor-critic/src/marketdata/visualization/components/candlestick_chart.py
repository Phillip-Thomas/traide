"""Candlestick chart component."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .base_chart import BaseChart, ChartDimensions, ChartScales
from ..utils.renderer import Renderer


@dataclass
class CandlestickStyle:
    """Candlestick chart style configuration."""
    up_color: Tuple[float, float, float] = (0.0, 0.8, 0.0)  # Green
    down_color: Tuple[float, float, float] = (0.8, 0.0, 0.0)  # Red
    wick_width: float = 1.0
    body_width: float = 8.0
    hollow_up: bool = False
    hollow_down: bool = False


class CandlestickChart(BaseChart):
    """Candlestick chart component."""
    
    def __init__(
        self,
        dimensions: ChartDimensions,
        scales: Optional[ChartScales] = None,
        style: Optional[CandlestickStyle] = None
    ):
        """Initialize the candlestick chart.
        
        Args:
            dimensions: Chart dimensions configuration
            scales: Chart scales configuration
            style: Candlestick style configuration
        """
        super().__init__(dimensions, scales)
        self.style = style or CandlestickStyle()
    
    def render(self, renderer: Renderer) -> None:
        """Render the candlestick chart.
        
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
        
        # Draw candlesticks
        self._draw_candlesticks(renderer)
        
        # Draw overlays
        self.overlay_manager.render_overlays(self, renderer)
        
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
        
        # Draw horizontal grid lines (price)
        price_range = visible_data['high'].max() - visible_data['low'].min()
        num_lines = 5
        price_step = price_range / num_lines
        
        for i in range(num_lines + 1):
            price = visible_data['low'].min() + i * price_step
            _, y = self.to_pixel_coords(visible_data['timestamp'].iloc[0], price)
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
            x, y = self.to_pixel_coords(t, visible_data['low'].min())
            renderer.draw_text(
                t.strftime(time_format),
                (x, y + 5),
                (0.2, 0.2, 0.2),  # Dark gray
                align='center',
                baseline='top'
            )
        
        # Draw price axis
        price_range = visible_data['high'].max() - visible_data['low'].min()
        num_labels = 5
        price_step = price_range / num_labels
        
        for i in range(num_labels + 1):
            price = visible_data['low'].min() + i * price_step
            x, y = self.to_pixel_coords(visible_data['timestamp'].iloc[0], price)
            renderer.draw_text(
                f'{price:.2f}',
                (self.dimensions.margin_left - 5, y),
                (0.2, 0.2, 0.2),  # Dark gray
                align='right',
                baseline='middle'
            )
    
    def _draw_candlesticks(self, renderer: Renderer) -> None:
        """Draw candlesticks.
        
        Args:
            renderer: Renderer implementation
        """
        # Get visible data
        visible_data = self.get_visible_data()
        if visible_data is None:
            return
        
        # Calculate candlestick width based on data density
        time_range = visible_data['timestamp'].max() - visible_data['timestamp'].min()
        num_candles = len(visible_data)
        if num_candles > 1:
            candle_spacing = time_range.total_seconds() / (num_candles - 1)
            x_range = self.dimensions.inner_width
            pixels_per_candle = x_range / num_candles
            body_width = min(self.style.body_width, pixels_per_candle * 0.8)
        else:
            body_width = self.style.body_width
        
        # Draw each candlestick
        for _, row in visible_data.iterrows():
            # Calculate coordinates
            timestamp = row['timestamp']
            center_x, _ = self.to_pixel_coords(timestamp, 0)
            open_x, open_y = self.to_pixel_coords(timestamp, row['open'])
            high_x, high_y = self.to_pixel_coords(timestamp, row['high'])
            low_x, low_y = self.to_pixel_coords(timestamp, row['low'])
            close_x, close_y = self.to_pixel_coords(timestamp, row['close'])
            
            # Determine if bullish or bearish
            is_bullish = row['close'] >= row['open']
            color = self.style.up_color if is_bullish else self.style.down_color
            hollow = (
                (is_bullish and self.style.hollow_up) or
                (not is_bullish and self.style.hollow_down)
            )
            
            # Draw wick
            renderer.draw_line(
                (center_x, high_y),
                (center_x, low_y),
                color,
                width=self.style.wick_width
            )
            
            # Draw body
            body_left = center_x - body_width / 2
            body_top = min(open_y, close_y)
            body_height = abs(close_y - open_y)
            
            if hollow:
                renderer.draw_rect(
                    body_left,
                    body_top,
                    body_width,
                    body_height,
                    color,
                    fill=False,
                    line_width=self.style.wick_width
                )
            else:
                renderer.draw_rect(
                    body_left,
                    body_top,
                    body_width,
                    body_height,
                    color,
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
        timestamp, price = self.from_pixel_coords(x, y)
        coord_text = f'{timestamp.strftime(self.scales.time_format)} - {price:.2f}'
        
        renderer.draw_text(
            coord_text,
            (x + 5, y - 5),
            (0.2, 0.2, 0.2),  # Dark gray
            align='left',
            baseline='bottom'
        ) 