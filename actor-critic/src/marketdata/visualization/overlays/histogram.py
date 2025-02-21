"""Histogram overlay for technical indicators."""

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
class HistogramStyle(OverlayStyle):
    """Style configuration for histogram overlays."""
    
    # Histogram-specific style options
    value_column: str = 'histogram'
    positive_color: Optional[Tuple[float, float, float]] = None
    negative_color: Optional[Tuple[float, float, float]] = None
    bar_width: float = 0.8  # Width of bars as fraction of x-scale step
    zero_line_color: Optional[Tuple[float, float, float]] = None
    zero_line_width: float = 1.0
    zero_line_style: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        super().__post_init__()
        if self.positive_color is None:
            self.positive_color = (0.0, 0.8, 0.0)  # Green
        if self.negative_color is None:
            self.negative_color = (0.8, 0.0, 0.0)  # Red
        if self.zero_line_color is None:
            self.zero_line_color = (0.5, 0.5, 0.5)  # Gray


class HistogramOverlay(IndicatorOverlay):
    """Overlay for histogram-based indicators like MACD."""
    
    def __init__(self, indicator, style: Optional[HistogramStyle] = None):
        """Initialize the histogram overlay.
        
        Args:
            indicator: The indicator to overlay
            style: Optional style configuration
        """
        if style is None:
            style = HistogramStyle()
        super().__init__(indicator, style)
    
    def _get_values(self) -> pd.Series:
        """Get the values to plot.
        
        Returns:
            Series with histogram values
        """
        if isinstance(self._result, pd.Series):
            return self._result
        elif isinstance(self._result, pd.DataFrame):
            if self.style.value_column not in self._result.columns:
                raise ValueError(f"Result DataFrame missing required column: {self.style.value_column}")
            return self._result[self.style.value_column]
        else:
            raise ValueError("Histogram overlay requires Series or DataFrame result")
    
    def _draw_zero_line(self, chart: 'BaseChart', renderer: Renderer, x_range: Tuple[float, float]):
        """Draw the zero line.
        
        Args:
            chart: The chart to draw on
            renderer: The renderer to use
            x_range: Tuple of (min_x, max_x) coordinates
        """
        style = self.style
        if style.zero_line_width <= 0:
            return
        
        y = chart.scales.y.transform(0)
        renderer.draw_line(
            start=(x_range[0], y),
            end=(x_range[1], y),
            color=style.zero_line_color,
            width=style.zero_line_width,
            dash=style.zero_line_style
        )
    
    def _draw_bars(self, chart: 'BaseChart', renderer: Renderer, points: list):
        """Draw histogram bars.
        
        Args:
            chart: The chart to draw on
            renderer: The renderer to use
            points: List of (x, y, value) points
        """
        if not points:
            return
        
        style = self.style
        x_step = chart.scales.x.step
        bar_width = x_step * style.bar_width
        zero_y = chart.scales.y.transform(0)
        
        for x, y, value in points:
            # Determine bar color based on value
            color = style.positive_color if value >= 0 else style.negative_color
            
            # Calculate bar dimensions
            bar_x = x - bar_width / 2
            bar_y = min(y, zero_y)
            bar_height = abs(y - zero_y)
            
            renderer.draw_rect(
                x=bar_x,
                y=bar_y,
                width=bar_width,
                height=bar_height,
                color=color,
                fill=True
            )
    
    def _render_indicator(self, chart: 'BaseChart', renderer: Renderer):
        """Render the histogram overlay.
        
        Args:
            chart: The chart to draw on
            renderer: The renderer to use
        """
        if self._result is None or len(self._result) == 0:
            return
        
        values = self._get_values()
        points = []
        min_x = float('inf')
        max_x = float('-inf')
        
        for idx, value in values.items():
            if pd.isna(value):
                continue
            
            # Convert index to datetime if it's not already
            if isinstance(idx, (int, np.integer)):
                timestamp = values.index[idx]
            else:
                timestamp = pd.Timestamp(idx)
            
            x = chart.scales.x.transform(timestamp)
            y = chart.scales.y.transform(value)
            points.append((x, y, value))
            
            min_x = min(min_x, x)
            max_x = max(max_x, x)
        
        if not points:
            return
        
        # Draw zero line
        self._draw_zero_line(chart, renderer, (min_x, max_x))
        
        # Draw bars
        self._draw_bars(chart, renderer, points) 