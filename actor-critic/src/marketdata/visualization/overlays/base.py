"""Base classes for indicator overlays."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from datetime import datetime

from ..utils.renderer import Renderer

if TYPE_CHECKING:
    from ..components.base_chart import BaseChart


@dataclass
class OverlayStyle:
    """Base class for all overlay styles."""
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0)  # Default blue
    line_width: float = 1.0
    line_style: str = 'solid'  # 'solid', 'dashed', 'dotted'
    opacity: float = 1.0
    show_points: bool = False
    point_size: float = 4.0
    z_index: int = 0
    visible: bool = True
    label: str = ''
    
    def __post_init__(self):
        """Post-initialization hook."""
        pass


class IndicatorOverlay(ABC):
    """Base class for indicator overlays."""
    
    def __init__(
        self,
        indicator: 'BaseIndicator',
        style: Optional[OverlayStyle] = None
    ):
        """Initialize the overlay.
        
        Args:
            indicator: The indicator to overlay
            style: Overlay style configuration
        """
        self.indicator = indicator
        self.style = style or OverlayStyle()
        self._result = None
        
    @property
    def visible(self) -> bool:
        """Get overlay visibility."""
        return self.style.visible
    
    @visible.setter
    def visible(self, value: bool):
        """Set overlay visibility."""
        self.style.visible = value
    
    @property
    def z_index(self) -> int:
        """Get overlay z-index."""
        return self.style.z_index
    
    @z_index.setter
    def z_index(self, value: int):
        """Set overlay z-index."""
        self.style.z_index = value
    
    def update(self, data: pd.DataFrame) -> None:
        """Update overlay with new data.
        
        Args:
            data: Input data with OHLCV columns
        """
        self._result = self.indicator.calculate(data)
    
    def render(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render the overlay.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        if not self.visible:
            return
        
        # Ensure we have data to render
        if self._result is None and chart._data is not None:
            self.update(chart._data)
        
        if self._result is None:
            return
        
        # Ensure scales are initialized
        if chart.scales.x is None or chart.scales.y is None:
            chart.scales.update_scales(
                x_domain=(
                    chart._x_domain[0] if chart._x_domain else datetime.now(),
                    chart._x_domain[1] if chart._x_domain else datetime.now()
                ),
                y_domain=(
                    chart._y_domain[0] if chart._y_domain else 0.0,
                    chart._y_domain[1] if chart._y_domain else 1.0
                ),
                dimensions=chart.dimensions
            )
        
        self._render_indicator(chart, renderer)
        self._render_legend(chart, renderer)
    
    @abstractmethod
    def _render_indicator(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render the indicator.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        pass
    
    def _render_legend(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render the legend entry.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        if not self.style.label:
            return
        
        # TODO: Implement legend rendering
        pass


class OverlayManager:
    """Manager for indicator overlays."""
    
    def __init__(self):
        """Initialize the overlay manager."""
        self.overlays: List[IndicatorOverlay] = []
    
    def add_overlay(self, overlay: IndicatorOverlay) -> None:
        """Add an overlay.
        
        Args:
            overlay: The overlay to add
        """
        self.overlays.append(overlay)
        self._sort_overlays()
    
    def remove_overlay(self, overlay: IndicatorOverlay) -> None:
        """Remove an overlay.
        
        Args:
            overlay: The overlay to remove
        """
        if overlay in self.overlays:
            self.overlays.remove(overlay)
    
    def clear_overlays(self) -> None:
        """Remove all overlays."""
        self.overlays.clear()
    
    def update_overlays(self, data: pd.DataFrame) -> None:
        """Update all overlays with new data.
        
        Args:
            data: Input data with OHLCV columns
        """
        for overlay in self.overlays:
            overlay.update(data)
    
    def render_overlays(self, chart: 'BaseChart', renderer: Renderer) -> None:
        """Render all overlays.
        
        Args:
            chart: The chart to render on
            renderer: The renderer to use
        """
        for overlay in self.overlays:
            overlay.render(chart, renderer)
    
    def _sort_overlays(self) -> None:
        """Sort overlays by z-index."""
        self.overlays.sort(key=lambda x: x.z_index) 