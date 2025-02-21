"""Base classes for technical indicators."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class IndicatorParams:
    """Base class for indicator parameters."""
    pass


@dataclass
class IndicatorStyle:
    """Base class for indicator visualization style."""
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0)  # Default blue
    line_width: float = 1.0
    line_style: str = 'solid'  # 'solid', 'dashed', 'dotted'
    show_points: bool = False
    point_size: float = 4.0
    z_index: int = 0
    visible: bool = True
    label: str = ''


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    values: pd.Series
    metadata: Dict = field(default_factory=dict)
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)


class BaseIndicator(ABC):
    """Abstract base class for technical indicators."""
    
    def __init__(
        self,
        params: Optional[IndicatorParams] = None,
        style: Optional[IndicatorStyle] = None
    ):
        """Initialize the indicator.
        
        Args:
            params: Indicator parameters
            style: Visualization style
        """
        self.params = params or self._default_params()
        self.style = style or IndicatorStyle()
        
        # Cache for calculation results
        self._cache: Dict[str, IndicatorResult] = {}
        self._last_update: Optional[pd.Timestamp] = None
    
    @abstractmethod
    def _default_params(self) -> IndicatorParams:
        """Get default parameters for the indicator.
        
        Returns:
            IndicatorParams: Default parameters
        """
        pass
    
    @abstractmethod
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the indicator values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: Calculation results
        """
        pass
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate or get cached indicator values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: Calculation results
        """
        # Generate cache key from data and parameters
        cache_key = self._generate_cache_key(data)
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if self._is_cache_valid(cached, data):
                return cached
        
        # Calculate new result
        result = self._calculate(data)
        
        # Update cache
        self._cache[cache_key] = result
        self._last_update = pd.Timestamp.now()
        
        # Clean old cache entries
        self._clean_cache()
        
        return result
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key from input data.
        
        Args:
            data: Input data
            
        Returns:
            str: Cache key
        """
        # Use data range and length as key
        start = data.index[0]
        end = data.index[-1]
        length = len(data)
        return f"{start}_{end}_{length}"
    
    def _is_cache_valid(
        self,
        cached: IndicatorResult,
        data: pd.DataFrame
    ) -> bool:
        """Check if cached result is still valid.
        
        Args:
            cached: Cached result
            data: New input data
            
        Returns:
            bool: True if cache is valid
        """
        # Check if data matches cached result
        if len(cached.values) != len(data):
            return False
        
        if not cached.values.index.equals(data.index):
            return False
        
        # Check if cache is too old (> 1 hour)
        age = pd.Timestamp.now() - cached.timestamp
        if age.total_seconds() > 3600:
            return False
        
        return True
    
    def _clean_cache(self) -> None:
        """Clean old cache entries."""
        # Keep only last 10 results
        if len(self._cache) > 10:
            # Sort by timestamp and keep newest
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp,
                reverse=True
            )
            for key in sorted_keys[10:]:
                del self._cache[key]


class IndicatorRegistry:
    """Registry for technical indicators."""
    
    _indicators: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, indicator_class: type) -> None:
        """Register an indicator class.
        
        Args:
            name: Indicator name
            indicator_class: Indicator class to register
        """
        if not issubclass(indicator_class, BaseIndicator):
            raise ValueError(
                f"Class {indicator_class.__name__} must inherit from BaseIndicator"
            )
        cls._indicators[name] = indicator_class
    
    @classmethod
    def create(
        cls,
        name: str,
        params: Optional[IndicatorParams] = None,
        style: Optional[IndicatorStyle] = None
    ) -> BaseIndicator:
        """Create an indicator instance.
        
        Args:
            name: Indicator name
            params: Indicator parameters
            style: Visualization style
            
        Returns:
            BaseIndicator: Indicator instance
        """
        if name not in cls._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        
        indicator_class = cls._indicators[name]
        return indicator_class(params, style)
    
    @classmethod
    def list_indicators(cls) -> List[str]:
        """Get list of registered indicators.
        
        Returns:
            List[str]: List of indicator names
        """
        return list(cls._indicators.keys())


def register_indicator(name: str):
    """Decorator to register an indicator class.
    
    Args:
        name: Indicator name
    """
    def decorator(cls):
        IndicatorRegistry.register(name, cls)
        return cls
    return decorator 