"""Scale utilities for chart visualization."""

from typing import Optional, Union, Tuple
from datetime import datetime
import numpy as np


class LinearScale:
    """Linear scale for numeric values."""
    
    def __init__(self, domain: Tuple[float, float], range: Tuple[float, float]):
        """Initialize the scale.
        
        Args:
            domain: Input domain (min, max)
            range: Output range (min, max)
        """
        self.domain = domain
        self.range = range
        self.step = (range[1] - range[0]) / (domain[1] - domain[0])
    
    def transform(self, value: float) -> float:
        """Transform a value from domain to range.
        
        Args:
            value: Input value
            
        Returns:
            Transformed value
        """
        domain_ratio = (value - self.domain[0]) / (self.domain[1] - self.domain[0])
        return self.range[0] + domain_ratio * (self.range[1] - self.range[0])
    
    def invert(self, value: float) -> float:
        """Transform a value from range back to domain.
        
        Args:
            value: Input value
            
        Returns:
            Original value
        """
        range_ratio = (value - self.range[0]) / (self.range[1] - self.range[0])
        return self.domain[0] + range_ratio * (self.domain[1] - self.domain[0])


class LogScale:
    """Logarithmic scale for numeric values."""
    
    def __init__(self, domain: Tuple[float, float], range: Tuple[float, float]):
        """Initialize the scale.
        
        Args:
            domain: Input domain (min, max)
            range: Output range (min, max)
        """
        self.domain = domain
        self.range = range
        self.step = (range[1] - range[0]) / (np.log(domain[1]) - np.log(domain[0]))
    
    def transform(self, value: float) -> float:
        """Transform a value from domain to range.
        
        Args:
            value: Input value
            
        Returns:
            Transformed value
        """
        log_ratio = (np.log(value) - np.log(self.domain[0])) / (np.log(self.domain[1]) - np.log(self.domain[0]))
        return self.range[0] + log_ratio * (self.range[1] - self.range[0])
    
    def invert(self, value: float) -> float:
        """Transform a value from range back to domain.
        
        Args:
            value: Input value
            
        Returns:
            Original value
        """
        range_ratio = (value - self.range[0]) / (self.range[1] - self.range[0])
        log_value = np.log(self.domain[0]) + range_ratio * (np.log(self.domain[1]) - np.log(self.domain[0]))
        return np.exp(log_value)


class TimeScale:
    """Time scale for datetime values."""
    
    def __init__(self, domain: Tuple[datetime, datetime], range: Tuple[float, float]):
        """Initialize the scale.
        
        Args:
            domain: Input domain (min, max)
            range: Output range (min, max)
        """
        self.domain = domain
        self.range = range
        self.step = (range[1] - range[0]) / (domain[1].timestamp() - domain[0].timestamp())
    
    def transform(self, value: Union[datetime, int, float, np.integer, np.floating]) -> float:
        """Transform a datetime or timestamp to pixel coordinate.
        
        Args:
            value: Input datetime or timestamp
            
        Returns:
            Pixel coordinate
        """
        if isinstance(value, (int, float, np.integer, np.floating)):
            timestamp = float(value)
        else:
            timestamp = value.timestamp()
            
        time_ratio = (timestamp - self.domain[0].timestamp()) / (self.domain[1].timestamp() - self.domain[0].timestamp())
        return self.range[0] + time_ratio * (self.range[1] - self.range[0])
    
    def invert(self, value: float) -> datetime:
        """Transform a pixel coordinate back to datetime.
        
        Args:
            value: Pixel coordinate
            
        Returns:
            Original datetime
        """
        range_ratio = (value - self.range[0]) / (self.range[1] - self.range[0])
        timestamp = self.domain[0].timestamp() + range_ratio * (self.domain[1].timestamp() - self.domain[0].timestamp())
        return datetime.fromtimestamp(timestamp) 