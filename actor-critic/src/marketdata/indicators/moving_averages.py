"""Moving average indicators."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .base import (
    BaseIndicator,
    IndicatorParams,
    IndicatorResult,
    register_indicator
)


@dataclass
class MovingAverageParams(IndicatorParams):
    """Moving average parameters."""
    period: int = 20
    source: str = 'close'


@register_indicator('sma')
class SimpleMovingAverage(BaseIndicator):
    """Simple Moving Average (SMA) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MovingAverageParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate SMA values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: SMA values
        """
        params = self.params
        values = data[params.source].rolling(
            window=params.period,
            min_periods=1
        ).mean()
        
        return IndicatorResult(
            values=values,
            metadata={
                'period': params.period,
                'source': params.source
            }
        )


@register_indicator('ema')
class ExponentialMovingAverage(BaseIndicator):
    """Exponential Moving Average (EMA) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MovingAverageParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate EMA values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: EMA values
        """
        params = self.params
        values = data[params.source].ewm(
            span=params.period,
            min_periods=1,
            adjust=False
        ).mean()
        
        return IndicatorResult(
            values=values,
            metadata={
                'period': params.period,
                'source': params.source
            }
        )


@register_indicator('wma')
class WeightedMovingAverage(BaseIndicator):
    """Weighted Moving Average (WMA) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MovingAverageParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate WMA values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: WMA values
        """
        params = self.params
        source = data[params.source]
        
        # Calculate weights
        weights = np.arange(1, params.period + 1)
        weights = weights / weights.sum()
        
        # Calculate WMA
        values = source.rolling(
            window=params.period,
            min_periods=1
        ).apply(
            lambda x: np.sum(weights[-len(x):] * x) / np.sum(weights[-len(x):])
            if len(x) > 0 else np.nan
        )
        
        return IndicatorResult(
            values=values,
            metadata={
                'period': params.period,
                'source': params.source
            }
        )


@register_indicator('hull')
class HullMovingAverage(BaseIndicator):
    """Hull Moving Average (HMA) indicator.
    
    The Hull Moving Average is designed to reduce lag while maintaining smoothness.
    It uses weighted moving averages to achieve this.
    """
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MovingAverageParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate HMA values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: HMA values
        """
        params = self.params
        source = data[params.source]
        
        # Calculate WMAs
        period_half = params.period // 2
        
        # Create WMA instances
        wma1 = WeightedMovingAverage(
            MovingAverageParams(period=period_half)
        )
        wma2 = WeightedMovingAverage(
            MovingAverageParams(period=params.period)
        )
        wma3 = WeightedMovingAverage(
            MovingAverageParams(period=int(np.sqrt(params.period)))
        )
        
        # Calculate components
        half_wma = wma1.calculate(data).values
        full_wma = wma2.calculate(data).values
        
        # Calculate raw Hull MA
        raw_hma = 2 * half_wma - full_wma
        
        # Create DataFrame for final WMA
        hull_data = pd.DataFrame({
            'timestamp': data.index,
            params.source: raw_hma
        }).set_index('timestamp')
        
        # Calculate final HMA
        values = wma3.calculate(hull_data).values
        
        return IndicatorResult(
            values=values,
            metadata={
                'period': params.period,
                'source': params.source
            }
        )


@register_indicator('vwma')
class VolumeWeightedMovingAverage(BaseIndicator):
    """Volume Weighted Moving Average (VWMA) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MovingAverageParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate VWMA values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: VWMA values
        """
        params = self.params
        source = data[params.source]
        volume = data['volume']
        
        # Calculate volume-weighted sum
        weighted_sum = (source * volume).rolling(
            window=params.period,
            min_periods=1
        ).sum()
        
        # Calculate volume sum
        volume_sum = volume.rolling(
            window=params.period,
            min_periods=1
        ).sum()
        
        # Calculate VWMA
        values = weighted_sum / volume_sum
        
        return IndicatorResult(
            values=values,
            metadata={
                'period': params.period,
                'source': params.source
            }
        )


class SimpleMovingAverage:
    """Simple moving average indicator."""

    def __init__(self, period: int = 20, column: str = 'close'):
        """Initialize the indicator.
        
        Args:
            period: The number of periods to average over.
            column: The column to calculate the average for.
        """
        self.period = period
        self.column = column
        self._values = None

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the simple moving average.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            Series containing the moving average values.
            
        Raises:
            ValueError: If the required column is missing.
        """
        if self.column not in data.columns:
            raise ValueError(f"Required column '{self.column}' not found in data")
        
        self._values = data[self.column].rolling(window=self.period).mean()
        return self._values

    @property
    def values(self) -> pd.Series:
        """Get the calculated values."""
        if self._values is None:
            raise ValueError("Indicator values not calculated yet")
        return self._values 