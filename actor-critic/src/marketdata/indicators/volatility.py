"""Volatility-based technical indicators."""

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
class BollingerBandsParams(IndicatorParams):
    """Bollinger Bands indicator parameters."""
    period: int = 20
    num_std: float = 2.0
    source: str = 'close'
    min_periods: int = None


@register_indicator('bbands')
class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return BollingerBandsParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate Bollinger Bands.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: Bollinger Bands values
        """
        params = self.params
        min_periods = params.min_periods if params.min_periods is not None else params.period
        
        # Calculate middle band (SMA)
        middle = data[params.source].rolling(
            window=params.period,
            min_periods=min_periods
        ).mean()
        
        # Calculate standard deviation
        std = data[params.source].rolling(
            window=params.period,
            min_periods=min_periods
        ).std()
        
        # Calculate bands
        band_width = params.num_std * std
        upper = middle + band_width
        lower = middle - band_width
        
        # Ensure band relationships are maintained
        upper = np.maximum(upper, middle)
        lower = np.minimum(lower, middle)
        
        # Calculate bandwidth
        width = (upper - lower) / middle
        width = width.replace([np.inf, -np.inf], np.nan)
        width = width.clip(lower=0)
        
        # Calculate %B (scaled to 0-1)
        percent_b = (data[params.source] - lower) / (upper - lower)
        percent_b = percent_b.replace([np.inf, -np.inf], np.nan)
        percent_b = percent_b.clip(0, 1)
        
        # Set NaN values for initial periods
        upper.iloc[:min_periods-1] = np.nan
        middle.iloc[:min_periods-1] = np.nan
        lower.iloc[:min_periods-1] = np.nan
        width.iloc[:min_periods-1] = np.nan
        percent_b.iloc[:min_periods-1] = np.nan
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'percent_b': percent_b
        })
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'period': params.period,
                'num_std': params.num_std,
                'source': params.source,
                'min_periods': min_periods
            }
        )


@dataclass
class ATRParams(IndicatorParams):
    """ATR indicator parameters."""
    period: int = 14
    smoothing: str = 'rma'  # 'sma', 'ema', 'rma'
    min_periods: int = None


@register_indicator('atr')
class AverageTrueRange(BaseIndicator):
    """Average True Range (ATR) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return ATRParams()
    
    def _true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            pd.Series: True Range values
        """
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # First value will be NaN due to shift, use high-low instead
        true_range.iloc[0] = high_low.iloc[0]
        
        # Handle negative values (shouldn't occur with valid data)
        true_range = true_range.clip(lower=0)
        
        return true_range
    
    def _smooth(self, series: pd.Series, min_periods: int) -> pd.Series:
        """Apply smoothing to a series.
        
        Args:
            series: Series to smooth
            min_periods: Minimum number of observations
            
        Returns:
            pd.Series: Smoothed series
        """
        params = self.params
        
        if params.smoothing == 'sma':
            result = series.rolling(
                window=params.period,
                min_periods=min_periods
            ).mean()
        elif params.smoothing == 'ema':
            result = series.ewm(
                span=params.period,
                min_periods=min_periods,
                adjust=False
            ).mean()
        else:  # 'rma'
            alpha = 1.0 / params.period
            result = series.ewm(
                alpha=alpha,
                min_periods=min_periods,
                adjust=False
            ).mean()
        
        # Ensure non-negative values
        result = result.clip(lower=0)
        
        return result
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate ATR values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: ATR values
        """
        params = self.params
        min_periods = params.min_periods if params.min_periods is not None else params.period
        
        # Calculate True Range
        tr = self._true_range(data)
        
        # Calculate ATR
        atr = self._smooth(tr, min_periods)
        
        # Calculate ATR%
        atr_percent = atr / data['close'].clip(lower=1e-10) * 100
        
        # Handle edge cases
        atr_percent = atr_percent.replace([np.inf, -np.inf], np.nan)
        
        # Set NaN values for initial periods
        if min_periods > 1:
            tr.iloc[:min_periods-1] = np.nan
            atr.iloc[:min_periods-1] = np.nan
            atr_percent.iloc[:min_periods-1] = np.nan
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'tr': tr,
            'atr': atr,
            'atr_percent': atr_percent
        })
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'period': params.period,
                'smoothing': params.smoothing,
                'min_periods': min_periods
            }
        )


class BollingerBands:
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_dev: float = 2.0, column: str = 'close'):
        """Initialize the indicator.
        
        Args:
            period: The number of periods for the moving average.
            std_dev: The number of standard deviations for the bands.
            column: The column to calculate the bands for.
        """
        self.period = period
        self.std_dev = std_dev
        self.column = column
        self._middle = None
        self._upper = None
        self._lower = None

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Bollinger Bands.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            DataFrame containing the middle, upper, and lower bands.
            
        Raises:
            ValueError: If the required column is missing.
        """
        if self.column not in data.columns:
            raise ValueError(f"Required column '{self.column}' not found in data")
        
        self._middle = data[self.column].rolling(window=self.period).mean()
        std = data[self.column].rolling(window=self.period).std()
        
        self._upper = self._middle + (std * self.std_dev)
        self._lower = self._middle - (std * self.std_dev)
        
        return pd.DataFrame({
            'middle': self._middle,
            'upper': self._upper,
            'lower': self._lower
        })

    @property
    def middle(self) -> pd.Series:
        """Get the middle band values."""
        if self._middle is None:
            raise ValueError("Indicator values not calculated yet")
        return self._middle

    @property
    def upper(self) -> pd.Series:
        """Get the upper band values."""
        if self._upper is None:
            raise ValueError("Indicator values not calculated yet")
        return self._upper

    @property
    def lower(self) -> pd.Series:
        """Get the lower band values."""
        if self._lower is None:
            raise ValueError("Indicator values not calculated yet")
        return self._lower 