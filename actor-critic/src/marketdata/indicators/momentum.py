"""Momentum-based technical indicators."""

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
class RSIParams(IndicatorParams):
    """RSI indicator parameters."""
    period: int = 14
    source: str = 'close'
    min_periods: int = None


@register_indicator('rsi')
class RelativeStrengthIndex(BaseIndicator):
    """Relative Strength Index (RSI) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return RSIParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate RSI values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: RSI values
        """
        params = self.params
        min_periods = params.min_periods if params.min_periods is not None else params.period
        
        # Calculate price changes
        delta = data[params.source].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses
        avg_gains = gains.ewm(
            span=params.period,
            min_periods=min_periods,
            adjust=False
        ).mean()
        
        avg_losses = losses.ewm(
            span=params.period,
            min_periods=min_periods,
            adjust=False
        ).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.clip(lower=1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure bounds
        rsi = rsi.clip(0, 100)
        
        # Set NaN values for initial period
        rsi.iloc[:min_periods-1] = np.nan
        
        return IndicatorResult(
            values=rsi,
            metadata={
                'period': params.period,
                'source': params.source,
                'min_periods': min_periods
            }
        )


@dataclass
class MACDParams(IndicatorParams):
    """MACD indicator parameters."""
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    source: str = 'close'
    min_periods: int = None


@register_indicator('macd')
class MovingAverageConvergenceDivergence(BaseIndicator):
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return MACDParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate MACD values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: MACD values
        """
        params = self.params
        min_periods = params.min_periods if params.min_periods is not None else params.slow_period
        
        # Calculate EMAs
        fast_ema = data[params.source].ewm(
            span=params.fast_period,
            min_periods=min_periods,
            adjust=False
        ).mean()
        
        slow_ema = data[params.source].ewm(
            span=params.slow_period,
            min_periods=min_periods,
            adjust=False
        ).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(
            span=params.signal_period,
            min_periods=params.signal_period,
            adjust=False
        ).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Set NaN values for initial period
        start_idx = max(min_periods - 1, params.signal_period - 1)
        macd_line.iloc[:start_idx] = np.nan
        signal_line.iloc[:start_idx] = np.nan
        histogram.iloc[:start_idx] = np.nan
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'fast_period': params.fast_period,
                'slow_period': params.slow_period,
                'signal_period': params.signal_period,
                'source': params.source,
                'min_periods': min_periods
            }
        )


class MovingAverageConvergenceDivergence:
    """Moving Average Convergence Divergence (MACD) indicator."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, column: str = 'close'):
        """Initialize the indicator.
        
        Args:
            fast_period: The number of periods for the fast EMA.
            slow_period: The number of periods for the slow EMA.
            signal_period: The number of periods for the signal line.
            column: The column to calculate the MACD for.
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column
        self._macd = None
        self._signal = None
        self._histogram = None

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the MACD.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            DataFrame containing the MACD line, signal line, and histogram.
            
        Raises:
            ValueError: If the required column is missing.
        """
        if self.column not in data.columns:
            raise ValueError(f"Required column '{self.column}' not found in data")
        
        fast_ema = data[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.column].ewm(span=self.slow_period, adjust=False).mean()
        
        self._macd = fast_ema - slow_ema
        self._signal = self._macd.ewm(span=self.signal_period, adjust=False).mean()
        self._histogram = self._macd - self._signal
        
        return pd.DataFrame({
            'macd': self._macd,
            'signal': self._signal,
            'histogram': self._histogram
        })

    @property
    def macd(self) -> pd.Series:
        """Get the MACD line values."""
        if self._macd is None:
            raise ValueError("Indicator values not calculated yet")
        return self._macd

    @property
    def signal(self) -> pd.Series:
        """Get the signal line values."""
        if self._signal is None:
            raise ValueError("Indicator values not calculated yet")
        return self._signal

    @property
    def histogram(self) -> pd.Series:
        """Get the histogram values."""
        if self._histogram is None:
            raise ValueError("Indicator values not calculated yet")
        return self._histogram


class RelativeStrengthIndex:
    """Relative Strength Index (RSI) indicator."""

    def __init__(self, period: int = 14, column: str = 'close'):
        """Initialize the indicator.
        
        Args:
            period: The number of periods for the RSI calculation.
            column: The column to calculate the RSI for.
        """
        self.period = period
        self.column = column
        self._values = None

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the RSI.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            Series containing the RSI values.
            
        Raises:
            ValueError: If the required column is missing.
        """
        if self.column not in data.columns:
            raise ValueError(f"Required column '{self.column}' not found in data")
        
        delta = data[self.column].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        self._values = 100 - (100 / (1 + rs))
        
        return self._values

    @property
    def values(self) -> pd.Series:
        """Get the RSI values."""
        if self._values is None:
            raise ValueError("Indicator values not calculated yet")
        return self._values 