"""Data processing utilities for market data."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Base class for market data processing."""
    
    def __init__(self):
        """Initialize the data processor."""
        self._supported_timeframes = [
            '1m', '5m', '15m', '30m',  # Minutes
            '1h', '2h', '4h', '6h', '12h',  # Hours
            '1d', '1w', '1mo'  # Days and up
        ]
        self._ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    
    def validate_data(
        self,
        df: pd.DataFrame,
        allow_zero_volume: bool = False
    ) -> None:
        """Validate market data format and content.
        
        Args:
            df: DataFrame to validate
            allow_zero_volume: Whether to allow zero volume values
            
        Raises:
            ValueError: If data format is invalid
        """
        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be monotonically increasing")
        
        # Check required columns
        missing_columns = [col for col in self._ohlcv_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for invalid values
        for col in self._ohlcv_columns:
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null values")
            
            # Special handling for volume
            if col == 'volume':
                if allow_zero_volume:
                    if (df[col] < 0).any():
                        raise ValueError(f"Column '{col}' contains negative values")
                else:
                    if (df[col] <= 0).any():
                        raise ValueError(f"Column '{col}' contains non-positive values")
            else:
                if (df[col] <= 0).any():
                    raise ValueError(f"Column '{col}' contains non-positive values")
        
        # Check OHLC relationships
        if not (
            (df['high'] >= df['low']).all() and
            (df['high'] >= df['open']).all() and
            (df['high'] >= df['close']).all() and
            (df['low'] <= df['open']).all() and
            (df['low'] <= df['close']).all()
        ):
            raise ValueError("Invalid OHLC relationships detected")
    
    def _timeframe_to_offset(self, timeframe: str) -> pd.Timedelta:
        """Convert timeframe string to pandas offset.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            pd.Timedelta: Pandas offset object
            
        Raises:
            ValueError: If timeframe format is invalid
        """
        if timeframe not in self._supported_timeframes:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported timeframes: {self._supported_timeframes}"
            )
        
        # Handle month timeframe separately
        if timeframe.endswith('mo'):
            amount = int(timeframe[:-2])
            return pd.Timedelta(days=amount * 30)  # Approximate
        
        # Parse number and unit
        amount = int(timeframe[:-1])
        unit = timeframe[-1]
        
        # Convert to pandas offset
        if unit == 'm':
            return pd.Timedelta(minutes=amount)
        elif unit == 'h':
            return pd.Timedelta(hours=amount)
        elif unit == 'd':
            return pd.Timedelta(days=amount)
        elif unit == 'w':
            return pd.Timedelta(weeks=amount)
        else:
            raise ValueError(f"Invalid timeframe unit: {unit}")
    
    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        volume_weighted: bool = True
    ) -> pd.DataFrame:
        """Resample OHLCV data to a new timeframe.
        
        Args:
            df: DataFrame to resample
            target_timeframe: Target timeframe (e.g., '1h', '1d')
            volume_weighted: Whether to use volume-weighted calculations
            
        Returns:
            pd.DataFrame: Resampled DataFrame
            
        Raises:
            ValueError: If input data or parameters are invalid
        """
        # Handle empty DataFrame
        if len(df) == 0:
            return df.copy()
        
        # Validate input data
        self.validate_data(df)
        
        # Get resampling offset
        offset = self._timeframe_to_offset(target_timeframe)
        
        # Define aggregation functions
        agg_funcs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        if volume_weighted:
            # Calculate volume-weighted prices
            df = df.copy()
            df['vw_price'] = df['close'] * df['volume']
            agg_funcs['vw_price'] = 'sum'
        
        # Resample data
        resampled = df.resample(offset).agg(agg_funcs)
        
        # Handle missing values for OHLC
        for col in ['open', 'high', 'low', 'close']:
            resampled[col] = resampled[col].ffill()
        
        # Handle missing volume values
        resampled['volume'] = resampled['volume'].fillna(0)
        
        if volume_weighted:
            # Calculate VWAP only for periods with volume
            resampled['vwap'] = np.where(
                resampled['volume'] > 0,
                resampled['vw_price'] / resampled['volume'],
                resampled['close']  # Use close price when no volume
            )
            resampled = resampled.drop('vw_price', axis=1)
        
        # Validate output with zero volume allowed
        self.validate_data(resampled, allow_zero_volume=True)
        
        return resampled 