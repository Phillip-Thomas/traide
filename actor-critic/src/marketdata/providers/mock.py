"""
Mock market data provider for testing purposes.

This module provides a mock implementation of the MarketDataProvider interface
that generates synthetic market data for testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union

from .base import MarketDataProvider


class MockMarketDataProvider(MarketDataProvider):
    """Mock market data provider that generates synthetic data."""
    
    def __init__(self, seed: int = 42):
        """Initialize the mock provider.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        super().__init__()
        self._supported_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self._base_seed = seed
        np.random.seed(seed)  # Set global seed for reproducibility
        self.rng = np.random.RandomState(seed)
        
    @property
    def supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return self._supported_timeframes
    
    def _generate_ohlcv(self, n_points: int, base_price: float = 100.0) -> pd.DataFrame:
        """Generate synthetic OHLCV data.
        
        Args:
            n_points: Number of data points to generate
            base_price: Starting price for the synthetic data
            
        Returns:
            pd.DataFrame: Synthetic OHLCV data
        """
        # Generate log returns from normal distribution
        returns = self.rng.normal(0.0001, 0.02, n_points)
        
        # Generate prices
        closes = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        highs = closes * np.exp(self.rng.uniform(0, 0.02, n_points))
        lows = closes * np.exp(-self.rng.uniform(0, 0.02, n_points))
        opens = lows + (highs - lows) * self.rng.uniform(0, 1, n_points)
        
        # Generate volume
        volume = self.rng.lognormal(10, 1, n_points).astype(int)
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume
        })
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to number of minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            int: Number of minutes
        """
        multiplier = int(timeframe[:-1])
        unit = timeframe[-1]
        
        if unit == 'm':
            return multiplier
        elif unit == 'h':
            return multiplier * 60
        elif unit == 'd':
            return multiplier * 60 * 24
        else:
            raise ValueError(f"Invalid timeframe unit: {unit}")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1d',
        include_extended_hours: bool = False
    ) -> pd.DataFrame:
        """Fetch synthetic historical market data.
        
        Args:
            symbol: The ticker symbol (used as random seed)
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe
            include_extended_hours: Not used in mock provider
            
        Returns:
            pd.DataFrame: Synthetic market data
        """
        self.validate_timeframe(timeframe)
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Calculate number of periods
        minutes = self._timeframe_to_minutes(timeframe)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{minutes}min',  # Use 'min' instead of 'T'
            inclusive='both'
        )
        
        # Reset random state with combined seed for reproducibility
        combined_seed = hash(f"{self._base_seed}_{symbol}_{timeframe}") % (2**32)
        self.rng = np.random.RandomState(combined_seed)
        
        # Generate OHLCV data
        df = self._generate_ohlcv(len(timestamps), base_price=100.0)
        df['timestamp'] = timestamps
        
        return self.standardize_data(df)
    
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """Get latest synthetic market data.
        
        Args:
            symbol: The ticker symbol (used as random seed)
            timeframe: Data timeframe
            
        Returns:
            pd.DataFrame: Latest synthetic market data
        """
        self.validate_timeframe(timeframe)
        
        # Generate one data point
        end_date = datetime.now().replace(microsecond=0)
        minutes = self._timeframe_to_minutes(timeframe)
        start_date = end_date - timedelta(minutes=minutes)
        
        df = self.fetch_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        # Return only the last data point
        return df.iloc[-1:].copy() 