"""
Base class for market data providers.

This module defines the abstract base class that all market data providers must implement.
It establishes a common interface for fetching and processing market data from various sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    def __init__(self):
        """Initialize the market data provider."""
        self._supported_timeframes = []
        self._rate_limit = None
        self._last_request_time = None
    
    @property
    @abstractmethod
    def supported_timeframes(self) -> List[str]:
        """List of timeframes supported by this provider.
        
        Returns:
            List[str]: List of supported timeframe strings (e.g., ['1m', '5m', '1h', '1d'])
        """
        pass
    
    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1d',
        include_extended_hours: bool = False
    ) -> pd.DataFrame:
        """Fetch historical market data for a given symbol and time range.
        
        Args:
            symbol: The ticker symbol to fetch data for
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            include_extended_hours: Whether to include pre/post market data
            
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Raises:
            ValueError: If timeframe is not supported
            ConnectionError: If data cannot be fetched
        """
        pass
    
    @abstractmethod
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """Get the most recent market data for a symbol.
        
        Args:
            symbol: The ticker symbol to fetch data for
            timeframe: Data timeframe (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            pd.DataFrame: DataFrame with latest market data
            
        Raises:
            ValueError: If timeframe is not supported
            ConnectionError: If data cannot be fetched
        """
        pass
    
    def validate_timeframe(self, timeframe: str) -> None:
        """Validate that a timeframe is supported by this provider.
        
        Args:
            timeframe: The timeframe to validate
            
        Raises:
            ValueError: If timeframe is not supported
        """
        if timeframe not in self.supported_timeframes:
            raise ValueError(
                f"Timeframe '{timeframe}' not supported. "
                f"Supported timeframes: {self.supported_timeframes}"
            )
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate market data format and content.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime if needed
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Check timestamp order
        timestamps = df['timestamp'].values
        if not np.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError("DataFrame timestamps must be monotonically increasing")
        
        # Check for invalid values
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null values")
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
    
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize market data format.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Ensure numeric columns are float64
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(np.float64)
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        return df 