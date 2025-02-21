"""
YFinance market data provider implementation.

This module implements a market data provider using the yfinance library to fetch data from Yahoo Finance.
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    """Market data provider using yfinance."""
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '2m': '2m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '60m': '1h',
        '90m': '90m',
        '1h': '1h',
        '1d': '1d',
        '5d': '5d',
        '1wk': '1wk',
        '1mo': '1mo',
        '3mo': '3mo'
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the YFinance provider.
        
        Args:
            cache_dir: Optional directory for caching data
        """
        super().__init__()
        self._cache_dir = cache_dir
        self._tickers = {}  # Cache for ticker objects
    
    @property
    def supported_timeframes(self) -> List[str]:
        """List of timeframes supported by YFinance.
        
        Returns:
            List[str]: List of supported timeframe strings
        """
        return list(self.TIMEFRAME_MAP.keys())
    
    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a ticker object for a symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            yf.Ticker: Ticker object for the symbol
        """
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol)
        return self._tickers[symbol]
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1d',
        include_extended_hours: bool = False
    ) -> pd.DataFrame:
        """Fetch historical market data from Yahoo Finance.
        
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
        # Validate timeframe
        self.validate_timeframe(timeframe)
        
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        try:
            # Get ticker object
            ticker = self._get_ticker(symbol)
            
            # Fetch data from yfinance
            df = ticker.history(
                interval=self.TIMEFRAME_MAP[timeframe],
                start=start_date,
                end=end_date,
                prepost=include_extended_hours,
                actions=False
            )
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Rename columns to match our standard format
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Validate and standardize the data
            self.validate_data(df)
            df = self.standardize_data(df)
            
            return df
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data from Yahoo Finance: {str(e)}")
    
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
        # For latest data, fetch last 2 periods to ensure we have the most recent
        end_date = datetime.now()
        
        # Calculate start date based on timeframe
        if timeframe in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
            start_date = end_date - timedelta(days=7)  # YFinance limitation for intraday
        else:
            start_date = end_date - timedelta(days=5)  # Sufficient for daily and above
            
        df = self.fetch_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        # Return only the last row
        return df.tail(1) 