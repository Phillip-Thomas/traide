"""
Utilities for custom OHLCV data aggregation.

This module provides functions for aggregating OHLCV (Open, High, Low, Close, Volume) data
using various methods and timeframes.
"""

from typing import Callable, Dict, Optional, Union
import numpy as np
import pandas as pd


def ohlcv_agg(df: pd.DataFrame) -> Dict[str, Callable]:
    """Get standard OHLCV aggregation functions.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dict[str, Callable]: Dictionary of column names to aggregation functions
    """
    return {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }


def vwap_agg(df: pd.DataFrame) -> float:
    """Calculate Volume Weighted Average Price.
    
    Args:
        df: DataFrame with price and volume data
        
    Returns:
        float: VWAP value
    """
    return (df['close'] * df['volume']).sum() / df['volume'].sum()


def time_weighted_agg(df: pd.DataFrame, price_col: str = 'close') -> float:
    """Calculate time-weighted average price.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        
    Returns:
        float: Time-weighted average price
    """
    time_diff = df.index.to_series().diff().dt.total_seconds()
    time_diff.iloc[0] = 0  # Handle first row
    return (df[price_col] * time_diff).sum() / time_diff.sum()


def tick_resample(
    df: pd.DataFrame,
    ticks: int,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None
) -> pd.DataFrame:
    """Resample data based on number of ticks.
    
    Args:
        df: DataFrame with OHLCV data
        ticks: Number of ticks per bar
        agg_funcs: Optional custom aggregation functions
        
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Default to standard OHLCV aggregation if none provided
    if agg_funcs is None:
        agg_funcs = ohlcv_agg(df)
    
    # Create groups based on tick count
    groups = np.arange(len(df)) // ticks
    
    # Apply aggregation
    return df.groupby(groups).agg(agg_funcs)


def volume_resample(
    df: pd.DataFrame,
    volume_threshold: float,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None
) -> pd.DataFrame:
    """Resample data based on volume threshold.
    
    Args:
        df: DataFrame with OHLCV data
        volume_threshold: Volume threshold for each bar
        agg_funcs: Optional custom aggregation functions
        
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Default to standard OHLCV aggregation if none provided
    if agg_funcs is None:
        agg_funcs = ohlcv_agg(df)
    
    # Calculate cumulative volume
    cum_vol = df['volume'].cumsum()
    
    # Create groups based on volume threshold
    groups = (cum_vol // volume_threshold).astype(int)
    
    # Apply aggregation
    resampled = df.groupby(groups).agg(agg_funcs)
    
    # Filter out any bars that don't meet the threshold
    resampled = resampled[resampled['volume'] >= volume_threshold]
    
    return resampled


def dollar_resample(
    df: pd.DataFrame,
    dollar_threshold: float,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None
) -> pd.DataFrame:
    """Resample data based on dollar volume threshold.
    
    Args:
        df: DataFrame with OHLCV data
        dollar_threshold: Dollar volume threshold for each bar
        agg_funcs: Optional custom aggregation functions
        
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Default to standard OHLCV aggregation if none provided
    if agg_funcs is None:
        agg_funcs = ohlcv_agg(df)
    
    # Calculate dollar volume
    dollar_vol = df['close'] * df['volume']
    cum_dollar = dollar_vol.cumsum()
    
    # Create groups based on dollar volume threshold
    groups = (cum_dollar // dollar_threshold).astype(int)
    
    # Apply aggregation
    resampled = df.groupby(groups).agg(agg_funcs)
    
    # Filter out any bars that don't meet the threshold
    dollar_volume = resampled['close'] * resampled['volume']
    resampled = resampled[dollar_volume >= dollar_threshold]
    
    return resampled


def custom_resample(
    df: pd.DataFrame,
    rule: str,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
    custom_aggs: Optional[Dict[str, Callable]] = None
) -> pd.DataFrame:
    """Resample data with custom aggregation functions.
    
    Args:
        df: DataFrame with OHLCV data
        rule: Pandas resample rule (e.g., '1min', '1H', '1D')
        agg_funcs: Optional custom aggregation functions for standard columns
        custom_aggs: Optional additional custom aggregations to compute
        
    Returns:
        pd.DataFrame: Resampled DataFrame with custom aggregations
    """
    # Default to standard OHLCV aggregation if none provided
    if agg_funcs is None:
        agg_funcs = ohlcv_agg(df)
    
    # Resample with standard aggregations
    resampled = df.resample(rule).agg(agg_funcs)
    
    # Add custom aggregations if provided
    if custom_aggs:
        for col, func in custom_aggs.items():
            resampled[col] = df.resample(rule).apply(func)
    
    return resampled 