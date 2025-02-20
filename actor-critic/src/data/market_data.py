"""Market data fetching and preprocessing utilities."""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

def fetch_market_data(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval ('1d', '1h', etc.)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    
    dfs = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            df.index = pd.to_datetime(df.index)
            df.columns = [f"{col.lower()}_{symbol}" for col in df.columns]
            dfs.append(df)
            logger.info(f"Successfully fetched data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No data could be fetched for any of the provided symbols")
    
    # Combine all dataframes
    market_data = pd.concat(dfs, axis=1)
    market_data = market_data.ffill().bfill()
    
    return market_data

def prepare_training_data(
    market_data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_samples: int = 252  # One trading year
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare training, validation, and test datasets.
    
    Args:
        market_data: Raw market data DataFrame
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        min_samples: Minimum number of samples required
        
    Returns:
        train_data, val_data, test_data: Split datasets
    """
    if len(market_data) < min_samples:
        raise ValueError(f"Insufficient data points. Need at least {min_samples}")
    
    # Calculate split indices
    train_idx = int(len(market_data) * train_ratio)
    val_idx = int(len(market_data) * (train_ratio + val_ratio))
    
    # Split data
    train_data = market_data.iloc[:train_idx]
    val_data = market_data.iloc[train_idx:val_idx]
    test_data = market_data.iloc[val_idx:]
    
    # Verify splits have sufficient data
    for name, data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
        if len(data) < min_samples / 4:  # At least quarter of min_samples
            logger.warning(f"{name} set has fewer than {min_samples/4} samples")
    
    return train_data, val_data, test_data

def create_synthetic_data(
    n_samples: int = 1000,
    n_assets: int = 1,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Create synthetic market data for testing or development.
    
    Args:
        n_samples: Number of time steps
        n_assets: Number of assets
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = {}
    for i in range(n_assets):
        symbol = f"ASSET_{i+1}"
        
        # Generate random walk for close prices
        close = 100 + np.random.randn(n_samples).cumsum()
        
        # Generate other OHLCV data
        high = close + np.abs(np.random.randn(n_samples)) * 2
        low = close - np.abs(np.random.randn(n_samples)) * 2
        open_ = low + np.random.rand(n_samples) * (high - low)
        volume = np.random.randint(1000, 10000, n_samples)
        
        data[f"open_{symbol}"] = open_
        data[f"high_{symbol}"] = high
        data[f"low_{symbol}"] = low
        data[f"close_{symbol}"] = close
        data[f"volume_{symbol}"] = volume
    
    # Create DataFrame with datetime index
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    return pd.DataFrame(data, index=dates)

def validate_market_data(data: pd.DataFrame) -> bool:
    """
    Validate market data quality.
    
    Args:
        data: Market data DataFrame
        
    Returns:
        bool: Whether data passes quality checks
    """
    # Check for required columns
    required_patterns = ['open', 'high', 'low', 'close', 'volume']
    if not all(any(pattern in col.lower() for col in data.columns) for pattern in required_patterns):
        logger.error("Missing required OHLCV columns")
        return False
    
    # Check for sufficient data points
    if len(data) < 252:  # One trading year
        logger.error("Insufficient data points")
        return False
    
    # Check for missing values
    if data.isnull().any().any():
        logger.warning("Data contains missing values")
        return False
    
    # Check for price consistency
    for symbol in set('_'.join(col.split('_')[1:]) for col in data.columns if 'close' in col.lower()):
        high_col = f"high_{symbol}"
        low_col = f"low_{symbol}"
        open_col = f"open_{symbol}"
        close_col = f"close_{symbol}"
        
        if not (
            (data[high_col] >= data[low_col]).all() and
            (data[high_col] >= data[open_col]).all() and
            (data[high_col] >= data[close_col]).all() and
            (data[low_col] <= data[open_col]).all() and
            (data[low_col] <= data[close_col]).all()
        ):
            logger.error(f"Price consistency check failed for {symbol}")
            return False
    
    # Check for negative volumes
    volume_cols = [col for col in data.columns if 'volume' in col.lower()]
    if (data[volume_cols] < 0).any().any():
        logger.error("Negative volumes detected")
        return False
    
    return True 