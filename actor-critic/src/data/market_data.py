"""Market data fetching and preprocessing utilities."""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Tuple, Dict
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

class MarketRegime:
    """Defines different market regimes with specific characteristics."""
    def __init__(
        self,
        name: str,
        trend: float,
        volatility: float,
        mean_reversion: float = 0.0,
        jumps_per_year: float = 0,
        jump_size_range: Tuple[float, float] = (-0.05, 0.05),
        seasonality_amplitude: float = 0.0
    ):
        self.name = name
        self.trend = trend
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.jumps_per_year = jumps_per_year
        self.jump_size_range = jump_size_range
        self.seasonality_amplitude = seasonality_amplitude

def create_synthetic_data(
    n_samples: int = 1000,
    n_assets: int = 1,
    seed: int = 42,
    regime: MarketRegime = None,
    include_regime_transitions: bool = True,
    trading_days_per_year: int = 252
) -> pd.DataFrame:
    """
    Generate synthetic market data with realistic properties.
    
    Args:
        n_samples: Number of time steps
        n_assets: Number of assets to simulate
        seed: Random seed
        regime: MarketRegime object defining market characteristics
        include_regime_transitions: Whether to include transitions between regimes
        trading_days_per_year: Number of trading days per year
    """
    np.random.seed(seed)
    
    # Default regime if none provided
    if regime is None:
        regime = MarketRegime(
            name="Normal",
            trend=0.05,  # 5% annual drift
            volatility=0.2,  # 20% annual volatility
            mean_reversion=0.1,  # Slight mean reversion
            jumps_per_year=5,  # 5 jumps per year
            jump_size_range=(-0.03, 0.03),  # ±3% jumps
            seasonality_amplitude=0.1  # 10% seasonal effect
        )
    
    # Time scaling
    dt = 1.0 / trading_days_per_year
    
    # Initialize price arrays
    prices = np.zeros((n_samples, n_assets))
    prices[0] = 100.0  # Starting price
    
    # Generate price paths
    for t in range(1, n_samples):
        # Daily trend component
        daily_trend = regime.trend * dt
        
        # Volatility component
        daily_vol = regime.volatility / np.sqrt(trading_days_per_year)
        random_shock = np.random.normal(0, daily_vol, n_assets)
        
        # Mean reversion component
        mean_reversion = regime.mean_reversion * (100 - prices[t-1]) * dt
        
        # Jump component
        jump = np.zeros(n_assets)
        if np.random.random() < regime.jumps_per_year / trading_days_per_year:
            jump = np.random.uniform(
                regime.jump_size_range[0],
                regime.jump_size_range[1],
                n_assets
            )
        
        # Seasonality component
        seasonality = regime.seasonality_amplitude * np.sin(2 * np.pi * t / trading_days_per_year)
        
        # Combine all components
        daily_return = (
            daily_trend +
            random_shock +
            mean_reversion +
            jump +
            seasonality
        )
        
        # Update prices
        prices[t] = prices[t-1] * (1 + daily_return)
    
    # Create DataFrame with OHLCV data
    data = {}
    for i in range(n_assets):
        asset_name = f"ASSET_{i+1}"
        # Generate realistic OHLC from daily close prices
        daily_vol = regime.volatility / np.sqrt(trading_days_per_year)
        
        data[f'open_{asset_name}'] = prices[:, i] * (1 + np.random.normal(0, daily_vol * 0.2, n_samples))
        data[f'high_{asset_name}'] = np.maximum(
            prices[:, i] * (1 + np.random.normal(0.001, daily_vol * 0.3, n_samples)),
            np.maximum(data[f'open_{asset_name}'], prices[:, i])
        )
        data[f'low_{asset_name}'] = np.minimum(
            prices[:, i] * (1 + np.random.normal(-0.001, daily_vol * 0.3, n_samples)),
            np.minimum(data[f'open_{asset_name}'], prices[:, i])
        )
        data[f'close_{asset_name}'] = prices[:, i]
        
        # Generate volume with price-volume correlation
        base_volume = np.random.lognormal(10, 1, n_samples)
        price_changes = np.diff(prices[:, i], prepend=prices[0, i])
        volume_adjustment = np.abs(price_changes) * 5  # Volume increases with price changes
        data[f'volume_{asset_name}'] = base_volume * (1 + volume_adjustment)
    
    return pd.DataFrame(data)

def create_market_regimes() -> Dict[str, MarketRegime]:
    """Create a dictionary of different market regimes for testing."""
    return {
        'bull_market': MarketRegime(
            name="Bull Market",
            trend=0.20,  # 20% annual upward trend
            volatility=0.15,  # Lower volatility
            mean_reversion=0.05,
            jumps_per_year=3,
            jump_size_range=(-0.02, 0.04)  # Positive bias in jumps
        ),
        'bear_market': MarketRegime(
            name="Bear Market",
            trend=-0.20,  # 20% annual downward trend
            volatility=0.25,  # Higher volatility
            mean_reversion=0.05,
            jumps_per_year=8,  # More frequent jumps
            jump_size_range=(-0.05, 0.03)  # Negative bias in jumps
        ),
        'high_volatility': MarketRegime(
            name="High Volatility",
            trend=0.0,
            volatility=0.40,  # Very high volatility
            mean_reversion=0.15,
            jumps_per_year=12,  # Frequent jumps
            jump_size_range=(-0.06, 0.06)
        ),
        'low_volatility': MarketRegime(
            name="Low Volatility",
            trend=0.05,
            volatility=0.10,  # Very low volatility
            mean_reversion=0.02,
            jumps_per_year=1
        ),
        'choppy_market': MarketRegime(
            name="Choppy Market",
            trend=0.0,
            volatility=0.20,
            mean_reversion=0.20,  # Strong mean reversion
            seasonality_amplitude=0.15  # Strong seasonality
        ),
        'crisis_market': MarketRegime(
            name="Crisis Market",
            trend=-0.40,  # Strong downward trend
            volatility=0.50,  # Extreme volatility
            jumps_per_year=20,  # Very frequent jumps
            jump_size_range=(-0.10, 0.08)  # Large, negatively biased jumps
        )
    }

def generate_regime_transitions(
    n_samples: int,
    regimes: Dict[str, MarketRegime],
    seed: int = 42,
    min_regime_length: int = 63  # Minimum 3 months per regime
) -> List[MarketRegime]:
    """Generate a sequence of regime transitions."""
    np.random.seed(seed)
    regime_sequence = []
    current_length = 0
    regime_names = list(regimes.keys())
    
    while current_length < n_samples:
        # Select random regime
        regime_name = np.random.choice(regime_names)
        # Random length between min_regime_length and 2*min_regime_length
        length = np.random.randint(min_regime_length, 2 * min_regime_length)
        length = min(length, n_samples - current_length)
        
        regime_sequence.extend([regimes[regime_name]] * length)
        current_length += length
    
    return regime_sequence[:n_samples]

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