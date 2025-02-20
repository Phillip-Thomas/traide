"""Feature engineering utilities for market data."""

import numpy as np
import pandas as pd
import talib
from typing import List, Dict, Union, Any
import torch

def calculate_vwap(data: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    return (typical_price * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()

def calculate_technical_features(
    data: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    window_sizes: List[int] = [14, 30, 50]
) -> pd.DataFrame:
    """
    Calculate technical indicators for feature engineering.
    
    Args:
        data: Price data DataFrame
        price_col: Name of price column
        volume_col: Name of volume column
        window_sizes: List of window sizes for indicators
        
    Returns:
        features: DataFrame of calculated features
    """
    features = pd.DataFrame(index=data.index)
    
    # Ensure required columns exist
    required_cols = ['high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            data[col] = data[price_col]
    
    # Extract OHLCV data
    high = data['high']
    low = data['low']
    close = data[price_col]
    volume = data[volume_col]
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    high = high.clip(lower=epsilon)
    low = low.clip(lower=epsilon)
    close = close.clip(lower=epsilon)
    volume = volume.clip(lower=epsilon)
    
    # Price-based features
    for window in window_sizes:
        # Moving averages
        features[f'sma_{window}'] = talib.SMA(close, timeperiod=window)
        features[f'ema_{window}'] = talib.EMA(close, timeperiod=window)
        
        # VWAP
        features[f'vwap_{window}'] = calculate_vwap(data, window)
        
        # Volatility
        features[f'bbands_upper_{window}'], features[f'bbands_middle_{window}'], features[f'bbands_lower_{window}'] = \
            talib.BBANDS(close, timeperiod=window)
        features[f'atr_{window}'] = talib.ATR(high, low, close, timeperiod=window)
        
        # Momentum
        features[f'rsi_{window}'] = talib.RSI(close, timeperiod=window)
        features[f'cci_{window}'] = talib.CCI(high, low, close, timeperiod=window)
        
        # Volume
        features[f'obv_{window}'] = talib.OBV(close, volume)
        features[f'adosc_{window}'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # Volume-based features
        features[f'volume_sma_{window}'] = talib.SMA(volume, timeperiod=window)
        features[f'volume_ema_{window}'] = talib.EMA(volume, timeperiod=window)
        features[f'volume_std_{window}'] = pd.Series(volume).rolling(window).std()
        
        # Volume ratios (using log to reduce skewness)
        features[f'volume_ratio_{window}'] = np.log1p(volume / features[f'volume_sma_{window}'].clip(lower=epsilon))
    
    # MACD
    features['macd'], features['macd_signal'], features['macd_hist'] = \
        talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Stochastic
    k, d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    features['stoch_k'] = k.clip(0, 100)
    features['stoch_d'] = d.clip(0, 100)
    
    # Trend
    features['adx'] = talib.ADX(high, low, close, timeperiod=14)
    
    # Returns (using log returns to avoid extreme values)
    for window in window_sizes:
        features[f'returns_{window}'] = np.log1p(close.pct_change(window).clip(-0.99, 10))
        features[f'log_returns_{window}'] = np.log(close / close.shift(window).clip(lower=epsilon))
    
    # Volatility features
    for window in window_sizes:
        features[f'volatility_{window}'] = close.pct_change().rolling(window).std()
        features[f'parkinson_{window}'] = np.sqrt(1 / (4 * np.log(2)) * 
            (np.log(high / low.clip(lower=epsilon))**2).rolling(window).mean())
    
    # Handle NaN values
    features = features.ffill().bfill()
    
    return features

def normalize_features(
    features: pd.DataFrame,
    method: str = 'zscore',
    lookback: int = 252,
    min_periods: int = 20
) -> pd.DataFrame:
    """
    Normalize features using specified method.
    
    Args:
        features: Feature DataFrame
        method: Normalization method ('zscore' or 'minmax')
        lookback: Lookback period for rolling normalization
        min_periods: Minimum periods for rolling calculations
        
    Returns:
        normalized: Normalized feature DataFrame
    """
    if method == 'zscore':
        # Rolling Z-score normalization with bias correction
        means = features.rolling(lookback, min_periods=min_periods).mean()
        stds = features.rolling(lookback, min_periods=min_periods).std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        stds = stds.clip(lower=epsilon)
        
        # Center and scale
        normalized = (features - means) / stds
        
        # Clip extreme values
        normalized = normalized.clip(-3, 3)
        
        # Ensure zero mean
        normalized = normalized - normalized.rolling(lookback, min_periods=min_periods).mean()
        
    elif method == 'minmax':
        # Rolling min-max normalization with proper scaling
        normalized = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
        
        for col in features.columns:
            series = features[col]
            roll_min = series.rolling(lookback, min_periods=min_periods).min()
            roll_max = series.rolling(lookback, min_periods=min_periods).max()
            
            # Handle constant values
            is_constant = (roll_max - roll_min).abs() < 1e-8
            normalized[col] = np.where(
                is_constant,
                0.5,  # Set to middle value if constant
                (series - roll_min) / (roll_max - roll_min + 1e-8)
            )
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Handle any remaining NaN values
    normalized = normalized.fillna(0.5)  # Fill NaNs with middle value
    
    return normalized

def prepare_market_features(data, config):
    """
    Prepare market features for training.
    
    Args:
        data: DataFrame with OHLCV data
        config: Feature configuration dictionary
        
    Returns:
        DataFrame with calculated features
    """
    # Get the first asset's columns (assuming all assets have same structure)
    asset_cols = [col for col in data.columns if 'ASSET_1' in col]
    if not asset_cols:
        # If no ASSET_1 found, assume single asset with direct column names
        price_data = data
    else:
        # Extract base asset name (e.g., 'ASSET_1')
        asset_name = asset_cols[0].split('_')[-1]
        # Select columns for this asset
        price_data = data[[f'open_{asset_name}', f'high_{asset_name}', 
                          f'low_{asset_name}', f'close_{asset_name}', 
                          f'volume_{asset_name}']]
        # Rename columns to standard names
        price_data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Convert to tensor for GPU calculations if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    price_tensor = torch.from_numpy(price_data[['open', 'high', 'low', 'close', 'volume']].values).to(
        device=device, dtype=torch.float32
    )
    
    # Calculate features on GPU
    features_tensor = calculate_technical_features_gpu(
        price_tensor=price_tensor,
        window_sizes=config['window_sizes'],
        device=device
    )
    
    # Normalize features
    normalized_features = normalize_features_gpu(
        features=features_tensor,
        method=config.get('normalization', 'zscore'),
        lookback=config.get('normalization_lookback', 100),
        device=device
    )
    
    # Convert back to CPU and create DataFrame
    feature_names = generate_feature_names(config)
    features = pd.DataFrame(
        normalized_features.cpu().numpy(),
        index=price_data.index,
        columns=feature_names
    )
    
    return features

def calculate_technical_features_gpu(price_tensor: torch.Tensor,
                                   window_sizes: List[int],
                                   device: torch.device) -> torch.Tensor:
    """Calculate technical indicators using GPU acceleration."""
    # Ensure input tensor is float32
    price_tensor = price_tensor.to(dtype=torch.float32)
    
    with torch.amp.autocast('cuda'):
        open_prices = price_tensor[:, 0]
        high_prices = price_tensor[:, 1]
        low_prices = price_tensor[:, 2]
        close_prices = price_tensor[:, 3]
        volume = price_tensor[:, 4]
        
        features = []
        
        for window in window_sizes:
            # Moving averages - use proper padding for 1D data
            pad_size = window - 1
            padded_close = torch.cat([
                close_prices[0].repeat(pad_size),
                close_prices,
                close_prices[-1].repeat(pad_size)
            ])
            
            # Calculate SMA using conv1d with explicit dtype
            weight = torch.ones(1, 1, window, device=device, dtype=torch.float32) / window
            sma = torch.nn.functional.conv1d(
                padded_close.view(1, 1, -1),
                weight
            ).view(-1)
            
            # Exponential moving average - vectorized calculation
            alpha = 2.0 / (window + 1)
            ema = torch.zeros_like(close_prices, dtype=torch.float32)
            ema[0] = close_prices[0]
            ema[1:] = close_prices[1:] * alpha + ema[:-1] * (1 - alpha)
            
            # RSI calculation
            price_diff = close_prices[1:] - close_prices[:-1]
            gains = torch.maximum(price_diff, torch.zeros_like(price_diff))
            losses = torch.maximum(-price_diff, torch.zeros_like(price_diff))
            
            # Pad gains and losses
            padded_gains = torch.cat([
                gains[0].repeat(pad_size),
                gains,
                gains[-1].repeat(pad_size)
            ])
            padded_losses = torch.cat([
                losses[0].repeat(pad_size),
                losses,
                losses[-1].repeat(pad_size)
            ])
            
            # Calculate averages using conv1d
            avg_gains = torch.nn.functional.conv1d(
                padded_gains.view(1, 1, -1),
                weight
            ).view(-1)
            
            avg_losses = torch.nn.functional.conv1d(
                padded_losses.view(1, 1, -1),
                weight
            ).view(-1)
            
            # Calculate RSI with stable division
            rs = avg_gains / (avg_losses + 1e-8)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # VWAP calculation - vectorized
            typical_price = (high_prices + low_prices + close_prices) / 3
            cumul_tp_vol = torch.cumsum(typical_price * volume, dim=0)
            cumul_vol = torch.cumsum(volume, dim=0)
            vwap = cumul_tp_vol / (cumul_vol + 1e-8)
            
            # Volatility calculation
            returns = torch.log(close_prices[1:] / close_prices[:-1].clamp(min=1e-8))
            padded_returns = torch.cat([
                returns[0].repeat(pad_size),
                returns,
                returns[-1].repeat(pad_size)
            ])
            
            # Calculate volatility using conv1d
            vol = torch.sqrt(
                torch.nn.functional.conv1d(
                    (padded_returns ** 2).view(1, 1, -1),
                    weight
                ).view(-1)
            ) * torch.sqrt(torch.tensor(252.0, device=device, dtype=torch.float32))
            
            # Ensure all features have the same length as input
            feature_len = len(close_prices)
            sma = sma[:feature_len]
            ema = ema[:feature_len]
            rsi = torch.cat([rsi[0].unsqueeze(0), rsi])[:feature_len]
            vwap = vwap[:feature_len]
            vol = torch.cat([vol[0].unsqueeze(0), vol])[:feature_len]
            
            # Stack features
            window_features = torch.stack([
                sma, ema, rsi, vwap, vol
            ], dim=1)
            
            features.append(window_features)
        
        # Combine all window sizes
        return torch.cat(features, dim=1)

def normalize_features_gpu(features: torch.Tensor,
                         method: str = 'zscore',
                         lookback: int = 100,
                         device: torch.device = None) -> torch.Tensor:
    """Normalize features using GPU acceleration."""
    if device is None:
        device = features.device
    
    # Ensure input is float32
    features = features.to(dtype=torch.float32)
    
    with torch.amp.autocast('cuda', enabled=False):  # Disable mixed precision for stability
        if method == 'zscore':
            # Prepare convolution weight for rolling calculations
            weight = torch.ones(1, 1, lookback, device=device, dtype=torch.float32) / lookback
            
            # Process each feature separately
            normalized_features = []
            
            for i in range(features.shape[1]):
                feature = features[:, i]
                feature_len = len(feature)
                
                # Pad the feature
                pad_size = lookback - 1
                padded_feature = torch.cat([
                    feature[0].repeat(pad_size),
                    feature,
                    feature[-1].repeat(pad_size)
                ])
                
                # Reshape for conv1d: [batch, channels, length]
                padded_feature = padded_feature.view(1, 1, -1)
                
                # Calculate mean (will have same length as padded_feature - lookback + 1)
                feature_mean_padded = torch.nn.functional.conv1d(
                    padded_feature,
                    weight.view(1, 1, -1),
                    padding=0
                ).squeeze()
                
                # Trim padding from mean to match original feature length
                feature_mean = feature_mean_padded[pad_size:pad_size+feature_len]
                
                # Calculate variance using the trimmed mean
                squared_diff = (feature - feature_mean) ** 2
                
                # Pad squared differences
                padded_squared_diff = torch.cat([
                    squared_diff[0].repeat(pad_size),
                    squared_diff,
                    squared_diff[-1].repeat(pad_size)
                ]).view(1, 1, -1)
                
                # Calculate variance (will have same length as padded_squared_diff - lookback + 1)
                feature_var_padded = torch.nn.functional.conv1d(
                    padded_squared_diff,
                    weight.view(1, 1, -1),
                    padding=0
                ).squeeze()
                
                # Trim padding from variance to match original feature length
                feature_var = feature_var_padded[pad_size:pad_size+feature_len]
                
                # Calculate std dev with numerical stability
                feature_std = torch.sqrt(feature_var + 1e-8)
                
                # Normalize with clipping
                normalized_feature = ((feature - feature_mean) / (feature_std + 1e-8)).clamp(-3, 3)
                normalized_features.append(normalized_feature)
            
            # Stack all normalized features
            normalized = torch.stack(normalized_features, dim=1)
            
        elif method == 'minmax':
            normalized_features = []
            
            for i in range(features.shape[1]):
                feature = features[:, i]
                feature_len = len(feature)
                
                # Calculate rolling windows
                pad_size = lookback - 1
                padded_feature = torch.cat([
                    feature[0].repeat(pad_size),
                    feature,
                    feature[-1].repeat(pad_size)
                ])
                
                # Use unfold for rolling windows
                rolling_windows = padded_feature.unfold(0, lookback, 1)
                rolling_min, _ = rolling_windows.min(dim=1)
                rolling_max, _ = rolling_windows.max(dim=1)
                
                # Trim padding from min/max
                rolling_min = rolling_min[pad_size:pad_size+feature_len]
                rolling_max = rolling_max[pad_size:pad_size+feature_len]
                
                # Handle constant features
                is_constant = (rolling_max - rolling_min).abs() < 1e-8
                feature_normalized = torch.where(
                    is_constant,
                    torch.tensor(0.5, device=device, dtype=torch.float32),
                    (feature - rolling_min) / (rolling_max - rolling_min + 1e-8)
                )
                
                normalized_features.append(feature_normalized)
            
            # Stack and clip
            normalized = torch.stack(normalized_features, dim=1)
            normalized = normalized.clamp(0, 1)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Handle any remaining NaN or Inf values
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized

def generate_feature_names(config: Dict) -> List[str]:
    """Generate feature names based on configuration."""
    feature_names = []
    for window in config['window_sizes']:
        feature_names.extend([
            f'sma_{window}',
            f'ema_{window}',
            f'rsi_{window}',
            f'vwap_{window}',
            f'vol_{window}'
        ])
    return feature_names 