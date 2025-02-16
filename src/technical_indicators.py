import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def calculate_rsi(prices, period=14):
    """Calculate RSI for a given price series"""
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period - 1) + upval)/period
        down = (down*(period - 1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal line, and Histogram"""
    # Convert to pandas Series first
    prices = pd.Series(prices)
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    # Convert back to numpy arrays
    return np.array(macd), np.array(signal_line), np.array(histogram)

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    prices = np.array(prices)
    rolling_mean = np.array(pd.Series(prices).rolling(window=period).mean())
    rolling_std = np.array(pd.Series(prices).rolling(window=period).std())
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_ma(prices, period):
    """Calculate Moving Average"""
    return np.array(pd.Series(prices).rolling(window=period).mean())

def calculate_volatility(prices, period=20):
    """Calculate price volatility"""
    returns = np.diff(prices) / prices[:-1]
    return np.array(pd.Series(returns).rolling(window=period).std())

def calculate_trend_strength(prices, period=14):
    """Calculate trend strength using ADX-like calculation"""
    prices = np.array(prices)
    dx = np.abs(np.diff(prices)) / prices[:-1]
    trend_strength = np.array(pd.Series(dx).rolling(window=period).mean())
    return np.pad(trend_strength, (1,0), mode='edge')  # Pad to match length

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    prices = np.array(prices)
    peak = prices[0]
    max_dd = 0

    for price in prices[1:]:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        max_dd = max(max_dd, dd)

    return max_dd

# GPU-accelerated versions
def calculate_rsi_gpu(prices_tensor, period=14, device=None):
    """Calculate RSI directly on GPU"""
    if device is None:
        device = prices_tensor.device

    # Cast to float32 for calculations
    prices_float = prices_tensor.to(torch.float32)

    # Calculate deltas
    deltas = prices_float[1:] - prices_float[:-1]

    # Initialize up and down tensors
    up = torch.zeros_like(deltas)
    down = torch.zeros_like(deltas)

    # Split into up and down moves
    up[deltas > 0] = deltas[deltas > 0]
    down[deltas < 0] = -deltas[deltas < 0]

    # Calculate rolling averages
    up_avg = torch.zeros_like(prices_float)
    down_avg = torch.zeros_like(prices_float)

    # First period
    up_avg[period] = up[:period].mean()
    down_avg[period] = down[:period].mean()

    # Calculate rest of periods
    alpha = 1.0 / period
    for i in range(period + 1, len(prices_float)):
        up_avg[i] = alpha * up[i-1] + (1 - alpha) * up_avg[i-1]
        down_avg[i] = alpha * down[i-1] + (1 - alpha) * down_avg[i-1]

    # Calculate RS and RSI
    rs = up_avg / (down_avg + 1e-7)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Convert back to original dtype
    return rsi.to(prices_tensor.dtype)

def calculate_macd_gpu(prices_tensor, fast=12, slow=26, signal=9, device=None):
    """Calculate MACD directly on GPU"""
    if device is None:
        device = prices_tensor.device

    # Cast to float32 for calculations
    prices_float = prices_tensor.to(torch.float32)

    # Calculate EMAs
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    alpha_signal = 2.0 / (signal + 1)

    fast_ema = torch.zeros_like(prices_float)
    slow_ema = torch.zeros_like(prices_float)

    # Initialize first value
    fast_ema[0] = prices_float[0]
    slow_ema[0] = prices_float[0]

    # Calculate EMAs
    for i in range(1, len(prices_float)):
        fast_ema[i] = alpha_fast * prices_float[i] + (1 - alpha_fast) * fast_ema[i-1]
        slow_ema[i] = alpha_slow * prices_float[i] + (1 - alpha_slow) * slow_ema[i-1]

    # Calculate MACD
    macd = fast_ema - slow_ema

    # Calculate signal line
    signal_line = torch.zeros_like(macd)
    signal_line[0] = macd[0]

    for i in range(1, len(macd)):
        signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i-1]

    # Calculate histogram
    histogram = macd - signal_line

    # Convert back to original dtype
    return (
        macd.to(prices_tensor.dtype),
        signal_line.to(prices_tensor.dtype),
        histogram.to(prices_tensor.dtype)
    )

def calculate_bollinger_bands_gpu(prices_tensor, period=20, num_std=2, device=None):
    """Calculate Bollinger Bands directly on GPU"""
    if device is None:
        device = prices_tensor.device

    # Ensure kernel is same dtype as input
    kernel = torch.ones(period, device=device, dtype=prices_tensor.dtype) / period
    kernel = kernel.view(1, 1, -1)

    # Cast input to float32 for numerical stability in convolution
    prices_float = prices_tensor.to(torch.float32)
    prices_padded = torch.nn.functional.pad(prices_float.view(1, 1, -1), (period-1, 0))

    # Calculate rolling mean
    rolling_mean = torch.nn.functional.conv1d(prices_padded, kernel.to(torch.float32)).squeeze()

    # Calculate rolling std
    squared_diff = (prices_float - rolling_mean) ** 2
    squared_diff_padded = torch.nn.functional.pad(squared_diff.view(1, 1, -1), (period-1, 0))
    rolling_var = torch.nn.functional.conv1d(squared_diff_padded, kernel.to(torch.float32)).squeeze()
    rolling_std = torch.sqrt(rolling_var + 1e-8)

    # Calculate bands and convert back to original dtype
    upper_band = (rolling_mean + (rolling_std * num_std)).to(prices_tensor.dtype)
    middle_band = rolling_mean.to(prices_tensor.dtype)
    lower_band = (rolling_mean - (rolling_std * num_std)).to(prices_tensor.dtype)

    return upper_band, middle_band, lower_band 