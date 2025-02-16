import os
import torch
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path
import pandas as pd

from .data_management import get_market_data_multi, get_cache_info
from .environment import get_state_size
from .utils import get_device
from .training import train_dqn

def load_global_best():
    """Load the all-time best model at startup"""
    BEST_CHECKPOINT_FILE = os.path.join("checkpoints", "best_checkpoint.pt")
    if os.path.exists(BEST_CHECKPOINT_FILE):
        best_checkpoint = torch.load(BEST_CHECKPOINT_FILE)
        best_profit = best_checkpoint['metrics']['profit']
        best_excess_return = best_checkpoint['metrics'].get('excess_return', float('-inf'))
        print(f"[INFO] Found global best checkpoint with profit={best_profit * 100:.3f}%, excess_return={best_excess_return:.3f}%")
        return best_checkpoint, best_profit, best_excess_return
    else:
        return None, float('-inf'), float('-inf')

def main():
    # Add cache info display
    print("\nCache Information:")
    print(get_cache_info())

    # Get market data with caching for YBTC
    # Using 1 month of data due to yfinance limitations:
    # - 15-minute granularity is only available for last 30 days
    # - This gives us ~4-5 dividend events to learn from
    full_data_dict = get_market_data_multi(
        tickers=['YBTC', 'YETH', 'XDTE', 'QDTE', 'RDTE', 'YMAG', 'YMAX', 'SPYD', 'MSTY', 'TSLY', 'NVDY', 'APLY'],  
        period='1mo',     # Maximum period for 15m data
        interval='15m',   # Highest granularity available
        use_cache=True
    )

    if not full_data_dict:
        print("No data available for training. Exiting.")
        return

    # Print data summary with more detailed statistics
    print("\nData Summary:")
    for ticker, data in full_data_dict.items():
        print(f"\n{ticker} Statistics:")
        print(f"  Total records: {len(data)}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Trading days: {len(pd.Series(data.index.date).unique())}")
        print(f"  Average daily records: {len(data) / len(pd.Series(data.index.date).unique()):.1f}")

    # Split data with clear separation point
    train_data_dict = {}
    val_data_dict = {}
    for ticker, data in full_data_dict.items():
        # Use the last 30% for validation to ensure we're testing on the most recent data
        split_idx = int(len(data) * 0.7)
        split_date = data.index[split_idx]
        
        train_data_dict[ticker] = data[:split_idx].copy()
        val_data_dict[ticker] = data[split_idx:].copy()
        
        print(f"\n{ticker} Training/Validation Split:")
        print(f"  Split date: {split_date}")
        print(f"  Training: {len(train_data_dict[ticker])} records ({train_data_dict[ticker].index[0]} to {train_data_dict[ticker].index[-1]})")
        print(f"  Validation: {len(val_data_dict[ticker])} records ({val_data_dict[ticker].index[0]} to {val_data_dict[ticker].index[-1]})")
        print(f"  Training days: {len(pd.Series(train_data_dict[ticker].index.date).unique())}")
        print(f"  Validation days: {len(pd.Series(val_data_dict[ticker].index.date).unique())}")

    # Define window_size at the top level
    # For 15-minute data:
    # - Trading day is ~6.5 hours = 26 fifteen-minute periods
    # - Using ~1 trading day of context
    window_size = 26  # Approximately one trading day of context
    input_size = get_state_size(window_size)
    
    # Print detailed state size breakdown
    print(f"\nState size breakdown:")
    print(f"  Price features ({window_size} periods x 5 values): {window_size * 5}")
    print(f"  Technical features:")
    print(f"    - RSI ({window_size} periods): {window_size}")
    print(f"    - MACD (3 lines x {window_size} periods): {window_size * 3}")
    print(f"    - Bollinger (3 bands x {window_size} periods): {window_size * 3}")
    print(f"  Yield ETF features:")
    print(f"    - Underlying correlation (3 metrics x {window_size} periods): {window_size * 3}")
    print(f"    - Dividend timing (3 values): 3")
    print(f"  Position info (4 values): 4")
    print(f"  Total features: {input_size}")
    
    print(f"\nInput size: {input_size}")
    print(f"Window size: {window_size} (approximately one trading day of context)")

    device = get_device()
    print(f"Using device: {device}")

    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        world_size = min(num_gpus, len(train_data_dict))
        if world_size > 1:
            print(f"Using {world_size} GPUs!")
            mp.spawn(
                train_dqn,
                args=(world_size, train_data_dict, val_data_dict, input_size, window_size),
                nprocs=world_size,
                join=True
            )
        else:
            print("Using single GPU")
            train_dqn(0, 1, train_data_dict, val_data_dict, input_size, window_size, n_episodes=50)
    else:
        print(f"Using {device}")
        train_dqn(0, 1, train_data_dict, val_data_dict, input_size, window_size, n_episodes=50)

if __name__ == "__main__":
    # Only enable cudnn benchmark for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    main() 
