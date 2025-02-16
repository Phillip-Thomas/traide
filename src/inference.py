import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from pathlib import Path
import argparse

from .models import OptimizedDQN
from .environment import SimpleTradeEnvGPU, get_state_size
from .data_management import get_market_data_multi
from .utils import get_device

def visualize_trades(model, env, device):
    """Run model inference and collect trading signals"""
    state = env.reset(random_start=False)  # Start from beginning for visualization
    done = False
    
    prices = []
    signals = []
    portfolio_values = []
    
    while not done:
        # Get current price and portfolio value
        current_price = env.raw_data['Close'][env.idx].item()
        prices.append(current_price)
        portfolio_values.append(env.calculate_portfolio_value().item())
        
        # Get model prediction
        with torch.no_grad():
            state_tensor = state.unsqueeze(0)  # Add batch dimension
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        
        signals.append(action)
        state, _, done = env.step(action)
    
    return np.array(prices), np.array(signals), np.array(portfolio_values)

def plot_trading_signals(data, prices, signals, portfolio_values, save_dir="visualizations"):
    """Create a comprehensive visualization of trading activity"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    fig.suptitle('Trading Model Performance Visualization', fontsize=16)
    
    # Plot 1: Price and Trading Signals
    dates = data.index[-len(prices):]
    ax1.plot(dates, prices, label='Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_points = dates[signals == 1]
    buy_prices = prices[signals == 1]
    ax1.scatter(buy_points, buy_prices, color='green', marker='^', 
                label='Buy Signal', s=100)
    
    # Plot sell signals
    sell_points = dates[signals == 2]
    sell_prices = prices[signals == 2]
    ax1.scatter(sell_points, sell_prices, color='red', marker='v',
                label='Sell Signal', s=100)
    
    ax1.set_title('Price Action and Trading Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Portfolio Value
    initial_value = portfolio_values[0]
    portfolio_returns = (portfolio_values - initial_value) / initial_value * 100
    
    # Calculate buy & hold returns for comparison
    buy_hold_shares = initial_value / prices[0]
    buy_hold_values = buy_hold_shares * prices
    buy_hold_returns = (buy_hold_values - initial_value) / initial_value * 100
    
    ax2.plot(dates, portfolio_returns, 
             label='Strategy Returns (%)', color='blue')
    ax2.plot(dates, buy_hold_returns,
             label='Buy & Hold Returns (%)', color='gray', linestyle='--')
    
    ax2.set_title('Performance Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Returns (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'trading_visualization_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.show()

def calculate_performance_metrics(prices, signals, portfolio_values):
    """Calculate detailed performance metrics"""
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    # Calculate returns
    total_return = (final_value - initial_value) / initial_value * 100
    buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100
    excess_return = total_return - buy_hold_return
    
    # Calculate trade metrics
    trades = signals != 0
    n_trades = np.sum(trades)
    
    # Calculate win rate
    trade_returns = []
    entry_price = None
    for i, signal in enumerate(signals):
        if signal == 1:  # Buy
            entry_price = prices[i]
        elif signal == 2 and entry_price is not None:  # Sell
            trade_return = (prices[i] - entry_price) / entry_price
            trade_returns.append(trade_return)
            entry_price = None
    
    win_rate = np.mean([r > 0 for r in trade_returns]) * 100 if trade_returns else 0
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': excess_return,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_trade_return': np.mean(trade_returns) * 100 if trade_returns else 0,
        'max_drawdown': calculate_max_drawdown(portfolio_values)
    }

def calculate_max_drawdown(values):
    """Calculate maximum drawdown percentage"""
    peak = values[0]
    max_dd = 0
    
    for value in values[1:]:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        max_dd = max(max_dd, dd)
    
    return max_dd

def run_inference(checkpoint_path=None):
    """Run model inference and visualization"""
    device = get_device()
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", "best_checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Please train the model first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get market data
    data_dict = get_market_data_multi(
        tickers=['YBTC'],
        period='1mo',
        interval='15m',
        use_cache=True
    )
    
    if not data_dict:
        print("No data available. Exiting.")
        return
    
    # Initialize model
    window_size = 26  # Same as training
    input_size = get_state_size(window_size)
    model = OptimizedDQN(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    while True:
        # Print available tickers
        tickers = list(data_dict.keys())
        print("\nAvailable tickers:")
        for i, ticker in enumerate(tickers):
            print(f"{i+1}. {ticker}")
        print("0. Exit")
        
        # Get user selection
        try:
            selection = int(input("\nSelect a ticker (enter number, 0 to exit): ")) - 1
            if selection == -1:
                print("Exiting...")
                return
            if 0 <= selection < len(tickers):
                selected_ticker = tickers[selection]
            else:
                print("Invalid selection. Please try again.")
                continue
        except ValueError:
            print("Please enter a number.")
            continue
        
        # Get data for selected ticker
        data = data_dict[selected_ticker]
        
        # Create environment and get trading signals
        env = SimpleTradeEnvGPU(data, window_size=window_size, device=device)
        prices, signals, portfolio_values = visualize_trades(model, env, device)
        
        # Calculate and display metrics
        metrics = calculate_performance_metrics(prices, signals, portfolio_values)
        
        print(f"\nPerformance Summary for {selected_ticker}:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2f}%")
        print(f"Excess Return: {metrics['excess_return']:.2f}%")
        print(f"Number of Trades: {metrics['n_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Average Trade Return: {metrics['avg_trade_return']:.2f}%")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Create visualization
        plot_trading_signals(data, prices, signals, portfolio_values)
        
        input("\nPress Enter to continue...")

def run_single_inference(ticker, checkpoint_path=None, save_dir="visualizations", no_plot=False):
    """Run inference for a single ticker without interactive mode"""
    device = get_device()
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", "best_checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Please train the model first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get market data
    data_dict = get_market_data_multi(
        tickers=[ticker],
        period='1mo',
        interval='15m',
        use_cache=True
    )
    
    if not data_dict or ticker not in data_dict:
        print(f"No data available for {ticker}. Exiting.")
        return
    
    # Initialize model
    window_size = 26  # Same as training
    input_size = get_state_size(window_size)
    model = OptimizedDQN(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create environment and get trading signals
    env = SimpleTradeEnvGPU(data_dict[ticker], window_size=window_size, device=device)
    prices, signals, portfolio_values = visualize_trades(model, env, device)
    
    # Calculate and display metrics
    metrics = calculate_performance_metrics(prices, signals, portfolio_values)
    
    print(f"\nPerformance Summary for {ticker}:")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2f}%")
    print(f"Excess Return: {metrics['excess_return']:.2f}%")
    print(f"Number of Trades: {metrics['n_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Average Trade Return: {metrics['avg_trade_return']:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    
    if not no_plot:
        # Create visualization
        plot_trading_signals(data_dict[ticker], prices, signals, portfolio_values, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trading model inference')
    parser.add_argument('--ticker', type=str, default='YBTC',
                      help='Ticker symbol to analyze (default: YBTC)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint (default: checkpoints/best_checkpoint.pt)')
    parser.add_argument('--save-dir', type=str, default='visualizations',
                      help='Directory to save visualizations (default: visualizations)')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode with multiple tickers')
    parser.add_argument('--no-plot', action='store_true',
                      help='Skip plotting and only show metrics')
    
    args = parser.parse_args()
    
    if args.interactive:
        run_inference(args.checkpoint)
    else:
        run_single_inference(args.ticker, args.checkpoint, args.save_dir, args.no_plot) 