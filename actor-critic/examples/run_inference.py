#!/usr/bin/env python3
"""Script for running inference with the trained SAC trading agent."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.market_data import create_synthetic_data
from src.data.feature_engineering import prepare_market_features
from src.env.trading_env import TradingEnvironment
from src.utils.risk_management import RiskParams
from src.models.sac_agent import SACAgent

def run_inference(
    model_path: str,
    config_path: str,
    n_samples: int = 1000,
    seed: int = 42,
    save_dir: str = None
):
    """Run inference with a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the config file used during training
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        save_dir: Directory to save results and plots
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate synthetic data
    print("Generating synthetic market data...")
    market_data = create_synthetic_data(
        n_samples=n_samples,
        n_assets=1,
        seed=seed
    )
    
    # Calculate features
    print("Calculating technical features...")
    asset_data = market_data[['open_ASSET_1', 'high_ASSET_1', 
                             'low_ASSET_1', 'close_ASSET_1', 
                             'volume_ASSET_1']]
    asset_data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    feature_config = {
        'window_sizes': [20, 50],
        'normalization': 'zscore',
        'normalization_lookback': 100,
        'features': {
            'price_based': ['sma', 'ema', 'bbands'],
            'momentum': ['rsi', 'macd'],
            'volume': ['vwap'],
            'volatility': ['volatility']
        }
    }
    features = prepare_market_features(asset_data, feature_config)
    
    # Align market data and features
    market_data = market_data.loc[features.index]
    price_data = market_data[['close_ASSET_1']]
    
    # Create environment
    env = TradingEnvironment(
        price_data=price_data,
        features=features,
        window_size=config['window_size'],
        commission=config['commission'],
        reward_scaling=config['env_params']['reward_scaling'],
        risk_aversion=config['env_params']['risk_aversion'],
        vol_lookback=config['env_params']['vol_lookback']
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['agent_params']['hidden_dim'],
        buffer_size=config['agent_params']['buffer_size'],
        batch_size=config['agent_params']['batch_size'],
        gamma=config['agent_params']['gamma'],
        tau=config['agent_params']['tau'],
        alpha=config['agent_params']['alpha'],
        learning_rate=config['agent_params']['learning_rate'],
        automatic_entropy_tuning=config['agent_params']['automatic_entropy_tuning'],
        gradient_clip=config['agent_params']['gradient_clip'],
        use_batch_norm=config['agent_params']['use_batch_norm']
    )
    
    # Load trained model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    agent.load(model_path)  # Use the agent's load method instead of load_state_dict
    
    # Set networks to evaluation mode
    agent.actor.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()
    agent.critic_1_target.eval()
    agent.critic_2_target.eval()
    
    # Run inference
    print("Running inference...")
    state, _ = env.reset()  # Properly unpack the reset() return values
    done = False
    portfolio_values = []  # Start empty
    positions = []
    returns = []
    
    # Record initial portfolio value
    portfolio_values.append(1.0)
    positions.append(0.0)  # Initial position is 0
    returns.append(0.0)  # Initial return is 0
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, truncated, info = env.step(action)  # Properly unpack all 5 values
        done = done or truncated  # Consider both done and truncated conditions
        
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        returns.append(info['step_return'])
        
        state = next_state
    
    # Calculate metrics
    total_return = portfolio_values[-1] - portfolio_values[0]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized, added epsilon
    max_drawdown = np.min([1 - val/np.maximum.accumulate(portfolio_values) for val in portfolio_values])
    avg_position = np.mean(np.abs(positions))
    
    print("\nInference Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Average Position Size: {avg_position:.3f}")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Create DataFrame with aligned indices
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_values[1:],
            'position': positions[1:],
            'return': returns[1:]
        })
        results_df.to_csv(save_dir / 'inference_results.csv', index=False)
        
        # Create trading visualization
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(15, 10))
        
        # Main chart with candlesticks and positions
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        
        # Prepare OHLC data with proper indexing
        # Skip the first window_size bars since that's where trading starts
        trading_start = config['window_size']
        trading_data = asset_data.iloc[trading_start:].copy()
        trading_data = trading_data.reset_index(drop=True)
        
        # Prepare candlestick data
        width = 0.8
        width2 = width/2
        
        up = trading_data[trading_data['close'] >= trading_data['open']]
        down = trading_data[trading_data['close'] < trading_data['open']]
        
        # Plot up candles
        ax1.bar(up.index, up['close']-up['open'], width, bottom=up['open'], color='g', alpha=1)
        ax1.bar(up.index, up['high']-up['close'], width2, bottom=up['close'], color='g', alpha=1)
        ax1.bar(up.index, up['low']-up['open'], width2, bottom=up['open'], color='g', alpha=1)
        
        # Plot down candles
        ax1.bar(down.index, down['close']-down['open'], width, bottom=down['open'], color='r', alpha=1)
        ax1.bar(down.index, down['high']-down['open'], width2, bottom=down['open'], color='r', alpha=1)
        ax1.bar(down.index, down['low']-down['close'], width2, bottom=down['close'], color='r', alpha=1)
        
        # Plot positions as a heatmap overlay
        for i, pos in enumerate(positions[1:]):  # Skip initial position
            color = 'green' if pos > 0 else 'red' if pos < 0 else 'gray'
            alpha = min(abs(pos) * 0.3, 0.3) if pos != 0 else 0.1
            ax1.axvspan(i, i+1, alpha=alpha, color=color, zorder=0)
        
        # Set proper axis limits
        ax1.set_xlim(-1, len(positions[1:]) + 1)
        y_min = trading_data['low'].min() * 0.995
        y_max = trading_data['high'].max() * 1.005
        ax1.set_ylim(y_min, y_max)
        
        plt.title('Trading Activity', pad=20)
        plt.ylabel('Price')
        plt.grid(True, alpha=0.2)
        
        # Add portfolio value subplot
        ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
        plt.plot(range(len(portfolio_values)), portfolio_values, color='cyan', linewidth=2)
        plt.title('Portfolio Value', pad=20)
        plt.ylabel('Value')
        plt.grid(True, alpha=0.2)
        
        # Set proper x-axis limits for portfolio value plot
        ax2.set_xlim(-1, len(positions[1:]) + 1)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.3, label='Long Position'),
            Patch(facecolor='red', alpha=0.3, label='Short Position'),
            Patch(facecolor='gray', alpha=0.1, label='No Position')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_dir / 'trading_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to {save_dir}")

def main():
    """Main entry point for inference script."""
    # Set paths
    model_path = os.path.join(os.path.dirname(__file__), "results", "model_final.pt")
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    save_dir = os.path.join(os.path.dirname(__file__), "results", "inference")
    
    # Run inference
    run_inference(
        model_path=model_path,
        config_path=config_path,
        n_samples=1000,
        seed=42,
        save_dir=save_dir
    )

if __name__ == "__main__":
    main() 