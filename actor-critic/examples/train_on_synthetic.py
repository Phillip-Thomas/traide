#!/usr/bin/env python3
"""Example script demonstrating how to train the SAC trading agent on synthetic data."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.market_data import create_synthetic_data
from src.data.feature_engineering import prepare_market_features
from src.env.trading_env import TradingEnvironment
from src.utils.risk_management import RiskParams
from src.models.sac_agent import SACAgent
from src.train.train import train_agent

def main():
    """Main entry point for training script."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create synthetic market data for multiple assets
    print("Generating synthetic market data...")
    market_data = create_synthetic_data(
        n_samples=1000,
        n_assets=1,  # Simplified to single asset
        seed=42
    )

    # Calculate features
    print("Calculating technical features...")
    asset_data = market_data[['open_ASSET_1', 'high_ASSET_1', 
                             'low_ASSET_1', 'close_ASSET_1', 
                             'volume_ASSET_1']]
    asset_data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Reduced feature set focusing on essential indicators
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

    # Ensure market data and features are aligned
    market_data = market_data.loc[features.index]
    
    # Extract just the close price for the single asset
    price_data = market_data[['close_ASSET_1']]

    print(f"Data shape: {price_data.shape}, Features shape: {features.shape}")

    # Create simplified config
    config = {
        "window_size": 50,
        "commission": 0.001,
        "save_interval": 10000,
        "num_episodes": 200,
        # Add environment parameters
        "env_params": {
            "reward_scaling": 1.0,
            "risk_aversion": 0.1,
            "vol_lookback": 20
        },
        "agent_params": {
            "hidden_dim": 128,
            "buffer_size": 500000,
            "batch_size": 256,
            "gamma": 0.99,  # Discount factor for future rewards
            "tau": 0.005,   # Soft update coefficient
            "alpha": 0.2,   # Temperature parameter for exploration
            "learning_rate": 5e-5,
            "automatic_entropy_tuning": True,
            "gradient_clip": 1.0,
            "use_batch_norm": False
        }
    }

    # Save config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Train the agent
    print("Starting training...")
    metrics = train_agent(
        config_path=config_path,
        price_data=price_data,
        features=features,
        save_dir=Path(results_dir),
        disable_progress=False,  # Enable progress bars
        log_level="INFO"  # Set logging level to INFO
    )

    # Save metrics with additional trading statistics
    episode_metrics = pd.DataFrame({
        "episode": range(len(metrics["episode_rewards"])),
        "reward": metrics["episode_rewards"],
        "portfolio_value": metrics["portfolio_values"],
        "sharpe_ratio": metrics["sharpe_ratios"],
        "max_drawdown": metrics["max_drawdowns"],
        "avg_position": metrics["avg_positions"],
        "total_costs": metrics["total_costs"],
        "volatility": metrics["volatilities"]
    })
    episode_metrics.to_csv(os.path.join(results_dir, "episode_metrics.csv"), index=False)
    
    # Save training metrics if they exist
    if metrics.get("critic_losses"):
        min_length = min(
            len(metrics["critic_losses"]), 
            len(metrics["actor_losses"]),
            len(metrics["alpha_losses"]),
            len(metrics["alphas"])
        )
        
        training_metrics = pd.DataFrame({
            "step": range(min_length),
            "critic_loss": metrics["critic_losses"][:min_length],
            "actor_loss": metrics["actor_losses"][:min_length],
            "alpha_loss": metrics["alpha_losses"][:min_length],
            "alpha": metrics["alphas"][:min_length]
        })
        training_metrics.to_csv(os.path.join(results_dir, "training_metrics.csv"), index=False)

    # Print final metrics with enhanced statistics
    print("\nTraining completed!")
    print(f"Final portfolio value: {metrics['portfolio_values'][-1]:.2f}")
    print(f"Final Sharpe ratio: {metrics['sharpe_ratios'][-1]:.2f}")
    print(f"Max drawdown: {max(metrics['max_drawdowns']):.2%}")
    print(f"Average position size: {np.mean(metrics['avg_positions']):.3f}")
    print(f"Total transaction costs: {sum(metrics['total_costs']):.4f}")
    print(f"Final volatility: {metrics['volatilities'][-1]:.2%}")
    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main() 