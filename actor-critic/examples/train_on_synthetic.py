#!/usr/bin/env python3
"""Example script demonstrating how to train the SAC trading agent on synthetic data."""

import os
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
        n_assets=20,  # Reduced from 100 to 20 assets for initial training
        seed=42
    )

    # Calculate features for each asset
    print("Calculating technical features...")
    all_features = []
    for i in range(20):  # Updated range to match n_assets
        asset_data = market_data[[f'open_ASSET_{i+1}', f'high_ASSET_{i+1}', 
                                f'low_ASSET_{i+1}', f'close_ASSET_{i+1}', 
                                f'volume_ASSET_{i+1}']]
        asset_data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Reduced feature set focusing on essential indicators
        feature_config = {
            'window_sizes': [20, 50],  # Reduced window sizes
            'normalization': 'zscore',
            'normalization_lookback': 504,  # Increased to 2 trading years
            'features': {
                'price_based': ['sma', 'ema', 'bbands'],  # Essential price indicators
                'momentum': ['rsi', 'macd'],              # Key momentum indicators
                'volume': ['vwap'],                       # Volume-weighted price
                'volatility': ['volatility']              # Volatility measure
            }
        }
        features = prepare_market_features(asset_data, feature_config)
        features.columns = [f'{col}_ASSET_{i+1}' for col in features.columns]
        all_features.append(features)

    # Combine features for all assets
    features = pd.concat(all_features, axis=1)

    # Ensure market data and features are aligned
    market_data = market_data.loc[features.index]

    print(f"Data shape: {market_data.shape}, Features shape: {features.shape}")

    # Create config with improved parameters
    config = {
        "window_size": 50,
        "commission": 0.001,
        "start_training_after_steps": 5000,  # Increased initial exploration
        "save_interval": 10000,
        "early_stopping_window": 50,         # Increased for more stable evaluation
        "target_sharpe": 1.0,               # More realistic target
        "num_episodes": 200,                # Increased training episodes
        "risk_params": {
            "max_position": 0.5,            # Reduced position size
            "max_leverage": 0.8,            # Reduced leverage
            "position_step": 0.05,          # Finer position adjustments
            "max_drawdown": 0.15,           # Stricter drawdown limit
            "vol_lookback": 50,             # Increased volatility window
            "vol_target": 0.10,             # Reduced volatility target
            "transaction_cost": 0.001
        },
        "agent_params": {
            "hidden_dim": 128,              # Reduced network complexity
            "buffer_size": 500000,          # Reduced buffer size
            "batch_size": 256,              # Increased batch size
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learning_rate": 5e-5,          # Reduced learning rate
            "automatic_entropy_tuning": True,
            "gradient_clip": 1.0,           # Added gradient clipping
            "use_batch_norm": False         # Using LayerNorm instead of BatchNorm
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
        price_data=market_data,
        features=features,
        save_dir=Path(results_dir)
    )

    # Save metrics
    episode_metrics = pd.DataFrame({
        "episode": range(len(metrics["episode_rewards"])),
        "reward": metrics["episode_rewards"],
        "sharpe_ratio": metrics["sharpe_ratios"],
        "max_drawdown": metrics["max_drawdowns"]
    })
    episode_metrics.to_csv(os.path.join(results_dir, "episode_metrics.csv"), index=False)
    
    # Save step metrics separately
    step_metrics = pd.DataFrame({
        "step": range(len(metrics["portfolio_values"])),
        "portfolio_value": metrics["portfolio_values"]
    })
    step_metrics.to_csv(os.path.join(results_dir, "step_metrics.csv"), index=False)
    
    # Save training metrics separately if they exist
    if metrics["critic_losses"]:
        # Get the minimum length of all arrays to ensure they match
        min_length = min(len(metrics["critic_losses"]), 
                        len(metrics["actor_losses"]),
                        len(metrics["alpha_losses"]),
                        len(metrics["alphas"]))
        
        training_metrics = pd.DataFrame({
            "step": range(min_length),
            "critic_loss": metrics["critic_losses"][:min_length],
            "actor_loss": metrics["actor_losses"][:min_length],
            "alpha_loss": metrics["alpha_losses"][:min_length],
            "alpha": metrics["alphas"][:min_length]
        })
        training_metrics.to_csv(os.path.join(results_dir, "training_metrics.csv"), index=False)

    # Print final metrics
    print("\nTraining completed!")
    print(f"Final portfolio value: {metrics['portfolio_values'][-1]:.2f}")
    print(f"Final Sharpe ratio: {metrics['sharpe_ratios'][-1]:.2f}")
    print(f"Maximum drawdown: {max(metrics['max_drawdowns']):.2%}")

    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main() 