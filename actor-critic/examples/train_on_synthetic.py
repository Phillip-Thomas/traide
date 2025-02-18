#!/usr/bin/env python3
"""Example script demonstrating how to train the SAC trading agent on synthetic data."""

import sys
from pathlib import Path
import yaml
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.market_data import create_synthetic_data
from src.data.feature_engineering import prepare_market_features
from src.train.train import train_agent

def main():
    # Generate synthetic data
    print("Generating synthetic market data...")
    market_data = create_synthetic_data(
        n_samples=1000,  # 1000 days
        n_assets=1,      # Single asset for simplicity
        seed=42
    )
    
    # Rename columns to match feature engineering expectations
    market_data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Prepare features
    print("Calculating technical features...")
    feature_config = {
        'window_sizes': [14, 30, 50],
        'normalization': 'zscore',
        'normalization_lookback': 252
    }
    features = prepare_market_features(market_data, feature_config)
    
    # Ensure market_data and features have matching indices
    market_data = market_data.loc[features.index]
    
    print(f"Data shape: {market_data.shape}, Features shape: {features.shape}")
    
    # Create save directory
    save_dir = project_root / "examples" / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    config = {
        "num_episodes": 100,
        "window_size": 50,
        "commission": 0.001,
        "start_training_after_steps": 1000,
        "save_interval": 10000,
        "early_stopping_window": 20,
        "target_sharpe": 1.5,
        "risk_params": {
            "max_position": 1.0,
            "max_leverage": 1.0,
            "position_step": 0.1,
            "max_drawdown": 0.15,
            "vol_lookback": 20,
            "vol_target": 0.15,
            "transaction_cost": 0.001
        },
        "agent_params": {
            "hidden_dim": 256,
            "buffer_size": 100000,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learning_rate": 0.0003,
            "automatic_entropy_tuning": True
        }
    }
    
    # Save config
    config_path = save_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Train agent
    print("Starting training...")
    metrics = train_agent(
        config_path=str(config_path),
        price_data=market_data,
        features=features,
        save_dir=save_dir,
        device=device
    )
    
    # Print final metrics
    print("\nTraining completed!")
    print(f"Final portfolio value: {metrics['portfolio_values'][-1]:.2f}")
    print(f"Final Sharpe ratio: {metrics['sharpe_ratios'][-1]:.2f}")
    print(f"Maximum drawdown: {max(metrics['max_drawdowns']):.2%}")
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == "__main__":
    main() 