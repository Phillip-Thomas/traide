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

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

# Import our modules
from src.marketdata.providers.mock import MockMarketDataProvider
from src.data.feature_engineering import prepare_market_features
from src.env.trading_env import TradingEnvironment
from src.utils.risk_management import RiskParams
from src.models.sac_agent import SACAgent
from src.train.train import train_agent

# Configure OpenTelemetry
resource = Resource.create({
    "service.name": os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "train-on-synthetic"),
    "service.version": "1.0.0"
})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otelcollector.whiskey.works:4317"),
    insecure=True,
    timeout=30
)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# Initialize tracer
tracer = trace.get_tracer(__name__)

def main():
    """Main entry point for training script."""
    with tracer.start_as_current_span("main") as main_span:
        try:
            # Set random seeds for reproducibility
            np.random.seed(42)
            torch.manual_seed(42)

            # Create synthetic market data for multiple assets
            with tracer.start_as_current_span("generate_synthetic_data") as data_span:
                print("Generating synthetic market data...")
                mock_provider = MockMarketDataProvider(seed=42)
                
                # Generate data for each asset - reducing to 5 assets initially for simpler learning
                market_data = []
                for i in range(5):  # Reduced from 20 to 5 assets
                    asset_data = mock_provider.fetch_historical_data(
                        symbol=f'ASSET_{i+1}',
                        start_date=datetime.now() - pd.Timedelta(days=500),  # Reduced from 1000 to 500 days
                        end_date=datetime.now(),
                        timeframe='1d'
                    )
                    # Rename columns to match asset
                    asset_data.columns = [f'{col}_ASSET_{i+1}' for col in asset_data.columns]
                    market_data.append(asset_data)
                
                # Combine all asset data
                market_data = pd.concat(market_data, axis=1)
                data_span.set_attribute("n_samples", len(market_data))
                data_span.set_attribute("n_assets", 5)  # Updated to reflect new asset count

            # Calculate features for each asset
            with tracer.start_as_current_span("calculate_features") as feature_span:
                print("Calculating technical features...")
                all_features = []
                for i in range(5):  # Updated range to match n_assets
                    asset_data = market_data[[f'open_ASSET_{i+1}', f'high_ASSET_{i+1}', 
                                            f'low_ASSET_{i+1}', f'close_ASSET_{i+1}', 
                                            f'volume_ASSET_{i+1}']]
                    asset_data.columns = ['open', 'high', 'low', 'close', 'volume']
                    
                    # Further reduced feature set for initial training
                    feature_config = {
                        'window_sizes': [20],  # Single window size
                        'normalization': 'zscore',
                        'normalization_lookback': 100,  # Reduced lookback
                        'features': {
                            'price_based': ['sma', 'ema'],  # Basic price indicators
                            'momentum': ['rsi'],            # Single momentum indicator
                            'volatility': ['volatility']    # Keep volatility
                        }
                    }
                    features = prepare_market_features(asset_data, feature_config)
                    features.columns = [f'{col}_ASSET_{i+1}' for col in features.columns]
                    all_features.append(features)

                # Combine features for all assets
                features = pd.concat(all_features, axis=1)
                feature_span.set_attribute("feature_shape", str(features.shape))

            # Ensure market data and features are aligned
            market_data = market_data.loc[features.index]

            print(f"Data shape: {market_data.shape}, Features shape: {features.shape}")

            # Create config with improved parameters
            with tracer.start_as_current_span("create_config") as config_span:
                config = {
                    "window_size": 20,                  # Reduced window size
                    "commission": 0.001,
                    "start_training_after_steps": 1000, # Reduced initial exploration
                    "save_interval": 5000,
                    "early_stopping_window": 20,        # Reduced for faster feedback
                    "target_sharpe": 0.5,              # Lower initial target
                    "num_episodes": 100,               # Reduced episodes
                    "eval_interval": 5,                # More frequent evaluation
                    "eval_episodes": 2,
                    "risk_params": {
                        "max_position": 0.2,           # Much smaller position size
                        "max_leverage": 0.5,           # Much lower leverage
                        "position_step": 0.02,         # Smaller position steps
                        "max_drawdown": 0.10,          # Tighter drawdown limit
                        "vol_lookback": 20,            # Shorter volatility window
                        "vol_target": 0.05,            # Lower volatility target
                        "transaction_cost": 0.001
                    },
                    "agent_params": {
                        "hidden_dim": 64,              # Much smaller network
                        "buffer_size": 100000,         # Smaller buffer
                        "batch_size": 64,              # Smaller batches
                        "gamma": 0.99,
                        "tau": 0.01,                   # Faster target updates
                        "alpha": 0.1,                  # Lower entropy
                        "learning_rate": 1e-4,         # Slightly higher learning rate
                        "automatic_entropy_tuning": True,
                        "gradient_clip": 0.5,          # Tighter gradient clipping
                        "use_batch_norm": True         # Use batch norm for stability
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
            with tracer.start_as_current_span("train_agent") as train_span:
                print("\nStarting training with reduced complexity:")
                print(f"- Number of assets: 5 (down from 20)")
                print(f"- Training days: 500 (down from 1000)")
                print(f"- Feature set: Basic indicators only")
                print(f"- Network size: 64 hidden units (down from 128)")
                print(f"- Position limits: 20% max position, 50% max leverage\n")
                
                metrics = train_agent(
                    config_path=config_path,
                    price_data=market_data,
                    features=features,
                    save_dir=Path(results_dir),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    disable_progress=False,
                    log_level="INFO",
                    tracer=tracer
                )
                train_span.set_attribute("final_portfolio_value", float(metrics["portfolio_values"][-1]))
                train_span.set_attribute("final_sharpe_ratio", float(metrics["sharpe_ratios"][-1]))

            # Save metrics with additional information
            with tracer.start_as_current_span("save_metrics") as metrics_span:
                # Add training summary
                print("\nTraining Summary:")
                print(f"Total episodes completed: {len(metrics['episode_rewards'])}")
                print(f"Total steps completed: {len(metrics['portfolio_values'])}")
                print(f"Average steps per episode: {len(metrics['portfolio_values']) / len(metrics['episode_rewards']):.1f}")
                
                # Save episode metrics
                episode_metrics = pd.DataFrame({
                    "episode": range(len(metrics["episode_rewards"])),
                    "reward": metrics["episode_rewards"],
                    "sharpe_ratio": metrics["sharpe_ratios"],
                    "max_drawdown": metrics["max_drawdowns"]
                })
                episode_metrics.to_csv(os.path.join(results_dir, "episode_metrics.csv"), index=False)
                
                # Save step metrics
                step_metrics = pd.DataFrame({
                    "step": range(len(metrics["portfolio_values"])),
                    "portfolio_value": metrics["portfolio_values"]
                })
                step_metrics.to_csv(os.path.join(results_dir, "step_metrics.csv"), index=False)
                
                # Save training metrics if available
                if all(key in metrics for key in ["critic_losses", "actor_losses", "alpha_losses", "alphas"]):
                    # Get the minimum length of all arrays to ensure they match
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
                    
                    # Print training statistics
                    print("\nTraining Statistics:")
                    print(f"Final critic loss: {metrics['critic_losses'][-1]:.4f}")
                    print(f"Final actor loss: {metrics['actor_losses'][-1]:.4f}")
                    print(f"Final alpha: {metrics['alphas'][-1]:.4f}")
                else:
                    print("\nNote: No training metrics (losses) were recorded during training")

            # Print final metrics
            print("\nFinal Performance Metrics:")
            print(f"Final portfolio value: {metrics['portfolio_values'][-1]:.2f}")
            print(f"Final Sharpe ratio: {metrics['sharpe_ratios'][-1]:.2f}")
            print(f"Maximum drawdown: {max(metrics['max_drawdowns']):.2%}")
            print(f"Average reward per episode: {np.mean(metrics['episode_rewards']):.2f}")

            print(f"\nResults saved to: {results_dir}")

        except Exception as e:
            main_span.set_status(Status(StatusCode.ERROR, str(e)))
            main_span.record_exception(e)
            raise

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure spans are flushed before exit
        tracer_provider.force_flush()
        tracer_provider.shutdown() 