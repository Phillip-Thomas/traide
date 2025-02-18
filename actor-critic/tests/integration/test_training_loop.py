import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.train.train import train_agent, evaluate_agent
from src.utils.risk_management import RiskParams

@pytest.fixture
def sample_data():
    """Create sample price and feature data for testing."""
    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 1000
    
    prices = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Generate synthetic features
    features = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples)
    })
    
    return prices, features

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.mark.slow
def test_training_loop(sample_data, temp_dir):
    """Test complete training loop execution."""
    price_data, features = sample_data
    
    # Create minimal config for testing
    config = {
        "num_episodes": 2,
        "window_size": 10,
        "commission": 0.001,
        "start_training_after_steps": 100,
        "save_interval": 100,
        "early_stopping_window": 2,
        "target_sharpe": 10.0,  # High value to prevent early stopping
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
            "hidden_dim": 64,
            "buffer_size": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learning_rate": 0.0003,
            "automatic_entropy_tuning": True
        }
    }
    
    # Save config
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    # Run training
    metrics = train_agent(
        config_path=str(config_path),
        price_data=price_data,
        features=features,
        save_dir=temp_dir
    )
    
    # Check training outputs
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        "episode_rewards",
        "portfolio_values",
        "sharpe_ratios",
        "max_drawdowns",
        "critic_losses",
        "actor_losses",
        "alpha_losses",
        "alphas"
    ])
    
    # Check model files
    assert (temp_dir / "model_final.pt").exists()
    assert (temp_dir / "training_metrics.csv").exists()
    
    # Check tensorboard logs
    assert (temp_dir / "logs").exists()

@pytest.mark.slow
def test_evaluation(sample_data, temp_dir):
    """Test model evaluation."""
    price_data, features = sample_data
    
    # First train a model
    config = {
        "num_episodes": 2,
        "window_size": 10,
        "commission": 0.001,
        "start_training_after_steps": 100,
        "save_interval": 100,
        "early_stopping_window": 2,
        "target_sharpe": 10.0,
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
            "hidden_dim": 64,
            "buffer_size": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learning_rate": 0.0003,
            "automatic_entropy_tuning": True
        }
    }
    
    # Save config
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    # Train model
    train_agent(
        config_path=str(config_path),
        price_data=price_data,
        features=features,
        save_dir=temp_dir
    )
    
    # Evaluate model
    metrics = evaluate_agent(
        model_path=str(temp_dir / "model_final.pt"),
        price_data=price_data,
        features=features,
        risk_params=RiskParams(**config["risk_params"])
    )
    
    # Check evaluation metrics
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        "final_portfolio_value",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "avg_position",
        "position_changes"
    ])
    
    # Check metric values are reasonable
    assert float(metrics["final_portfolio_value"]) > 0
    assert -100 <= float(metrics["total_return"]) <= 100
    assert -10 <= float(metrics["sharpe_ratio"]) <= 10
    assert 0 <= float(metrics["max_drawdown"]) <= 1
    assert 0 <= float(metrics["avg_position"]) <= 1

@pytest.mark.slow
def test_training_gpu_availability(sample_data, temp_dir):
    """Test training with GPU if available."""
    price_data, features = sample_data
    
    # Create config
    config = {
        "num_episodes": 1,
        "window_size": 10,
        "commission": 0.001,
        "start_training_after_steps": 100,
        "save_interval": 100,
        "early_stopping_window": 2,
        "target_sharpe": 10.0,
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
            "hidden_dim": 64,
            "buffer_size": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learning_rate": 0.0003,
            "automatic_entropy_tuning": True
        }
    }
    
    # Save config
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    # Train with GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = train_agent(
        config_path=str(config_path),
        price_data=price_data,
        features=features,
        save_dir=temp_dir,
        device=device
    )
    
    assert isinstance(metrics, dict)
    assert len(metrics["episode_rewards"]) > 0
    # Convert any pandas Series to float for comparison
    for key in metrics:
        if isinstance(metrics[key], pd.Series):
            metrics[key] = metrics[key].values 