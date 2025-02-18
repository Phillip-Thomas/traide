import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import yaml

from src.train.train import train_agent, evaluate_agent
from src.models.sac_agent import SACAgent
from src.env.trading_env import TradingEnvironment
from src.utils.risk_management import RiskParams

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    n_samples = 200
    np.random.seed(42)
    
    # Generate synthetic price data
    time = np.linspace(0, 4*np.pi, n_samples)
    trend1 = np.sin(time) * 10 + time * 2
    trend2 = np.cos(time) * 5 + time * 1.5
    
    price_data = pd.DataFrame({
        'close_ASSET_1': 100 + trend1,
        'high_ASSET_1': 102 + trend1,
        'low_ASSET_1': 98 + trend1,
        'volume_ASSET_1': np.random.randint(1000, 10000, n_samples),
        'close_ASSET_2': 50 + trend2,
        'high_ASSET_2': 52 + trend2,
        'low_ASSET_2': 48 + trend2,
        'volume_ASSET_2': np.random.randint(1000, 10000, n_samples)
    })
    
    # Generate features
    features = pd.DataFrame({
        'feature1_ASSET_1': np.random.randn(n_samples),
        'feature2_ASSET_1': np.random.randn(n_samples),
        'feature1_ASSET_2': np.random.randn(n_samples),
        'feature2_ASSET_2': np.random.randn(n_samples)
    })
    
    return price_data, features

@pytest.fixture
def temp_dir():
    """Create temporary directory for saving models and configs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def config_file(temp_dir):
    """Create sample training configuration file."""
    config = {
        "num_episodes": 2,
        "start_training_after_steps": 50,
        "save_interval": 100,
        "early_stopping_window": 10,
        "target_sharpe": 2.0,
        "window_size": 10,
        "commission": 0.001,
        "eval_interval": 1,
        "eval_episodes": 1,
        "risk_params": {
            "max_position": 1.0,
            "max_leverage": 1.0,
            "position_step": 0.5,
            "max_drawdown": 0.15,
            "vol_lookback": 20,
            "vol_target": 0.15,
            "transaction_cost": 0.001
        },
        "agent_params": {
            "hidden_dim": 64,
            "learning_rate": 0.001,
            "batch_size": 32,
            "buffer_size": 1000,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "automatic_entropy_tuning": True,
            "use_batch_norm": False
        }
    }
    
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def test_train_agent(sample_data, temp_dir, config_file):
    """Test training function."""
    price_data, features = sample_data
    
    # Train agent
    metrics = train_agent(
        config_path=str(config_file),
        price_data=price_data,
        features=features,
        save_dir=temp_dir
    )
    
    # Check metrics
    assert "episode_rewards" in metrics
    assert "portfolio_values" in metrics
    assert "sharpe_ratios" in metrics
    assert "max_drawdowns" in metrics
    assert len(metrics["episode_rewards"]) > 0
    
    # Check saved files
    assert (temp_dir / "model_final.pt").exists()
    assert (temp_dir / "training_metrics.csv").exists()
    assert (temp_dir / "logs").exists()

def test_evaluate_agent(sample_data, temp_dir):
    """Test evaluation function."""
    price_data, features = sample_data
    
    # Create and save a test model
    env = TradingEnvironment(
        price_data=price_data,
        features=features,
        risk_params=RiskParams(),
        window_size=10
    )
    
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=64,
        learning_rate=0.001,
        use_batch_norm=False
    )
    
    model_path = temp_dir / "test_model.pt"
    agent.save(model_path)
    
    # Evaluate agent
    metrics = evaluate_agent(
        agent=agent,
        env=env,
        n_episodes=1
    )
    
    # Check metrics
    assert "final_portfolio_value" in metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "avg_position" in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_train_agent_early_stopping(sample_data, temp_dir, config_file):
    """Test early stopping functionality."""
    price_data, features = sample_data
    
    # Modify config for quick early stopping
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    config["early_stopping_window"] = 1
    config["target_sharpe"] = -20.0  # Set very low target to trigger early stopping
    config["num_episodes"] = 20  # Increase total episodes
    config["eval_interval"] = 1  # Evaluate every episode
    
    modified_config_path = temp_dir / "modified_config.yaml"
    with open(modified_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Train agent
    metrics = train_agent(
        config_path=str(modified_config_path),
        price_data=price_data,
        features=features,
        save_dir=temp_dir
    )
    
    # Check early stopping
    assert len(metrics["episode_rewards"]) < config["num_episodes"]

def test_train_agent_checkpointing(sample_data, temp_dir, config_file):
    """Test model checkpointing during training."""
    price_data, features = sample_data
    
    # Modify config for frequent checkpointing
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    config["save_interval"] = 10  # Save more frequently
    
    modified_config_path = temp_dir / "checkpoint_config.yaml"
    with open(modified_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Train agent
    train_agent(
        config_path=str(modified_config_path),
        price_data=price_data,
        features=features,
        save_dir=temp_dir
    )
    
    # Check for checkpoint files
    checkpoint_files = list(temp_dir.glob("model_step_*.pt"))
    assert len(checkpoint_files) > 0 