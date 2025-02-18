import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import tempfile
import shutil
import sys
import torch
from unittest.mock import patch, MagicMock

from src.main import main
from src.models.sac_agent import SACAgent
from src.env.trading_env import TradingEnvironment
from src.utils.risk_management import RiskParams

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    n_samples = 100
    np.random.seed(42)
    
    price_data = pd.DataFrame({
        'close_ASSET_1': np.random.randn(n_samples).cumsum() + 100,
        'high_ASSET_1': np.random.randn(n_samples).cumsum() + 102,
        'low_ASSET_1': np.random.randn(n_samples).cumsum() + 98,
        'volume_ASSET_1': np.random.randint(1000, 10000, n_samples)
    })
    
    features = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    
    return price_data, features

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def config_file(temp_dir):
    """Create sample configuration file."""
    config = {
        "num_episodes": 2,
        "window_size": 10,
        "commission": 0.001,
        "start_training_after_steps": 100,
        "save_interval": 1000,
        "early_stopping_window": 10,
        "target_sharpe": 1.5,
        "eval_interval": 1,
        "eval_episodes": 1,
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
            "automatic_entropy_tuning": True,
            "use_batch_norm": False
        }
    }
    
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
def mock_model(sample_data, temp_dir):
    """Create and save a mock model for testing."""
    price_data, features = sample_data
    
    # Create environment and agent
    env = TradingEnvironment(
        price_data=price_data,
        features=features,
        risk_params=RiskParams(),
        window_size=10
    )
    
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=64
    )
    
    # Save model
    model_path = temp_dir / "model.pt"
    agent.save(str(model_path))
    
    return model_path

def test_main_training_mode(sample_data, temp_dir, config_file):
    """Test main function in training mode."""
    price_data, features = sample_data
    
    # Save sample data
    price_data_path = temp_dir / "price_data.csv"
    features_path = temp_dir / "features.csv"
    price_data.to_csv(price_data_path, index=False)
    features.to_csv(features_path, index=False)
    
    # Mock command line arguments
    test_args = [
        "main.py",
        "--mode", "train",
        "--config", str(config_file),
        "--price_data", str(price_data_path),
        "--features", str(features_path),
        "--save_dir", str(temp_dir)
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
    
    # Check output files
    assert (temp_dir / "training_metrics.csv").exists()

def test_main_evaluation_mode(sample_data, temp_dir, config_file, mock_model):
    """Test main function in evaluation mode."""
    price_data, features = sample_data
    
    # Save sample data
    price_data_path = temp_dir / "price_data.csv"
    features_path = temp_dir / "features.csv"
    price_data.to_csv(price_data_path, index=False)
    features.to_csv(features_path, index=False)
    
    # Mock command line arguments
    test_args = [
        "main.py",
        "--mode", "evaluate",
        "--config", str(config_file),
        "--price_data", str(price_data_path),
        "--features", str(features_path),
        "--save_dir", str(temp_dir),
        "--model_path", str(mock_model)
    ]
    
    with patch.object(sys, 'argv', test_args), \
         patch('builtins.print') as mock_print:  # Capture print output
        main()
    
    # Check that evaluation metrics were printed
    assert mock_print.call_count > 0

def test_main_missing_model_path(sample_data, temp_dir, config_file):
    """Test main function fails correctly when model path is missing in evaluation mode."""
    price_data, features = sample_data
    
    # Save sample data
    price_data_path = temp_dir / "price_data.csv"
    features_path = temp_dir / "features.csv"
    price_data.to_csv(price_data_path, index=False)
    features.to_csv(features_path, index=False)
    
    # Mock command line arguments without model path
    test_args = [
        "main.py",
        "--mode", "evaluate",
        "--config", str(config_file),
        "--price_data", str(price_data_path),
        "--features", str(features_path),
        "--save_dir", str(temp_dir)
    ]
    
    with patch.object(sys, 'argv', test_args), \
         pytest.raises(ValueError, match="Model path must be provided for evaluation"):
        main()

def test_main_invalid_mode(sample_data, temp_dir, config_file):
    """Test main function fails correctly with invalid mode."""
    price_data, features = sample_data
    
    # Save sample data
    price_data_path = temp_dir / "price_data.csv"
    features_path = temp_dir / "features.csv"
    price_data.to_csv(price_data_path, index=False)
    features.to_csv(features_path, index=False)
    
    # Mock command line arguments with invalid mode
    test_args = [
        "main.py",
        "--mode", "invalid",
        "--config", str(config_file),
        "--price_data", str(price_data_path),
        "--features", str(features_path),
        "--save_dir", str(temp_dir)
    ]
    
    with patch.object(sys, 'argv', test_args), \
         pytest.raises(SystemExit):  # argparse will exit on invalid choice
        main() 