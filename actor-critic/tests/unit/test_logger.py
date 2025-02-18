import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
from src.utils.logger import TrainingLogger

@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_config():
    """Create sample training configuration."""
    return {
        "learning_rate": 0.001,
        "batch_size": 256,
        "hidden_dim": 128,
        "gamma": 0.99
    }

@pytest.fixture
def logger(temp_log_dir, sample_config):
    """Create logger instance for testing."""
    return TrainingLogger(
        log_dir=temp_log_dir,
        experiment_name="test_experiment",
        config=sample_config
    )

def test_logger_initialization(logger, temp_log_dir, sample_config):
    """Test logger initialization and directory creation."""
    # Check log directory creation
    assert temp_log_dir.exists()
    assert (temp_log_dir / f"{logger.experiment_name}.log").exists()
    
    # Check config file contents
    config_file = list(temp_log_dir.glob("config_*.json"))[0]
    with open(config_file, 'r') as f:
        saved_config = json.load(f)
    assert saved_config == sample_config
    
    # Check metric storage initialization
    assert all(key in logger.metrics for key in [
        "episode_rewards",
        "portfolio_values",
        "sharpe_ratios",
        "max_drawdowns",
        "positions",
        "returns"
    ])

def test_log_episode(logger):
    """Test episode logging functionality."""
    episode = 1
    metrics = {
        "reward": 1.5,
        "portfolio_value": 1.1,
        "sharpe_ratio": 0.8,
        "max_drawdown": 0.1
    }
    step_metrics = {
        "critic_loss": 0.5,
        "actor_loss": 0.3,
        "alpha": 0.2
    }
    
    logger.log_episode(episode, metrics, step_metrics)
    
    # Check metrics storage
    assert logger.metrics["episode_rewards"][-1] == metrics["reward"]
    assert logger.metrics["portfolio_values"][-1] == metrics["portfolio_value"]
    assert logger.metrics["sharpe_ratios"][-1] == metrics["sharpe_ratio"]
    assert logger.metrics["max_drawdowns"][-1] == metrics["max_drawdown"]

def test_log_training_step(logger):
    """Test training step logging."""
    step = 100
    metrics = {
        "critic_loss": 0.5,
        "actor_loss": 0.3,
        "alpha_loss": 0.1,
        "alpha": 0.2
    }
    
    logger.log_training_step(step, metrics)
    
    # No direct way to check tensorboard, but ensure no errors

def test_log_evaluation(logger):
    """Test evaluation logging."""
    metrics = {
        "final_portfolio_value": 1.2,
        "total_return": 0.2,
        "sharpe_ratio": 1.1,
        "max_drawdown": 0.15,
        "avg_position": 0.5
    }
    
    logger.log_evaluation(metrics, step=1000)
    
    # Check log file contains evaluation results
    with open(logger.log_dir / f"{logger.experiment_name}.log", 'r') as f:
        log_contents = f.read()
        assert "Evaluation Results" in log_contents
        for key, value in metrics.items():
            assert f"{key}: {value:.4f}" in log_contents

def test_save_metrics(logger):
    """Test metrics saving functionality."""
    # Add some test metrics
    logger.metrics["episode_rewards"] = [1.0, 1.5, 2.0]
    logger.metrics["portfolio_values"] = [1.0, 1.1, 1.2]
    logger.metrics["sharpe_ratios"] = [0.5, 0.7, 0.9]
    logger.metrics["max_drawdowns"] = [0.1, 0.15, 0.12]
    logger.metrics["positions"] = [0.5, 0.6, 0.7]
    
    logger.save_metrics()
    
    # Check metrics CSV file
    metrics_df = pd.read_csv(logger.log_dir / "training_metrics.csv")
    assert len(metrics_df) == 3
    assert all(col in metrics_df.columns for col in [
        "episode_rewards",
        "portfolio_values",
        "sharpe_ratios",
        "max_drawdowns",
        "positions"
    ])
    
    # Check summary JSON file
    with open(logger.log_dir / "summary.json", 'r') as f:
        summary = json.load(f)
        assert all(key in summary for key in [
            "final_portfolio_value",
            "mean_reward",
            "final_sharpe",
            "max_drawdown",
            "mean_position"
        ])

def test_logger_close(logger):
    """Test logger cleanup on close."""
    # Add some test metrics
    logger.metrics["episode_rewards"] = [1.0, 1.5, 2.0]
    
    logger.close()
    
    # Check that metrics were saved
    assert (logger.log_dir / "training_metrics.csv").exists()
    assert (logger.log_dir / "summary.json").exists()

def test_logger_error_handling(temp_log_dir):
    """Test logger error handling."""
    # Test with invalid log directory
    with pytest.raises(Exception):
        TrainingLogger(log_dir="/nonexistent/path")
    
    # Test with invalid metrics
    logger = TrainingLogger(log_dir=temp_log_dir)
    with pytest.raises(Exception):
        logger.log_episode(1, {"invalid_metric": 1.0}) 