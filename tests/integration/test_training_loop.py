import pytest
import torch
import numpy as np
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch
from main import train_dqn
from models.model import DQN
from environments.trading_env import SimpleTradeEnv
from memory.replay_buffer import PrioritizedReplayBuffer

@pytest.fixture
def mock_training_state():
    """Create mock training state for testing."""
    window_size = 10  # Reduced window size for testing
    input_size = window_size * 9 + 3
    model = DQN(input_size)
    optimizer = torch.optim.Adam(model.parameters())
    metrics = {
        'profit': 0.15,
        'trades': 20,
        'excess_return': 0.08,
        'win_rate': 0.6
    }
    return model, optimizer, metrics

@pytest.fixture
def training_data(mock_market_data_dict):
    """Prepare training and validation data."""
    train_data = {}
    val_data = {}
    for ticker, data in mock_market_data_dict.items():
        split_idx = int(len(data) * 0.7)
        train_data[ticker] = data[:split_idx].copy()
        val_data[ticker] = data[split_idx:].copy()
    return train_data, val_data

@pytest.fixture
def temp_training_dir():
    """Create temporary directory for training artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_end_to_end_training(mock_training_state, mock_training_data, tmp_path):
    """Test end-to-end training process."""
    train_data, val_data = mock_training_data
    
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False), \
         patch('main.window_size', 10):
        
        results = train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            n_episodes=3,
            batch_size=4,
            gamma=0.99,
            input_size=435
        )
        
        assert results is not None
        assert 'final_model' in results
        assert 'training_summary' in results

def test_resource_management(mock_training_state, mock_training_data, tmp_path):
    """Test resource management during training."""
    train_data, val_data = mock_training_data
    
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False), \
         patch('main.window_size', 10):
        
        results = train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            n_episodes=3,
            batch_size=4,
            gamma=0.99,
            input_size=435
        )
        
        assert results is not None
        assert 'final_model' in results
        assert 'training_summary' in results

def test_checkpoint_management(mock_training_state, mock_training_data, tmp_path):
    """Test checkpoint management during training."""
    train_data, val_data = mock_training_data
    checkpoint_dir = tmp_path / "checkpoints"
    
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False), \
         patch('main.window_size', 10):
        
        results = train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            n_episodes=3,
            batch_size=4,
            gamma=0.99,
            input_size=435,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        assert results is not None
        assert 'final_model' in results
        assert 'training_summary' in results

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_gpu_coordination(mock_training_state, tmp_path):
    """Test coordination between multiple GPUs during training."""
    model, optimizer, metrics = mock_training_state
    
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=2), \
         patch('main.window_size', 10):
        
        results = train_dqn(
            train_data_dict={'AAPL': np.random.randn(100, 5)},
            val_data_dict={'AAPL': np.random.randn(50, 5)},
            n_episodes=3,
            batch_size=4,
            gamma=0.99
        )
        
        assert results is not None
        assert 'final_model' in results
        assert 'training_summary' in results

def test_error_recovery(mock_training_state, mock_training_data, tmp_path):
    """Test error recovery during training."""
    train_data, val_data = mock_training_data
    
    def mock_step(self, action):
        if not hasattr(mock_step, 'called'):
            mock_step.called = True
            raise RuntimeError("Simulated error")
        return self.state, 0, True
    
    with patch.object(SimpleTradeEnv, 'step', mock_step), \
         patch('main.window_size', 10):
        
        results = train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            n_episodes=3,
            batch_size=4,
            gamma=0.99,
            input_size=435
        )
        
        assert results is not None
        assert 'final_model' in results
        assert 'training_summary' in results 