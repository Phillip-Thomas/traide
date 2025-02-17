import pytest
import torch
import os
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from models.model import DQN
from utils.save_utils import save_experiment, save_last_checkpoint, load_checkpoint
from main import train_dqn

@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoint testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

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

def test_checkpoint_creation(mock_training_state, temp_checkpoint_dir):
    """Test creation of checkpoint files."""
    model, optimizer, metrics = mock_training_state
    checkpoint_dir = Path(temp_checkpoint_dir)
    
    # Save experiment checkpoint
    checkpoint_path = save_experiment(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_dir=str(checkpoint_dir),
        avg_loss=0.5
    )
    
    # Verify checkpoint files
    assert os.path.exists(checkpoint_path)
    assert os.path.exists(checkpoint_dir / "metrics.json")
    
    # Check checkpoint content
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'metrics' in checkpoint
    assert checkpoint['metrics']['profit'] == 0.15

def test_checkpoint_loading(mock_training_state, temp_checkpoint_dir):
    """Test loading from checkpoints."""
    model, optimizer, metrics = mock_training_state
    checkpoint_dir = Path(temp_checkpoint_dir)
    
    # Save checkpoint
    checkpoint_path = save_experiment(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_dir=str(checkpoint_dir),
        avg_loss=0.5
    )
    
    # Load checkpoint
    device = torch.device('cpu')
    loaded_checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Create new model and load state
    window_size = 10  # Reduced window size for testing
    input_size = window_size * 9 + 3
    new_model = DQN(input_size)
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    # Compare model parameters
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1.data, p2.data)

def test_checkpoint_versioning(mock_training_state, temp_checkpoint_dir):
    """Test checkpoint version management."""
    model, optimizer, metrics = mock_training_state
    checkpoint_dir = Path(temp_checkpoint_dir)
    
    # Save multiple versions
    metrics_versions = [
        {'profit': 0.1, 'trades': 10},
        {'profit': 0.15, 'trades': 15},
        {'profit': 0.12, 'trades': 12}
    ]
    
    for i, metrics in enumerate(metrics_versions):
        checkpoint_path = save_experiment(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            checkpoint_dir=str(checkpoint_dir),
            avg_loss=0.5
        )
        
        # Also save numbered versions
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_dir / f"checkpoint_{i}.pt")
    
    # Verify all versions exist
    assert os.path.exists(checkpoint_path)
    for i in range(len(metrics_versions)):
        assert os.path.exists(checkpoint_dir / f"checkpoint_{i}.pt")

def test_checkpoint_recovery(training_data, temp_checkpoint_dir):
    """Test recovery from interrupted training using checkpoints."""
    train_data, val_data = training_data
    window_size = 48  # Use standard window size
    input_size = window_size * 9 + 3  # Match the environment's state size
    checkpoint_dir = Path(temp_checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create initial checkpoint
    model = DQN(input_size)
    optimizer = torch.optim.Adam(model.parameters())
    metrics = {'profit': 0.1}
    
    # Save initial checkpoint
    save_last_checkpoint(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Verify last checkpoint exists
    assert os.path.exists(checkpoint_dir / "LAST_checkpoint.pt")
    
    # Run training with checkpoint
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False), \
         patch('main.window_size', window_size):  # Patch window size in main module
        
        results = train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            n_episodes=5,
            batch_size=4,
            gamma=0.99,
            checkpoint_dir=str(checkpoint_dir),
            input_size=input_size
        )
        
        assert results is not None
        assert results['final_model'] is not None
        assert isinstance(results['final_model'], DQN)

def test_checkpoint_compatibility(mock_training_state, temp_checkpoint_dir):
    """Test checkpoint compatibility across different devices."""
    model, optimizer, metrics = mock_training_state
    checkpoint_dir = Path(temp_checkpoint_dir)
    
    # Save on CPU
    save_experiment(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Mock different devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    
    # Load on each device
    window_size = 10  # Reduced window size for testing
    input_size = window_size * 9 + 3
    
    for device_type in devices:
        device = torch.device(device_type)
        loaded_checkpoint = load_checkpoint(
            checkpoint_path=str(checkpoint_dir / "best_model.pt"),
            device=device
        )
        
        new_model = DQN(input_size).to(device)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        # Verify model works on device
        test_input = torch.randn(1, input_size).to(device)
        with torch.no_grad():
            output, _ = new_model(test_input)
            # Check device type only, not index
            assert output.device.type == device.type

def test_checkpoint_metrics_tracking(mock_training_state, temp_checkpoint_dir):
    """Test tracking of metrics in checkpoints."""
    model, optimizer, metrics = mock_training_state
    checkpoint_dir = Path(temp_checkpoint_dir)
    
    # Save checkpoints with different metrics
    metrics_history = []
    for i in range(3):
        current_metrics = {
            'profit': 0.1 + i * 0.05,
            'trades': 10 + i * 5,
            'excess_return': 0.05 + i * 0.02
        }
        metrics_history.append(current_metrics)
        
        checkpoint_path = save_experiment(
            model=model,
            optimizer=optimizer,
            metrics=current_metrics,
            checkpoint_dir=str(checkpoint_dir),
            avg_loss=0.5
        )
        
        # Save metrics separately
        with open(checkpoint_dir / f"metrics_{i}.json", 'w') as f:
            json.dump(current_metrics, f)
    
    # Verify metrics history
    for i, expected_metrics in enumerate(metrics_history):
        with open(checkpoint_dir / f"metrics_{i}.json", 'r') as f:
            saved_metrics = json.load(f)
        assert saved_metrics == expected_metrics
    
    # Verify best model has best metrics
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=True)
    assert best_checkpoint['metrics']['profit'] == max(m['profit'] for m in metrics_history) 