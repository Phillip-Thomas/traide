import pytest
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from utils.logger import TrainingLogger
from utils.visualizer import TrainingVisualizer
from utils.save_utils import save_experiment, save_last_checkpoint, load_checkpoint

@pytest.fixture
def mock_metrics_data():
    """Create mock training metrics data."""
    return {
        'episodes': list(range(10)),
        'returns': [float(i) for i in range(10)],
        'lengths': [100 + i for i in range(10)],
        'losses': [0.5 - i*0.05 for i in range(10)],
        'priorities': [1.0 - i*0.1 for i in range(10)],
        'validations': [
            {'episode': i, 'profit': i*0.1, 'trades': i*2}
            for i in range(0, 10, 2)
        ]
    }

@pytest.fixture
def logger(tmp_path):
    """Create a TrainingLogger instance with temporary directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return TrainingLogger(log_dir=str(log_dir))

@pytest.fixture
def visualizer(tmp_path, mock_metrics_data):
    """Create a TrainingVisualizer instance with mock data."""
    metrics_file = tmp_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(mock_metrics_data, f)
    return TrainingVisualizer(metrics_file=str(metrics_file))

def test_logger_initialization(logger, tmp_path):
    """Test logger initialization and directory creation."""
    assert os.path.exists(tmp_path / "logs")
    assert os.path.exists(tmp_path / "logs" / "metrics.json")
    assert hasattr(logger, 'metrics_file')

def test_log_episode(logger):
    """Test episode logging functionality."""
    logger.log_episode(
        episode_num=1,
        returns=1.0,
        length=100,
        std_dev=0.5,
        priority=1.0,
        epsilon=0.9,
        loss=0.5,
        trades_info=[1, -1, 1]
    )
    
    with open(logger.metrics_file, 'r') as f:
        metrics = json.load(f)
    
    assert 1 in metrics['episodes']
    assert 1.0 in metrics['returns']
    assert 100 in metrics['lengths']
    assert 0.5 in metrics['losses']
    assert 1.0 in metrics['priorities']

def test_log_validation(logger):
    """Test validation logging functionality."""
    metrics = {
        'profit': 0.1,
        'win_rate': 0.6,
        'num_trades': 10,
        'excess_return': 0.05
    }
    logger.log_validation(episode=1, metrics=metrics)
    
    with open(logger.metrics_file, 'r') as f:
        saved_metrics = json.load(f)
    
    assert len(saved_metrics['validations']) > 0
    last_validation = saved_metrics['validations'][-1]
    assert last_validation['episode'] == 1
    assert last_validation['profit'] == 0.1

def test_log_model_save(logger, tmp_path):
    """Test model save logging functionality."""
    checkpoint_path = tmp_path / "model.pt"
    metrics = {'profit': 0.1, 'trades': 10}
    logger.log_model_save(episode=1, path=str(checkpoint_path), metrics=metrics)
    
    with open(logger.metrics_file, 'r') as f:
        saved_metrics = json.load(f)
    
    assert len(saved_metrics['model_saves']) > 0
    last_save = saved_metrics['model_saves'][-1]
    assert last_save['episode'] == 1
    assert last_save['path'] == str(checkpoint_path)

def test_visualizer_initialization(visualizer):
    """Test visualizer initialization with metrics data."""
    assert hasattr(visualizer, 'metrics')
    assert 'episodes' in visualizer.metrics
    assert 'returns' in visualizer.metrics

def test_plot_training_progress(visualizer, tmp_path):
    """Test training progress plotting."""
    with patch('matplotlib.pyplot.show'):
        fig = visualizer.plot_training_progress()
        assert isinstance(fig, plt.Figure)
        
        # Save plot for visual inspection
        plot_path = tmp_path / "training_progress.png"
        fig.savefig(plot_path)
        assert os.path.exists(plot_path)

def test_plot_trade_distribution(tmp_path):
    """Test trade distribution plotting functionality."""
    # Create a metrics file with sample data
    metrics_file = tmp_path / "metrics.json"
    metrics_data = {
        'validations': [
            {
                'trades': [
                    {'ticker': 'AAPL', 'profit': 0.05, 'type': 'long'},
                    {'ticker': 'AAPL', 'profit': -0.02, 'type': 'long'},
                    {'ticker': 'AAPL', 'profit': 0.03, 'type': 'short'}
                ]
            },
            {
                'trades': [
                    {'ticker': 'GOOGL', 'profit': 0.04, 'type': 'long'},
                    {'ticker': 'GOOGL', 'profit': 0.01, 'type': 'short'}
                ]
            }
        ]
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f)
    
    # Initialize visualizer with metrics file
    visualizer = TrainingVisualizer(metrics_file=str(metrics_file))
    
    # Set up output path
    output_path = tmp_path / "trade_dist.png"
    
    # Plot and save
    fig = visualizer.plot_trade_distribution()
    if fig is not None:
        fig.savefig(output_path)
        plt.close(fig)
    
    # Verify the plot was created
    assert os.path.exists(output_path)

def test_generate_summary_report(visualizer, tmp_path):
    """Test summary report generation."""
    report_path = tmp_path / "report.txt"
    visualizer.generate_summary_report(output_file=str(report_path))
    
    assert os.path.exists(report_path)
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    assert "Training Summary" in report_content
    assert "Total Episodes" in report_content
    assert "Average Return" in report_content

def test_save_experiment(tmp_path):
    """Test experiment saving functionality."""
    # Create mock model and optimizer
    model = MagicMock()
    model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
    
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"state": {"step": 0}}
    optimizer.param_groups = [{"lr": 0.001}]
    
    metrics = {
        'profit': 0.1,
        'trades': 10,
        'excess_return': 0.05,
        'win_rate': 0.6
    }
    
    checkpoint_dir = tmp_path / "checkpoints"
    
    # Test saving
    checkpoint_path = save_experiment(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_dir=str(checkpoint_dir),
        avg_loss=0.5
    )
    
    assert os.path.exists(checkpoint_path)
    assert os.path.exists(checkpoint_dir / "metrics.json")
    
    # Test loading
    loaded_checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device('cpu')
    )
    
    assert 'model_state_dict' in loaded_checkpoint
    assert 'optimizer_state_dict' in loaded_checkpoint
    assert 'metrics' in loaded_checkpoint
    assert loaded_checkpoint['metrics']['profit'] == 0.1

def test_save_last_checkpoint(tmp_path):
    """Test last checkpoint saving functionality."""
    model = MagicMock()
    model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
    
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"state": {"step": 0}}
    
    metrics = {
        'profit': 0.1,
        'trades': 10,
        'win_rate': 0.6
    }
    
    checkpoint_dir = tmp_path / "checkpoints"
    
    # Test saving
    checkpoint_path = save_last_checkpoint(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    assert os.path.exists(checkpoint_path)
    
    # Test loading
    loaded_checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device('cpu')
    )
    
    assert 'model_state_dict' in loaded_checkpoint
    assert 'optimizer_state_dict' in loaded_checkpoint
    assert 'metrics' in loaded_checkpoint
    assert loaded_checkpoint['metrics']['profit'] == 0.1

def test_device_selection():
    """Test device selection logic."""
    from utils.device import get_device, is_mps_available, is_cuda_available
    
    # Test device selection priority
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'mps', 'cpu']
    
    # Test MPS availability check
    mps_available = is_mps_available()
    assert isinstance(mps_available, bool)
    
    # Test CUDA availability check
    cuda_available = is_cuda_available()
    assert isinstance(cuda_available, bool)

def test_file_system_operations(tmp_path):
    """Test file system utility functions."""
    from utils.fs import ensure_dir, safe_save, safe_load
    
    # Test directory creation
    test_dir = tmp_path / "test_dir"
    ensure_dir(test_dir)
    assert os.path.exists(test_dir)
    
    # Test safe file saving
    data = {"test": "data"}
    file_path = test_dir / "test.json"
    safe_save(data, file_path)
    assert os.path.exists(file_path)
    
    # Test safe file loading
    loaded_data = safe_load(file_path)
    assert loaded_data == data
    
    # Test error handling for non-existent file
    non_existent = test_dir / "non_existent.json"
    loaded_data = safe_load(non_existent)
    assert loaded_data is None 