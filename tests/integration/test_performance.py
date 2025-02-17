import pytest
import torch
import numpy as np
import time
import psutil
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
def performance_data(mock_market_data_dict):
    """Prepare larger dataset for performance testing."""
    # Create expanded dataset by repeating data
    expanded_data = {}
    for ticker, data in mock_market_data_dict.items():
        expanded_data[ticker] = data.copy()
        for i in range(3):  # Create 3 copies with slight modifications
            modified_data = data.copy()
            modified_data['Close'] *= (1 + np.random.randn(len(data)) * 0.01)
            expanded_data[f"{ticker}_{i}"] = modified_data
    
    # Split into train/val
    train_data = {}
    val_data = {}
    for ticker, data in expanded_data.items():
        split_idx = int(len(data) * 0.7)
        train_data[ticker] = data[:split_idx].copy()
        val_data[ticker] = data[split_idx:].copy()
    
    return train_data, val_data

@pytest.fixture
def temp_perf_dir():
    """Create temporary directory for performance testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_training_dir():
    """Create temporary directory for training."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

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

def measure_memory_usage():
    """Helper function to measure current memory usage."""
    process = psutil.Process()
    return process.memory_info().rss

def measure_gpu_memory():
    """Helper function to measure GPU memory usage if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have memory tracking API yet
        return 0
    return 0

def test_training_speed(training_data, temp_training_dir):
    """Test training speed and performance."""
    train_data, val_data = training_data
    
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False), \
         patch('main.window_size', 10):
        
        start_time = time.time()
        results = train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            n_episodes=3,
            batch_size=4,
            gamma=0.99,
            input_size=93
        )
        training_time = time.time() - start_time
        
        assert results is not None
        assert 'final_model' in results
        assert training_time < 60  # Training should complete within 60 seconds

def test_memory_efficiency(performance_data, temp_perf_dir):
    """Test memory usage during training."""
    train_data, val_data = performance_data
    
    # Measure baseline memory
    initial_memory = measure_memory_usage()
    initial_gpu_memory = measure_gpu_memory()
    
    # Run training with memory tracking
    memory_samples = []
    gpu_memory_samples = []
    
    def memory_callback():
        memory_samples.append(measure_memory_usage())
        gpu_memory_samples.append(measure_gpu_memory())
    
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        
        train_dqn(
            train_data_dict=train_data,
            val_data_dict=val_data,
            input_size=435,
            n_episodes=2,
            batch_size=4,
            gamma=0.99
        )
        memory_callback()
    
    # Calculate memory statistics
    max_memory_increase = max(0, max(memory_samples) - initial_memory)
    avg_memory_increase = np.mean([m - initial_memory for m in memory_samples])
    
    # Memory usage assertions
    assert max_memory_increase < 2e9, "Memory usage exceeded 2GB"
    assert avg_memory_increase < 1e9, "Average memory usage exceeded 1GB"

def test_batch_processing_speed():
    """Test batch processing performance."""
    # Create synthetic batch data
    batch_sizes = [4, 8, 16, 32]
    sequence_length = 10
    feature_size = 435
    
    results = {}
    for batch_size in batch_sizes:
        batch = {
            'states': torch.randn(batch_size, sequence_length, feature_size),
            'actions': torch.randint(0, 3, (batch_size, sequence_length)),
            'rewards': torch.randn(batch_size, sequence_length),
            'next_states': torch.randn(batch_size, sequence_length, feature_size),
            'dones': torch.zeros(batch_size, sequence_length),
            'weights': torch.ones(batch_size),
            'indices': list(range(batch_size))
        }
        
        model = DQN(feature_size)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Time batch processing
        start_time = time.time()
        for _ in range(10):  # Run multiple times for better measurement
            states = batch['states'].reshape(-1, feature_size)
            q_values, _ = model(states)
            loss = torch.mean(q_values)  # Dummy loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        duration = (time.time() - start_time) / 10  # Average time per batch
        results[batch_size] = duration
    
    # Log results
    for batch_size, duration in results.items():
        print(f"Batch Size {batch_size}: {duration:.4f} seconds per batch")
    
    # Performance assertions
    assert all(results[b] < 1.0 for b in batch_sizes), "Batch processing too slow"

def test_multi_gpu_scaling():
    """Test training scaling with multiple GPUs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    train_data, val_data = performance_data
    
    # Test with different numbers of GPUs
    gpu_counts = [1, 2] if torch.cuda.device_count() >= 2 else [1]
    results = {}
    
    for gpu_count in gpu_counts:
        with patch('torch.cuda.device_count', return_value=gpu_count):
            start_time = time.time()
            
            train_dqn(
                train_data_dict=train_data,
                val_data_dict=val_data,
                input_size=435,
                n_episodes=2,
                batch_size=4 * gpu_count,  # Scale batch size with GPUs
                gamma=0.99
            )
            
            duration = time.time() - start_time
            results[gpu_count] = duration
    
    # Log results
    for gpu_count, duration in results.items():
        print(f"{gpu_count} GPU(s): {duration:.2f} seconds")
    
    # Scaling assertions
    if len(results) > 1:
        speedup = results[1] / results[2]  # Speedup from 1 to 2 GPUs
        assert speedup > 1.2, "Multi-GPU scaling less than 20% improvement"

def test_data_pipeline_efficiency(performance_data):
    """Test efficiency of data loading and preprocessing pipeline."""
    train_data, val_data = performance_data
    
    # Measure data loading time
    start_time = time.time()
    env = SimpleTradeEnv(next(iter(train_data.values())))
    data_load_time = time.time() - start_time
    
    # Measure state preprocessing time
    start_time = time.time()
    state = env.reset()
    preprocess_time = time.time() - start_time
    
    # Measure batch preparation time
    memory = PrioritizedReplayBuffer(1000)
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(env.get_valid_actions())
            next_state, reward, done = env.step(action)
            memory.push_episode([(state, action, reward, next_state, done)])
            state = next_state
    
    start_time = time.time()
    batch = memory.sample(32, seq_len=10, device='cpu')
    batch_prep_time = time.time() - start_time
    
    # Log timing results
    print(f"Data Loading Time: {data_load_time:.4f} seconds")
    print(f"State Preprocessing Time: {preprocess_time:.4f} seconds")
    print(f"Batch Preparation Time: {batch_prep_time:.4f} seconds")
    
    # Performance assertions
    assert data_load_time < 1.0, "Data loading too slow"
    assert preprocess_time < 0.1, "State preprocessing too slow"
    assert batch_prep_time < 0.1, "Batch preparation too slow"

def test_device_transfer_overhead():
    """Test overhead of transferring data between devices."""
    if not (torch.cuda.is_available() or 
            (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())):
        pytest.skip("No hardware acceleration available")
    
    # Create test data
    data_sizes = [(100, 435), (1000, 435), (10000, 435)]
    results = {}
    
    for size in data_sizes:
        data = torch.randn(*size)
        
        # Test CUDA transfer if available
        if torch.cuda.is_available():
            start_time = time.time()
            data_cuda = data.cuda()
            cuda_to_cpu = data_cuda.cpu()
            cuda_time = time.time() - start_time
            results[f'cuda_{size}'] = cuda_time
        
        # Test MPS transfer if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            start_time = time.time()
            data_mps = data.to('mps')
            mps_to_cpu = data_mps.cpu()
            mps_time = time.time() - start_time
            results[f'mps_{size}'] = mps_time
    
    # Log results
    for config, duration in results.items():
        print(f"{config}: {duration:.4f} seconds")
    
    # Performance assertions
    for size in data_sizes:
        if f'cuda_{size}' in results:
            assert results[f'cuda_{size}'] < 0.1, f"CUDA transfer too slow for size {size}"
        if f'mps_{size}' in results:
            assert results[f'mps_{size}'] < 0.1, f"MPS transfer too slow for size {size}" 