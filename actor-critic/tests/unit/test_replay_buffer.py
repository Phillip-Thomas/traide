import torch
import numpy as np
import pytest
from src.utils.replay_buffer import ReplayBuffer

@pytest.fixture
def replay_buffer():
    return ReplayBuffer(state_dim=10, buffer_size=100)

def test_replay_buffer_initialization(replay_buffer):
    """Test that the buffer initializes correctly."""
    assert replay_buffer.buffer_size == 100
    assert replay_buffer.states.shape == (100, 10)
    assert replay_buffer.actions.shape == (100, 1)
    assert replay_buffer.rewards.shape == (100, 1)
    assert replay_buffer.next_states.shape == (100, 10)
    assert replay_buffer.dones.shape == (100, 1)
    assert replay_buffer.priorities.shape == (100,)
    assert replay_buffer.ptr == 0
    assert replay_buffer.size == 0

def test_replay_buffer_add(replay_buffer):
    """Test adding experiences to the buffer."""
    state = np.random.randn(10)
    action = np.random.randn(1)
    reward = np.random.randn(1)
    next_state = np.random.randn(10)
    done = False
    
    # Add experience
    replay_buffer.add(state, action, reward, next_state, done)
    
    # Check buffer state
    assert replay_buffer.ptr == 1
    assert replay_buffer.size == 1
    assert np.allclose(replay_buffer.states[0], state)
    assert np.allclose(replay_buffer.actions[0], action)
    assert np.allclose(replay_buffer.rewards[0], reward)
    assert np.allclose(replay_buffer.next_states[0], next_state)
    assert np.allclose(replay_buffer.dones[0], done)
    assert replay_buffer.priorities[0] == replay_buffer.max_priority

def test_replay_buffer_wraparound(replay_buffer):
    """Test buffer wraparound behavior."""
    # Fill buffer beyond capacity
    for _ in range(150):  # Buffer size is 100
        state = np.random.randn(10)
        action = np.random.randn(1)
        reward = np.random.randn(1)
        next_state = np.random.randn(10)
        done = False
        replay_buffer.add(state, action, reward, next_state, done)
    
    assert replay_buffer.ptr == 50  # Wrapped around to 50
    assert replay_buffer.size == 100  # Maxed at buffer size

def test_replay_buffer_sample(replay_buffer):
    """Test sampling from the buffer."""
    # Fill buffer with some experiences
    for _ in range(50):
        state = np.random.randn(10)
        action = np.random.randn(1)
        reward = np.random.randn(1)
        next_state = np.random.randn(10)
        done = False
        replay_buffer.add(state, action, reward, next_state, done)
    
    # Sample batch
    batch_size = 32
    batch = replay_buffer.sample(batch_size)
    
    # Check batch properties
    assert isinstance(batch, dict)
    assert all(key in batch for key in ["states", "actions", "rewards", "next_states", "dones", "weights", "indices"])
    assert batch["states"].shape == (batch_size, 10)
    assert batch["actions"].shape == (batch_size, 1)
    assert batch["rewards"].shape == (batch_size, 1)
    assert batch["next_states"].shape == (batch_size, 10)
    assert batch["dones"].shape == (batch_size, 1)
    assert batch["weights"].shape == (batch_size,)
    assert len(batch["indices"]) == batch_size
    
    # Check device and type
    assert batch["states"].device.type == replay_buffer.device.type  # Compare only device type
    assert batch["states"].dtype == torch.float32

def test_replay_buffer_priority_update(replay_buffer):
    """Test priority updates."""
    # Fill buffer with known values
    for i in range(10):
        state = np.zeros(10, dtype=np.float32)
        action = np.array([0.0], dtype=np.float32)
        reward = np.array([0.0], dtype=np.float32)
        next_state = np.zeros(10, dtype=np.float32)
        done = False
        replay_buffer.add(state, action, reward, next_state, done)
    
    # Set known indices and priorities
    indices = np.array([0, 2, 4], dtype=np.int64)
    new_priorities = np.array([0.5, 1.0, 0.7], dtype=np.float32)
    
    # Update priorities
    replay_buffer.update_priorities(indices, new_priorities)
    
    # Check updated priorities
    for i, idx in enumerate(indices):
        expected = new_priorities[i] + 1e-6
        actual = replay_buffer.priorities[idx]
        assert abs(actual - expected) < 1e-5, f"Priority mismatch at index {idx}: expected {expected}, got {actual}"
    
    # Check max priority
    expected_max = max(new_priorities.max() + 1e-6, 1.0)  # 1.0 is initial priority
    assert abs(replay_buffer.max_priority - expected_max) < 1e-5

def test_replay_buffer_empty_sample():
    """Test sampling from empty buffer raises error."""
    buffer = ReplayBuffer(state_dim=10, buffer_size=100)
    
    with pytest.raises(ValueError):
        buffer.sample(32) 