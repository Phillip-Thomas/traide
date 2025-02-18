import torch
import numpy as np
import pytest
from src.utils.replay_buffer import ReplayBuffer

@pytest.fixture
def replay_buffer():
    return ReplayBuffer(state_dim=10, action_dim=3, max_size=100)

def test_replay_buffer_initialization(replay_buffer):
    """Test that the buffer initializes correctly."""
    assert replay_buffer.max_size == 100
    assert replay_buffer.states.shape == (100, 10)
    assert replay_buffer.actions.shape == (100, 3)
    assert replay_buffer.rewards.shape == (100,)
    assert replay_buffer.next_states.shape == (100, 10)
    assert replay_buffer.dones.shape == (100,)
    assert replay_buffer.ptr == 0
    assert replay_buffer.size == 0

def test_replay_buffer_add(replay_buffer):
    """Test adding experiences to the buffer."""
    state = np.random.randn(10)
    action = np.random.randn(3)
    reward = np.random.randn()
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

def test_replay_buffer_wraparound(replay_buffer):
    """Test buffer wraparound behavior."""
    # Fill buffer beyond capacity
    for _ in range(150):  # Buffer size is 100
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = np.random.randn()
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
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        replay_buffer.add(state, action, reward, next_state, done)
    
    # Sample batch
    batch_size = 32
    batch = replay_buffer.sample(batch_size)
    
    # Check batch properties
    assert isinstance(batch, dict)
    assert all(key in batch for key in ["states", "actions", "rewards", "next_states", "dones"])
    assert batch["states"].shape == (batch_size, 10)
    assert batch["actions"].shape == (batch_size, 3)
    assert batch["rewards"].shape == (batch_size,)
    assert batch["next_states"].shape == (batch_size, 10)
    assert batch["dones"].shape == (batch_size,)
    
    # Check device and type
    assert isinstance(batch["states"], torch.Tensor)
    assert batch["states"].dtype == torch.float32

def test_replay_buffer_empty_sample():
    """Test sampling from empty buffer raises error."""
    buffer = ReplayBuffer(state_dim=10, action_dim=3, max_size=100)
    
    with pytest.raises(ValueError):
        buffer.sample(32) 