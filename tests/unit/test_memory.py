import pytest
import numpy as np
import torch
from memory.replay_buffer import PrioritizedReplayBuffer

@pytest.fixture
def buffer():
    """Create a small replay buffer for testing."""
    return PrioritizedReplayBuffer(capacity=10, alpha=0.6, beta=0.4)

@pytest.fixture
def sample_episode():
    """Create a sample episode for testing."""
    # Each transition: (state, action, reward, next_state, done)
    return [
        (np.array([1.0, 0.0]), 0, 0.1, np.array([1.1, 0.0]), False),
        (np.array([1.1, 0.0]), 1, 0.2, np.array([1.2, 0.0]), False),
        (np.array([1.2, 0.0]), 0, 0.3, np.array([1.3, 0.0]), False),
        (np.array([1.3, 0.0]), 2, 0.4, np.array([1.4, 0.0]), True)
    ]

@pytest.fixture
def device():
    """Use CPU for testing."""
    return torch.device('cpu')

def test_buffer_initialization(buffer):
    """Test buffer initialization."""
    assert buffer.capacity == 10
    assert buffer.alpha == 0.6
    assert buffer.beta == 0.4
    assert len(buffer) == 0
    assert buffer.episodes == []
    assert buffer.priorities == []

def test_push_episode(buffer, sample_episode):
    """Test pushing episodes to the buffer."""
    buffer.push_episode(sample_episode)
    assert len(buffer) == 1
    assert len(buffer.episodes) == 1
    assert len(buffer.priorities) == 1
    assert buffer.priorities[0] > 0

def test_buffer_capacity(buffer, sample_episode):
    """Test that buffer respects capacity limits."""
    # Fill buffer beyond capacity
    for i in range(15):
        modified_episode = [(s * (i+1), a, r * (i+1), ns * (i+1), d) 
                          for s, a, r, ns, d in sample_episode]
        buffer.push_episode(modified_episode)
    
    assert len(buffer) == buffer.capacity
    assert len(buffer.episodes) == buffer.capacity
    assert len(buffer.priorities) == buffer.capacity

def test_sample_batch(buffer, sample_episode, device):
    """Test batch sampling functionality."""
    # Add multiple episodes
    for i in range(5):
        modified_episode = [(s * (i+1), a, r * (i+1), ns * (i+1), d) 
                          for s, a, r, ns, d in sample_episode]
        buffer.push_episode(modified_episode)
    
    batch = buffer.sample(batch_size=3, seq_len=2, device=device)
    
    assert batch is not None
    assert isinstance(batch, dict)
    assert all(k in batch for k in ['states', 'actions', 'rewards', 
                                   'next_states', 'dones', 'weights', 'indices'])
    assert batch['states'].shape == (3, 2, 2)  # (batch_size, seq_len, state_dim)
    assert batch['actions'].shape == (3, 2)    # (batch_size, seq_len)
    assert batch['rewards'].shape == (3, 2)    # (batch_size, seq_len)
    assert batch['weights'].shape == (3,)      # (batch_size,)
    assert len(batch['indices']) == 3          # [batch_size]

def test_sample_with_insufficient_data(buffer, device):
    """Test sampling behavior with insufficient data."""
    # Empty buffer
    batch = buffer.sample(batch_size=3, seq_len=2, device=device)
    assert batch is None
    
    # Single episode shorter than sequence length
    short_episode = [(np.array([1.0, 0.0]), 0, 0.1, np.array([1.1, 0.0]), True)]
    buffer.push_episode(short_episode)
    batch = buffer.sample(batch_size=1, seq_len=2, device=device)
    assert batch is None

def test_priority_updates(buffer, sample_episode):
    """Test priority updating functionality."""
    buffer.push_episode(sample_episode)
    initial_priority = buffer.priorities[0]
    
    # Update priority
    buffer.update_priorities([0], [1.0])
    assert buffer.priorities[0] != initial_priority
    assert buffer.priorities[0] == (1.0 + buffer.eps) ** buffer.alpha

def test_priority_sampling(buffer, sample_episode):
    """Test that sampling respects priorities."""
    device = torch.device('cpu')  # Use CPU directly for this test
    
    # Add episodes with different rewards
    for i in range(5):
        modified_episode = [(s, a, r * (10 ** i), ns, d) 
                          for s, a, r, ns, d in sample_episode]
        buffer.push_episode(modified_episode)
    
    # Sample multiple times and check if higher priority episodes are sampled more often
    sample_counts = np.zeros(5)
    n_samples = 100
    
    for _ in range(n_samples):
        batch = buffer.sample(batch_size=1, seq_len=2, device=device)
        if batch is not None:
            sample_counts[batch['indices'][0]] += 1
    
    # Higher priority episodes should be sampled more frequently
    # Check if there's at least some variation in sampling frequency
    assert not np.allclose(sample_counts, sample_counts.mean())

def test_beta_annealing(buffer, sample_episode, device):
    """Test that beta parameter increases over time."""
    buffer.push_episode(sample_episode)
    initial_beta = buffer.beta
    
    # Sample multiple times
    for _ in range(100):
        buffer.sample(batch_size=1, seq_len=2, device=device)
    
    assert buffer.beta > initial_beta
    assert buffer.beta <= 1.0

def test_sequence_sampling(buffer, sample_episode, device):
    """Test that sampled sequences are contiguous and valid."""
    buffer.push_episode(sample_episode)
    batch = buffer.sample(batch_size=1, seq_len=2, device=device)
    
    if batch is not None:
        # Check that states are consecutive
        states = batch['states'][0].numpy()
        for i in range(len(states) - 1):
            assert np.allclose(states[i+1][0], states[i][0] + 0.1)

def test_nan_error_handling(buffer, sample_episode):
    """Test handling of NaN errors in priority updates."""
    buffer.push_episode(sample_episode)
    initial_priority = buffer.priorities[0]
    
    # Update with NaN error
    buffer.update_priorities([0], [float('nan')])
    # Priority should remain unchanged
    assert buffer.priorities[0] == initial_priority 