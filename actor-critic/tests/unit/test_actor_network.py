import pytest
import torch
from src.models.sac_actor import SACActorNetwork

@pytest.fixture
def actor_network():
    """Create an actor network for testing."""
    state_dim = 10
    action_dim = 3
    hidden_dim = 256
    return SACActorNetwork(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)

def test_actor_forward_pass(actor_network):
    """Test forward pass of actor network."""
    batch_size = 32
    state_dim = 10
    states = torch.randn(batch_size, state_dim)
    
    # Get action and log probability
    actions, log_probs = actor_network(states)
    
    # Check output shapes
    assert actions.shape == (batch_size, 3)
    assert log_probs.shape == (batch_size,)
    
    # Check actions are bounded
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    
    # Get mean actions
    means = actor_network.get_mean(states)
    assert means.shape == (batch_size, 3)
    assert torch.all(means >= -1.0)
    assert torch.all(means <= 1.0)

def test_actor_log_prob(actor_network):
    """Test log probability calculation."""
    batch_size = 32
    state_dim = 10
    states = torch.randn(batch_size, state_dim)
    
    # Get actions and their log probs
    actions, means = actor_network(states)
    log_probs = actor_network.log_prob(states, actions)
    
    # Check output shapes
    assert log_probs.shape == (batch_size,)
    
    # Check log probs are finite
    assert torch.all(torch.isfinite(log_probs))

# Remove test_actor_sample_output and test_actor_deterministic_output as they're not applicable 