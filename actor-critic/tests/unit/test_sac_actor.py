import torch
import pytest
import numpy as np
from src.models.sac_actor import SACActorNetwork
import torch.nn as nn

@pytest.fixture
def actor_network():
    # Set all random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return SACActorNetwork(state_dim=10, action_dim=3, hidden_dim=64)

def test_actor_network_initialization(actor_network):
    """Test that the network initializes with correct dimensions."""
    assert isinstance(actor_network, SACActorNetwork)
    
    # Check network architecture
    feature_net = actor_network.feature_net
    assert isinstance(feature_net[0], nn.Linear)
    assert isinstance(feature_net[1], nn.LayerNorm)  # Default is LayerNorm
    assert isinstance(feature_net[2], nn.ReLU)
    assert feature_net[0].in_features == 10  # state_dim
    assert feature_net[0].out_features == 64  # hidden_dim
    
    # Check mean and log_std networks
    assert isinstance(actor_network.mean_net, nn.Linear)
    assert isinstance(actor_network.log_std_net, nn.Linear)
    assert actor_network.mean_net.out_features == 3  # action_dim
    assert actor_network.log_std_net.out_features == 3  # action_dim

def test_actor_forward_pass(actor_network):
    """Test the forward pass of the actor network."""
    batch_size = 32
    state_dim = 10
    action_dim = 3
    
    # Create random input
    state = torch.randn(batch_size, state_dim)
    
    # Test forward pass
    action, log_prob = actor_network(state)
    
    # Check output shapes
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size,)  # One log prob per sample
    
    # Check if outputs are normalized
    assert torch.all(action >= -1) and torch.all(action <= 1)

def test_actor_sample_output(actor_network):
    """Test the sampling behavior of the actor network."""
    batch_size = 32
    state_dim = 10
    action_dim = 3
    
    # Create random input
    state = torch.randn(batch_size, state_dim)
    
    # Test mean output
    mean = actor_network.get_mean(state)
    assert mean.shape == (batch_size, action_dim)
    assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)
    
    # Test action sampling (using forward in training mode)
    actor_network.train()
    action, _ = actor_network(state)
    assert action.shape == (batch_size, action_dim)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

def test_actor_deterministic_output(actor_network):
    """Test deterministic output (means) from the actor network."""
    batch_size = 32
    state_dim = 10
    state = torch.randn(batch_size, state_dim)
    
    # Set to eval mode for deterministic output
    actor_network.eval()
    
    # Get means twice with the same input
    means1 = actor_network.get_mean(state)
    means2 = actor_network.get_mean(state)
    
    # Check means are identical for same input
    assert torch.allclose(means1, means2)
    
    # Check means are bounded
    assert torch.all(means1 >= -1.0)
    assert torch.all(means1 <= 1.0)

def test_actor_gradient_flow(actor_network):
    """Test that gradients flow through the network properly."""
    batch_size = 32
    state_dim = 10
    
    # Create random input
    state = torch.randn(batch_size, state_dim, requires_grad=True)
    
    # Forward pass and compute loss
    action, log_prob = actor_network(state)
    loss = action.mean() + log_prob.mean()
    
    # Backward pass
    loss.backward()
    
    # Check all parameters have gradients
    for param in actor_network.parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))

def test_actor_numerical_stability(actor_network):
    """Test numerical stability of log probability calculation."""
    batch_size = 32
    state_dim = 10
    state = torch.randn(batch_size, state_dim)
    
    # Test with actions close to bounds
    actions = torch.tensor([0.99, -0.99, 0.0]).repeat(batch_size, 1)
    log_probs = actor_network.log_prob(state, actions)
    
    # Check log probs are finite
    assert torch.all(torch.isfinite(log_probs))
    
    # Test with very small standard deviations
    with torch.no_grad():
        actor_network.log_std_net.bias.data.fill_(-10)
    actions, log_probs = actor_network(state)
    assert torch.all(torch.isfinite(log_probs)) 