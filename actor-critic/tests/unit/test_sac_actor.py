import torch
import pytest
import numpy as np
from src.models.sac_actor import SACActorNetwork

@pytest.fixture
def actor_network():
    # Set all random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return SACActorNetwork(state_dim=10, hidden_dim=64)

def test_actor_network_initialization(actor_network):
    """Test that the network initializes with correct dimensions."""
    assert isinstance(actor_network, SACActorNetwork)
    
    # Check network architecture
    assert actor_network.feature_net[0].normalized_shape[0] == 10  # Input dimension
    assert actor_network.feature_net[1].in_features == 10
    assert actor_network.feature_net[1].out_features == 64
    assert actor_network.mean_head.in_features == 64
    assert actor_network.mean_head.out_features == 1
    assert actor_network.log_std_head.in_features == 64
    assert actor_network.log_std_head.out_features == 1

def test_actor_forward_pass(actor_network):
    """Test the forward pass of the network."""
    batch_size = 32
    state_dim = 10
    
    # Create random input
    state = torch.randn(batch_size, state_dim)
    
    # Forward pass
    mean, log_std = actor_network(state)
    
    # Check output dimensions
    assert mean.shape == (batch_size, 1)
    assert log_std.shape == (batch_size, 1)
    
    # Check log_std bounds
    assert torch.all(log_std >= actor_network.log_std_min)
    assert torch.all(log_std <= actor_network.log_std_max)

def test_actor_sample_output(actor_network):
    """Test the action sampling from the network."""
    batch_size = 32
    state_dim = 10
    
    # Create random input
    state = torch.randn(batch_size, state_dim)
    
    # Sample action
    action, log_prob = actor_network.sample(state)
    
    # Check output dimensions
    assert action.shape == (batch_size, 1)
    assert log_prob.shape == (batch_size, 1)
    
    # Check action bounds
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    
    # Check log_prob is finite
    assert torch.all(torch.isfinite(log_prob))

def test_actor_deterministic_output(actor_network):
    """Test that the network produces deterministic output with fixed seed."""
    batch_size = 32
    state_dim = 10
    
    # Create random input with fixed seed
    torch.manual_seed(42)
    state = torch.randn(batch_size, state_dim)
    
    # Set network to evaluation mode
    actor_network.eval()
    
    # Get outputs twice with the same input
    with torch.no_grad():
        mean1, log_std1 = actor_network(state)
        mean2, log_std2 = actor_network(state)
    
    # Check outputs are identical
    assert torch.equal(mean1, mean2)
    assert torch.equal(log_std1, log_std2)
    
    # Reset network to training mode
    actor_network.train()

def test_actor_gradient_flow(actor_network):
    """Test that gradients flow through the network properly."""
    batch_size = 32
    state_dim = 10
    
    # Create random input
    state = torch.randn(batch_size, state_dim, requires_grad=True)
    
    # Forward pass and compute loss
    action, log_prob = actor_network.sample(state)
    loss = action.mean() + log_prob.mean()
    
    # Backward pass
    loss.backward()
    
    # Check all parameters have gradients
    for param in actor_network.parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad)) 