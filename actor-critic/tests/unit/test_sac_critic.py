import torch
import pytest
import numpy as np
from src.models.sac_critic import SACCriticNetwork
import torch.nn as nn

@pytest.fixture
def critic_network():
    # Set all random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return SACCriticNetwork(state_dim=10, action_dim=3, hidden_dim=64)

def test_critic_network_initialization(critic_network):
    """Test the initialization of the critic network."""
    # Check network structure
    assert isinstance(critic_network.q_net, nn.Sequential)
    assert isinstance(critic_network.q_net[0], nn.Linear)
    assert isinstance(critic_network.q_net[1], nn.LayerNorm)  # Default is LayerNorm
    assert isinstance(critic_network.q_net[2], nn.ReLU)
    
    # Check input/output dimensions
    assert critic_network.q_net[0].in_features == 13  # state_dim + action_dim
    assert critic_network.q_net[0].out_features == 64  # hidden_dim
    assert isinstance(critic_network.q_net[-1], nn.Linear)
    assert critic_network.q_net[-1].out_features == 1  # Single Q-value output

def test_critic_forward_pass(critic_network):
    """Test the forward pass of the critic network."""
    batch_size = 32
    state_dim = 10
    action_dim = 3
    
    # Create random input
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    
    # Forward pass
    q_value = critic_network(state, action)
    
    # Check output shape
    assert q_value.shape == (batch_size, 1)
    
    # Check if outputs have reasonable values
    assert torch.all(torch.isfinite(q_value))
    assert torch.all(q_value >= -100) and torch.all(q_value <= 100)  # Reasonable Q-value range

def test_critic_deterministic_output(critic_network):
    """Test that the network produces deterministic output with fixed seed."""
    batch_size = 32
    state_dim = 10
    action_dim = 3
    
    # Create random input with fixed seed
    torch.manual_seed(42)
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    
    # Set network to evaluation mode
    critic_network.eval()
    
    # Get outputs twice with the same input
    with torch.no_grad():
        q1 = critic_network(state, action)
        q2 = critic_network(state, action)
    
    # Check outputs are identical
    assert torch.equal(q1, q2)

def test_critic_gradient_flow(critic_network):
    """Test that gradients flow through the network properly."""
    batch_size = 32
    state_dim = 10
    action_dim = 3
    
    # Create random input
    state = torch.randn(batch_size, state_dim, requires_grad=True)
    action = torch.randn(batch_size, action_dim, requires_grad=True)
    
    # Forward pass and compute loss
    q_value = critic_network(state, action)
    loss = q_value.mean()
    
    # Backward pass
    loss.backward()
    
    # Check all parameters have gradients
    for param in critic_network.parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))
    
    # Check input gradients
    assert state.grad is not None
    assert action.grad is not None
    assert torch.all(torch.isfinite(state.grad))
    assert torch.all(torch.isfinite(action.grad))

def test_critic_numerical_stability(critic_network):
    """Test numerical stability with extreme inputs."""
    batch_size = 32
    state_dim = 10
    action_dim = 3
    
    # Test with large values
    state = torch.randn(batch_size, state_dim) * 1000
    action = torch.randn(batch_size, action_dim) * 1000
    q_value = critic_network(state, action)
    assert torch.all(torch.isfinite(q_value))
    
    # Test with small values
    state = torch.randn(batch_size, state_dim) * 1e-6
    action = torch.randn(batch_size, action_dim) * 1e-6
    q_value = critic_network(state, action)
    assert torch.all(torch.isfinite(q_value)) 