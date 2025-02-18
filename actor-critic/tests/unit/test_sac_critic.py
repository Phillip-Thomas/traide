import torch
import pytest
import numpy as np
from src.models.sac_critic import SACCriticNetwork

@pytest.fixture
def critic_network():
    # Set all random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return SACCriticNetwork(state_dim=10, hidden_dim=64)

def test_critic_network_initialization(critic_network):
    """Test that the network initializes with correct dimensions."""
    assert isinstance(critic_network, SACCriticNetwork)
    
    # Check network architecture
    assert critic_network.state_net[0].normalized_shape[0] == 10  # Input dimension
    assert critic_network.state_net[1].in_features == 10
    assert critic_network.state_net[1].out_features == 64
    
    # Check Q-network dimensions
    assert critic_network.q1_net[0].in_features == 65  # hidden_dim + action_dim
    assert critic_network.q1_net[0].out_features == 64
    assert critic_network.q1_net[-1].out_features == 1
    
    assert critic_network.q2_net[0].in_features == 65
    assert critic_network.q2_net[0].out_features == 64
    assert critic_network.q2_net[-1].out_features == 1

def test_critic_forward_pass(critic_network):
    """Test the forward pass of the network."""
    batch_size = 32
    state_dim = 10
    action_dim = 1
    
    # Create random input
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    
    # Forward pass
    q1, q2 = critic_network(state, action)
    
    # Check output dimensions
    assert q1.shape == (batch_size, 1)
    assert q2.shape == (batch_size, 1)
    
    # Check outputs are finite
    assert torch.all(torch.isfinite(q1))
    assert torch.all(torch.isfinite(q2))

def test_critic_q1_output(critic_network):
    """Test the Q1 network output."""
    batch_size = 32
    state_dim = 10
    action_dim = 1
    
    # Create random input with fixed seed
    torch.manual_seed(42)
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    
    # Set network to evaluation mode
    critic_network.eval()
    
    # Get Q1 value
    with torch.no_grad():
        q1 = critic_network.q1(state, action)
        q1_forward, _ = critic_network(state, action)
    
    # Check output dimension
    assert q1.shape == (batch_size, 1)
    
    # Check output is finite
    assert torch.all(torch.isfinite(q1))
    
    # Check Q1 matches forward pass
    assert torch.equal(q1, q1_forward)
    
    # Reset network to training mode
    critic_network.train()

def test_critic_deterministic_output(critic_network):
    """Test that the network produces deterministic output with fixed seed."""
    batch_size = 32
    state_dim = 10
    action_dim = 1
    
    # Create random input with fixed seed
    torch.manual_seed(42)
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    
    # Set network to evaluation mode
    critic_network.eval()
    
    # Get outputs twice with the same input
    with torch.no_grad():
        q1_1, q2_1 = critic_network(state, action)
        q1_2, q2_2 = critic_network(state, action)
    
    # Check outputs are identical
    assert torch.equal(q1_1, q1_2)
    assert torch.equal(q2_1, q2_2)
    
    # Reset network to training mode
    critic_network.train()

def test_critic_gradient_flow(critic_network):
    """Test that gradients flow through the network properly."""
    batch_size = 32
    state_dim = 10
    action_dim = 1
    
    # Create random input
    state = torch.randn(batch_size, state_dim, requires_grad=True)
    action = torch.randn(batch_size, action_dim, requires_grad=True)
    
    # Forward pass and compute loss
    q1, q2 = critic_network(state, action)
    loss = q1.mean() + q2.mean()
    
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