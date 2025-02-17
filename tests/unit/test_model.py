import pytest
import torch
import numpy as np
from models.model import DQN

@pytest.fixture
def input_size():
    """Standard input size for testing."""
    return 435  # window_size * 9 + 3

@pytest.fixture
def model(input_size, device):
    """Create a DQN model for testing."""
    model = DQN(input_size).to(device)
    model.eval()  # Set to evaluation mode for deterministic behavior
    return model

def test_model_initialization(model, input_size):
    """Test model initialization and architecture."""
    # Check model structure
    assert isinstance(model, DQN)
    assert isinstance(model.feature_net, torch.nn.Sequential)
    assert isinstance(model.advantage, torch.nn.Sequential)
    assert isinstance(model.value, torch.nn.Sequential)
    
    # Check first layer input size
    first_layer = model.feature_net[0]
    assert first_layer.in_features == input_size
    assert first_layer.out_features == 128

def test_model_output_shape(model, input_size, device):
    """Test model output shapes with both single and batch inputs."""
    # Test single input
    x = torch.randn(input_size).to(device)
    q_values, _ = model(x)
    assert q_values.shape == (1, 3)  # (batch_size=1, n_actions=3)
    
    # Test batch input
    batch_size = 32
    x_batch = torch.randn(batch_size, input_size).to(device)
    q_values_batch, _ = model(x_batch)
    assert q_values_batch.shape == (batch_size, 3)

def test_model_weight_initialization(model):
    """Test that weights are initialized using orthogonal initialization."""
    def check_orthogonal_init(layer):
        if isinstance(layer, torch.nn.Linear):
            # Check if weights follow orthogonal initialization
            # For orthogonal matrices, W * W^T should be close to identity
            W = layer.weight.data
            WWT = torch.mm(W, W.t())
            I = torch.eye(WWT.shape[0], device=W.device)
            assert torch.allclose(WWT, I, rtol=1e-3, atol=1e-3)
            
            # Check if biases are initialized to zero
            assert torch.allclose(layer.bias.data, torch.zeros_like(layer.bias.data))
    
    # Check all layers
    model.apply(check_orthogonal_init)

def test_model_dueling_architecture(model, input_size, device):
    """Test the dueling architecture implementation."""
    with torch.no_grad():  # Disable gradients for deterministic behavior
        x = torch.randn(1, input_size).to(device)
        
        # Get intermediate activations
        features = model.feature_net(x)
        advantage = model.advantage(features)
        value = model.value(features)
        
        # Manual dueling combination
        q_values_manual = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Compare with forward pass
        q_values_forward, _ = model(x)
        
        # Check that the relative differences between Q-values are preserved
        q_manual_diffs = q_values_manual - q_values_manual.mean()
        q_forward_diffs = q_values_forward - q_values_forward.mean()
        
        assert torch.allclose(q_manual_diffs, q_forward_diffs, rtol=1e-4, atol=1e-4)

def test_model_deterministic_output(model, input_size, device):
    """Test that model produces deterministic outputs for the same input."""
    with torch.no_grad():  # Disable gradients for deterministic behavior
        x = torch.randn(1, input_size).to(device)
        
        # Multiple forward passes
        q_values1, _ = model(x)
        q_values2, _ = model(x)
        q_values3, _ = model(x)
        
        # All outputs should be identical
        assert torch.allclose(q_values1, q_values2, rtol=1e-6, atol=1e-6)
        assert torch.allclose(q_values2, q_values3, rtol=1e-6, atol=1e-6)

def test_model_gradient_flow(model, input_size, device):
    """Test that gradients can flow through the model."""
    model.train()  # Set to training mode
    x = torch.randn(32, input_size).to(device)
    target = torch.randint(0, 3, (32,)).to(device)
    
    # Forward pass
    q_values, _ = model(x)
    loss = torch.nn.functional.cross_entropy(q_values, target)
    
    # Backward pass
    model.zero_grad()  # Clear any existing gradients
    loss.backward()
    
    # Check that at least some parameters have non-zero gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-8):
            has_grad = True
            break
    
    assert has_grad, "No parameter received meaningful gradients" 