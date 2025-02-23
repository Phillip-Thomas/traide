import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

class SACActorNetwork(nn.Module):
    """
    Actor network for SAC, outputs mean and log_std of action distribution.
    Handles multi-dimensional continuous actions with improved stability.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        # Shared feature layers with layer norm and dropout for regularization
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Always use LayerNorm for better single-sample performance
            nn.ReLU(),
            # nn.Dropout(0.1)
        ])
        
        # Hidden layer
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Always use LayerNorm for better single-sample performance
            nn.ReLU(),
            # nn.Dropout(0.1)
        ])
        
        self.feature_net = nn.Sequential(*layers)
        
        # Mean and log_std heads with proper initialization
        self.mean_net = nn.Linear(hidden_dim, action_dim)
        self.log_std_net = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network with improved stability."""
        # Ensure input is at least 2D for LayerNorm
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        features = self.feature_net(state)
        
        # Get mean and log_std
        mean = self.mean_net(features)
        log_std = torch.clamp(self.log_std_net(features), 
                            self.log_std_min, 
                            self.log_std_max)
        
        if self.training:
            # Sample action using reparameterization trick with improved numerical stability
            std = torch.exp(log_std)
            eps = torch.randn_like(mean)
            eps = torch.clamp(eps, -5, 5)  # Clip noise for stability
            action_raw = mean + eps * std
        else:
            # Use mean action in eval mode
            action_raw = mean
        
        # Apply tanh with improved numerical stability
        action = torch.tanh(action_raw)
        
        # Calculate log probability with improved stability
        log_prob = self._compute_log_prob(mean, log_std, action, action_raw)
        
        return action, log_prob
    
    def get_mean(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic mean action for exploitation."""
        # Ensure input is at least 2D for LayerNorm
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        features = self.feature_net(state)
        mean = self.mean_net(features)
        return torch.tanh(mean)
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability with improved efficiency."""
        # Ensure input is at least 2D for LayerNorm
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        features = self.feature_net(state)
        mean = self.mean_net(features)
        log_std = torch.clamp(self.log_std_net(features), 
                            self.log_std_min, 
                            self.log_std_max)
        
        return self._compute_log_prob(mean, log_std, action, action)
    
    def _compute_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        action: torch.Tensor,
        action_raw: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability with improved numerical stability."""
        # Calculate log probability under Gaussian with improved stability
        std = torch.exp(log_std)
        log_prob = -0.5 * (
            ((action_raw - mean) / (std + 1e-8)).pow(2) + 
            2 * log_std + 
            np.log(2 * np.pi)
        )
        
        # Account for tanh transformation with improved stability
        log_prob -= torch.log(1 - action.pow(2) + 1e-8)
        
        # Sum across action dimensions and clip for stability
        return torch.clamp(log_prob.sum(-1), -100, 20) 