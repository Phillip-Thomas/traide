import torch
import torch.nn as nn
import numpy as np

class SACCriticNetwork(nn.Module):
    """
    Critic network for SAC, estimates Q-values for state-action pairs.
    Handles multi-dimensional continuous actions with improved stability.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        # Build Q-network with improved architecture
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1)
        ])
        
        # Hidden layer 1
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1)
        ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.q_net = nn.Sequential(*layers)
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network with improved stability."""
        # Concatenate state and action with improved numerical stability
        x = torch.cat([state, action], dim=1)
        
        # Forward pass with value clipping for stability
        q_value = self.q_net(x)
        return torch.clamp(q_value, -100, 100)  # Clip Q-values for stability 