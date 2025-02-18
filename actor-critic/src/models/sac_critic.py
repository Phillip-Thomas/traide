import torch
import torch.nn as nn
import torch.nn.functional as F

class SACCriticNetwork(nn.Module):
    """
    Critic network for SAC, implementing Q-value estimation.
    Uses dual Q-networks to reduce overestimation bias.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Shared feature extraction for state with LayerNorm instead of BatchNorm
        self.state_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for action
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for action
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute Q-values for state-action pairs.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Action tensor (batch_size, 1)
            
        Returns:
            q1_value: Q-value from first network
            q2_value: Q-value from second network
        """
        state_features = self.state_net(state)
        sa = torch.cat([state_features, action], dim=1)
        
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q1 value only (used for policy gradient).
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Action tensor (batch_size, 1)
            
        Returns:
            q1_value: Q-value from first network
        """
        state_features = self.state_net(state)
        sa = torch.cat([state_features, action], dim=1)
        return self.q1_net(sa) 