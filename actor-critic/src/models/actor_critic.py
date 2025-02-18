import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic model for portfolio management.
    The actor outputs position sizes for multiple assets,
    while the critic evaluates the state value.
    """
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_size: int = 128,
        n_hidden: int = 2,
    ):
        super().__init__()
        
        # Shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(n_features + n_assets, hidden_size),  # +n_assets for current positions
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
                for _ in range(n_hidden - 1)
            ]
        )
        
        # Actor head - outputs position sizes for each asset
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_assets),
            nn.Tanh()  # Bound positions between -1 and 1
        )
        
        # Critic head - evaluates state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape (batch_size, n_features + n_assets)
                  containing features and current positions
        
        Returns:
            actions: Tensor of shape (batch_size, n_assets) containing position sizes
            value: Tensor of shape (batch_size, 1) containing state values
        """
        features = self.feature_layers(state)
        actions = self.actor(features)
        value = self.critic(features)
        return actions, value 