import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SACActorNetwork(nn.Module):
    """
    Actor network for SAC, implementing a squashed Gaussian policy.
    Outputs mean and log_std for position sizing between -1 and 1.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extraction layers with LayerNorm instead of BatchNorm
        self.feature_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute action distribution parameters.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            
        Returns:
            mean: Mean of the policy distribution
            log_std: Log standard deviation of the policy distribution
        """
        features = self.feature_net(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy given the current state.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            
        Returns:
            action: Sampled action from the policy
            log_prob: Log probability of the sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash sample using tanh to bound between -1 and 1
        action = torch.tanh(x_t)
        
        # Compute log probability, using change of variables formula
        log_prob = normal.log_prob(x_t)
        # Account for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob 