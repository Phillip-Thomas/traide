# models/model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        # Value and Advantage streams
        self.advantage = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 actions: hold (0), buy (1), sell (2)
        )
        
        self.value = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,)
        Returns:
            Tuple of (q_values, None), where q_values has shape (batch_size, n_actions)
        """
        # Add batch dimension if input is a single state
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        features = self.feature_net(x)
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Combine value and advantage (Dueling DQN)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, None