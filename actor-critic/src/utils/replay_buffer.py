import numpy as np
import torch
from typing import Dict, Tuple

class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    Handles multi-dimensional states and actions.
    Uses pinned memory for faster GPU transfer.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        max_size: int = 1_000_000,
        device: str = "cpu"
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Use pinned memory if using CUDA
        pin_memory = device.startswith("cuda")
        
        # Initialize buffers directly as tensors
        self.states = torch.zeros((max_size, state_dim), 
                                dtype=torch.float32, 
                                pin_memory=pin_memory)
        self.actions = torch.zeros((max_size, action_dim), 
                                 dtype=torch.float32, 
                                 pin_memory=pin_memory)
        self.rewards = torch.zeros(max_size, 
                                 dtype=torch.float32, 
                                 pin_memory=pin_memory)
        self.next_states = torch.zeros((max_size, state_dim), 
                                     dtype=torch.float32, 
                                     pin_memory=pin_memory)
        self.dones = torch.zeros(max_size, 
                               dtype=torch.float32, 
                               pin_memory=pin_memory)
        
        # Pre-allocate sample indices buffer
        self.sample_indices = torch.zeros(max_size, dtype=torch.long, pin_memory=pin_memory)
        
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        # Convert numpy arrays to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state)
            
        with torch.no_grad():
            self.states[self.ptr].copy_(state)
            self.actions[self.ptr].copy_(action)
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr].copy_(next_state)
            self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions using efficient indexing."""
        if self.size < batch_size:
            raise ValueError(f"Not enough transitions ({self.size}) to sample batch of {batch_size}")
            
        # Generate random indices on CPU
        ind = torch.randint(0, self.size, (batch_size,))
        
        # Sample and move only the batch to device
        with torch.no_grad():
            batch = {
                'states': self.states[ind].to(self.device),
                'actions': self.actions[ind].to(self.device),
                'rewards': self.rewards[ind].to(self.device),
                'next_states': self.next_states[ind].to(self.device),
                'dones': self.dones[ind].to(self.device)
            }
        
        return batch
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size 