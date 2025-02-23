import numpy as np
import torch
from typing import Dict, Tuple
import logging

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
        max_size: int = 1000000,
        device: str = "cuda"
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Always use CUDA for better performance
        self.use_cuda = device.startswith("cuda")
        
        # Initialize buffers directly on GPU if CUDA is available
        if self.use_cuda:
            with torch.cuda.device(device):
                self._initialize_buffers()
        else:
            self._initialize_buffers()
        
    def _initialize_buffers(self):
        """Initialize buffers with mixed precision."""
        # Use mixed precision for better memory efficiency
        with torch.amp.autocast('cuda') if self.use_cuda else torch.no_grad():
            # States and next_states in half precision
            self.states = torch.zeros((self.max_size, self.state_dim),
                                    dtype=torch.float16 if self.use_cuda else torch.float32,
                                    device=self.device)
            self.next_states = torch.zeros((self.max_size, self.state_dim),
                                         dtype=torch.float16 if self.use_cuda else torch.float32,
                                         device=self.device)
            
            # Other tensors in full precision
            self.actions = torch.zeros((self.max_size, self.action_dim),
                                     dtype=torch.float32,
                                     device=self.device)
            self.rewards = torch.zeros(self.max_size,
                                     dtype=torch.float32,
                                     device=self.device)
            self.dones = torch.zeros(self.max_size,
                                   dtype=torch.bool,
                                   device=self.device)
            
            # Pre-allocate indices buffer on GPU
            self.sample_indices = torch.zeros(self.max_size,
                                            dtype=torch.long,
                                            device=self.device)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a new experience to memory."""
        try:
            # Convert all inputs to tensors
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_tensor = torch.FloatTensor(action).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            done_tensor = torch.BoolTensor([done]).to(self.device)  # Convert bool to tensor
            
            # Store experience
            self.states[self.ptr] = state_tensor
            self.actions[self.ptr] = action_tensor
            self.rewards[self.ptr] = reward_tensor
            self.next_states[self.ptr] = next_state_tensor
            self.dones[self.ptr] = done_tensor
            
            # Update pointer
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
        except Exception as e:
            logging.error(f"Error adding to replay buffer: {str(e)}")
            raise
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch efficiently on GPU."""
        if self.size < batch_size:
            raise ValueError(f"Not enough transitions ({self.size}) to sample batch of {batch_size}")
        
        with torch.amp.autocast('cuda') if self.use_cuda else torch.no_grad():
            # Generate random indices directly on GPU
            ind = torch.randint(0, self.size, (batch_size,),
                              device=self.device,
                              dtype=torch.long)
            
            # Efficient indexing on GPU
            batch = {
                'states': self.states[ind],  # Already in FP16
                'actions': self.actions[ind],
                'rewards': self.rewards[ind],
                'next_states': self.next_states[ind],  # Already in FP16
                'dones': self.dones[ind]
            }
            
            return batch
    
    def __len__(self) -> int:
        return self.size 