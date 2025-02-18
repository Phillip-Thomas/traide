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
        max_size: int = 100_000,
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
        with torch.cuda.amp.autocast() if self.use_cuda else torch.no_grad():
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
        """Add a transition directly to GPU memory."""
        with torch.cuda.amp.autocast() if self.use_cuda else torch.no_grad():
            # Convert to tensors and move to GPU in one operation
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(device=self.device, 
                                                 dtype=torch.float16 if self.use_cuda else torch.float32,
                                                 non_blocking=True)
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(device=self.device,
                                                   dtype=torch.float32,
                                                   non_blocking=True)
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).to(device=self.device,
                                                           dtype=torch.float16 if self.use_cuda else torch.float32,
                                                           non_blocking=True)
            
            # Efficient in-place operations
            self.states[self.ptr].copy_(state, non_blocking=True)
            self.actions[self.ptr].copy_(action, non_blocking=True)
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr].copy_(next_state, non_blocking=True)
            self.dones[self.ptr] = done
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch efficiently on GPU."""
        if self.size < batch_size:
            raise ValueError(f"Not enough transitions ({self.size}) to sample batch of {batch_size}")
        
        with torch.cuda.amp.autocast() if self.use_cuda else torch.no_grad():
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