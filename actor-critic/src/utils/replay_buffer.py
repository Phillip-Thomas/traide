import numpy as np
import torch
from typing import Dict, Tuple

class ReplayBuffer:
    """
    Replay buffer for storing and sampling trading experiences.
    Implements efficient numpy-based storage with prioritized sampling.
    """
    def __init__(
        self,
        state_dim: int,
        buffer_size: int = 1_000_000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.buffer_size = buffer_size
        self.device = torch.device(device)  # Convert to torch.device object
        
        # Initialize buffers with float32 dtype
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # Priority variables
        self.priorities = np.ones(buffer_size, dtype=np.float32)  # Initialize with 1s for stability
        self.max_priority = 1.0
        
        self.ptr = 0
        self.size = 0
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
        
    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state array
            action: Action taken
            reward: Reward received
            next_state: Next state array
            done: Whether episode ended
        """
        # Convert inputs to float32
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray([action], dtype=np.float32)
        reward = np.asarray([reward], dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        done = np.asarray([done], dtype=np.float32)
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Set max priority for new experience
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size: int, alpha: float = 0.6, beta: float = 0.4) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences with prioritized replay.
        
        Args:
            batch_size: Number of experiences to sample
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent
            
        Returns:
            Dictionary containing batch tensors on specified device
        """
        if self.size < batch_size:
            raise ValueError("Not enough experiences in buffer")
            
        # Compute sampling probabilities
        probs = self.priorities[:self.size] ** alpha
        probs /= probs.sum()
        
        # Sample indices and compute importance weights
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** -beta
        weights /= weights.max()
        
        # Get batch tensors
        batch = {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
            "weights": torch.FloatTensor(weights).to(self.device),
            "indices": indices
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for experiences based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        # Ensure inputs are float32 numpy arrays
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float32)
        
        # Add small constant for stability and ensure positive values
        priorities = np.abs(priorities) + 1e-6
        
        # Update priorities
        self.priorities[indices] = priorities
        
        # Update max priority
        self.max_priority = max(self.max_priority, priorities.max()) 