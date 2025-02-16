import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OptimizedDQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Store input size for validation
        self.input_size = input_size
        
        # Calculate hidden sizes based on input size
        hidden_size = min(2048, max(1024, input_size * 2))  # Scale with input size
        reduced_size = hidden_size // 2

        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        # Parallel processing paths
        self.parallel_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, reduced_size),
                nn.LayerNorm(reduced_size),
                nn.SiLU(),
                nn.Dropout(0.1)
            ) for _ in range(4)  # 4 parallel paths
        ])

        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.LayerNorm(reduced_size),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        # Output layer
        self.output = nn.Linear(reduced_size, 3)  # 3 actions: HOLD, BUY, SELL

        # Print model architecture summary
        print(f"\nDQN Architecture:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Reduced size: {reduced_size}")
        print(f"  Parallel paths: 4")
        print(f"  Output size: 3")

    def forward(self, x):
        # Validate input shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        elif x.dim() == 3:
            x = x.squeeze(1)  # Remove extra dimension
            
        # Validate input size
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Input size mismatch. Got {x.size(-1)} features but model expects {self.input_size}.\n"
                f"Input shape: {x.shape}"
            )
            
        # Process features
        features = self.feature_net(x)

        # Process in parallel paths
        parallel_outputs = [net(features) for net in self.parallel_nets]
        combined = torch.stack(parallel_outputs, dim=1)
        combined = combined.mean(dim=1)

        # Final processing
        combined = self.combine(features)  # Use features directly
        q_values = self.output(combined)
        
        # Return Q-values and auxiliary outputs (features and parallel outputs for regularization)
        return q_values, (features, parallel_outputs)

class ReplayBuffer:
    def __init__(self, capacity, state_size, device):
        self.capacity = capacity
        self.state_size = state_size
        self.device = device
        
        # Pre-allocate tensors on specified device
        self.states = torch.zeros((capacity, state_size), device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros((capacity, state_size), device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.position = 0
        self.size = 0
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        """Save a batch of transitions with device handling"""
        batch_size = len(states)
        indices = torch.arange(self.position, self.position + batch_size) % self.capacity
        
        # Ensure all inputs are on correct device
        self.states[indices] = states.to(self.device, non_blocking=True)
        self.actions[indices] = actions.to(self.device, non_blocking=True)
        self.rewards[indices] = rewards.to(self.device, non_blocking=True)
        self.next_states[indices] = next_states.to(self.device, non_blocking=True)
        self.dones[indices] = dones.to(self.device, non_blocking=True)
        
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

def compute_q_loss(current_q_values, next_q_values, actions, rewards, dones, gamma):
    """Compute Q-learning loss"""
    # Get Q-values for taken actions
    q_values = current_q_values.gather(1, actions.unsqueeze(1))

    # Compute target Q-values
    with torch.no_grad():
        max_next_q = next_q_values.max(1)[0].unsqueeze(1)
        target_q = rewards.unsqueeze(1) + gamma * max_next_q * (1 - dones.unsqueeze(1))

    # Compute loss
    loss = F.smooth_l1_loss(q_values, target_q)

    return loss

def select_actions_gpu(q_values, epsilon, device):
    """Select actions using epsilon-greedy policy on GPU"""
    # Handle case where q_values might be a tuple
    if isinstance(q_values, tuple):
        q_values = q_values[0]  # Extract just the Q-values
        
    # Ensure all tensors are on CPU for worker processes
    q_values = q_values.cpu()
    batch_size = q_values.size(0)

    # Generate random numbers for epsilon-greedy on CPU
    random_actions = torch.randint(0, 3, (batch_size,))
    greedy_actions = torch.argmax(q_values, dim=1)

    # Create random mask for epsilon-greedy
    random_mask = torch.rand(batch_size) < epsilon

    # Combine random and greedy actions
    actions = torch.where(random_mask, random_actions, greedy_actions)

    return actions

def create_optimized_tensors(batch_size, input_size, device):
    """Create pre-allocated tensors optimized for GPU performance"""
    # Use much larger sizes for better GPU utilization
    expanded_batch = batch_size * 8  # Increase batch multiplier

    # Create tensors on CPU first, then move to device
    tensors = {
        'states': torch.zeros(
            (expanded_batch, input_size),
            dtype=torch.float16
        ),
        'next_states': torch.zeros(
            (expanded_batch, input_size),
            dtype=torch.float16
        ),
        'actions': torch.zeros(
            expanded_batch,
            dtype=torch.int32
        ),
        'rewards': torch.zeros(
            expanded_batch,
            dtype=torch.float16
        ),
        'dones': torch.zeros(
            expanded_batch,
            dtype=torch.bool
        ),
        'q_values': torch.zeros(
            (expanded_batch, 3),
            dtype=torch.float16
        )
    }

    # Pin memory for CPU tensors and move to device
    for key in tensors:
        if device.type == 'cuda':
            tensors[key] = tensors[key].pin_memory().to(device, non_blocking=True)
        else:
            tensors[key] = tensors[key].to(device)
        tensors[key] = tensors[key].contiguous()

    return tensors 