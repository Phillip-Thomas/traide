# memory/replay_buffer.py
import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Gradually increase beta
        self.episodes = []
        self.priorities = []
        self.eps = 1e-6

    def __len__(self):
        return len(self.episodes)
        
    def push_episode(self, episode):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
            self.priorities.pop(0)
        
        # Calculate priority based on returns and episode characteristics
        returns = [transition[2] for transition in episode]
        total_return = sum(returns)
        return_std = np.std(returns) if len(returns) > 1 else 0
        
        # Higher priority for episodes with:
        # 1. Higher absolute returns
        # 2. Higher return variance (more interesting episodes)
        # 3. More state transitions
        priority = (abs(total_return) + return_std + len(episode)/1000 + self.eps) ** self.alpha
        priority = np.clip(priority, self.eps, 10.0)  # Limit extreme values
        
        print(f"Episode stats - Returns: {total_return:.4f}, Std: {return_std:.4f}, Length: {len(episode)}")
        print(f"Priority assigned: {priority:.4f}")
        
        self.episodes.append(episode)
        self.priorities.append(float(priority))
    
    def sample(self, batch_size, seq_len, device):
        if not self.episodes or len(self.episodes) < batch_size:
            return None
        
        # Increase beta for more uniform sampling over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = np.clip(priorities, self.eps, None)
        
        # Add temperature scaling for better exploration
        temperature = 1.0
        probs = np.exp(np.log(priorities) / temperature)
        probs = probs / np.sum(probs)
        
        # Sample with diversity (avoid sampling same episode multiple times)
        indices = []
        remaining_indices = list(range(len(self.episodes)))
        for _ in range(batch_size):
            if not remaining_indices:
                break
            idx = np.random.choice(remaining_indices, p=probs[remaining_indices]/np.sum(probs[remaining_indices]))
            indices.append(idx)
            remaining_indices.remove(idx)
        
        # Calculate importance sampling weights
        weights = (len(self.episodes) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        return self._get_sequences(indices, seq_len, weights, device)
    
    def _get_sequences(self, indices, seq_len, weights, device):
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_next_states, batch_dones = [], []
        valid_indices = []
        
        for idx in indices:
            episode = self.episodes[idx]
            if len(episode) < seq_len:
                continue
            
            # Sample sequences with momentum
            valid_start_indices = []
            for i in range(len(episode) - seq_len + 1):
                sequence = episode[i:i + seq_len]
                sequence_return = sum(t[2] for t in sequence)
                if abs(sequence_return) > 0.001:  # Minimum return threshold
                    valid_start_indices.append(i)
            
            if not valid_start_indices:
                continue
            
            # Prefer sequences with higher returns
            sequence_weights = []
            for start_idx in valid_start_indices:
                sequence = episode[start_idx:start_idx + seq_len]
                sequence_return = sum(t[2] for t in sequence)
                sequence_weights.append(abs(sequence_return))
            
            sequence_probs = np.array(sequence_weights) / np.sum(sequence_weights)
            start_idx = np.random.choice(valid_start_indices, p=sequence_probs)
            
            sequence = episode[start_idx:start_idx + seq_len]
            states, actions, rewards, next_states, dones = zip(*sequence)
            
            batch_states.append(np.array(states))
            batch_actions.append(np.array(actions))
            batch_rewards.append(np.array(rewards))
            batch_next_states.append(np.array(next_states))
            batch_dones.append(np.array(dones))
            valid_indices.append(idx)
        
        if not batch_states:
            return None
        
        # Convert to tensors
        return {
            'states': torch.tensor(np.array(batch_states), device=device, dtype=torch.float32),
            'actions': torch.tensor(np.array(batch_actions), device=device, dtype=torch.long),
            'rewards': torch.tensor(np.array(batch_rewards), device=device, dtype=torch.float32),
            'next_states': torch.tensor(np.array(batch_next_states), device=device, dtype=torch.float32),
            'dones': torch.tensor(np.array(batch_dones), device=device, dtype=torch.float32),
            'weights': torch.tensor(weights[:len(valid_indices)], device=device, dtype=torch.float32),
            'indices': valid_indices
        }
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                if np.isnan(error):
                    print(f"Warning: NaN error detected for index {idx}")
                    continue
                error = np.clip(float(error), -1e6, 1e6)
                priority = np.power(abs(error) + self.eps, self.alpha)
                self.priorities[idx] = float(priority)