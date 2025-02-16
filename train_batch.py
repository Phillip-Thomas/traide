import time
import torch
import torch.nn.functional as F

def train_batch(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    batch = memory.sample(batch_size, seq_len=10, device=device)
    if batch is None:
        return 0
    
    states, actions, rewards, next_states, dones, weights = batch
    
    # Double Q-learning
    with torch.no_grad():
        next_q_values, _ = policy_net(next_states)
        next_actions = next_q_values.max(1)[1]
        next_q_target, _ = target_net(next_states)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1))
        target_q = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * next_q
    
    # Current Q-values
    q_values, _ = policy_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1))
    
    # Huber loss with importance sampling weights
    loss = F.smooth_l1_loss(q_value, target_q, reduction='none')
    weighted_loss = (loss * weights.unsqueeze(1)).mean()
    
    # Gradient clipping
    optimizer.zero_grad()
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Update priorities
    errors = loss.detach().cpu().numpy()
    memory.update_priorities(batch['indices'], errors)
    
    return weighted_loss.item()
