import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .sac_actor import SACActorNetwork
from .sac_critic import SACCriticNetwork
from ..utils.replay_buffer import ReplayBuffer

class SACAgent:
    """
    Soft Actor-Critic agent implementation.
    Combines policy and value networks with SAC training logic.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hidden_dim: int = 256,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        learning_rate: float = 3e-4,
        automatic_entropy_tuning: bool = True,
        gradient_clip: float = 1.0,
        use_batch_norm: bool = True
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.gradient_clip = gradient_clip
        
        # Initialize networks with batch norm
        self.actor = SACActorNetwork(
            state_dim, action_dim, hidden_dim, use_batch_norm=use_batch_norm
        ).to(device)
        
        self.critic_1 = SACCriticNetwork(
            state_dim, action_dim, hidden_dim, use_batch_norm=use_batch_norm
        ).to(device)
        
        self.critic_2 = SACCriticNetwork(
            state_dim, action_dim, hidden_dim, use_batch_norm=use_batch_norm
        ).to(device)
        
        # Initialize target networks
        self.critic_1_target = SACCriticNetwork(
            state_dim, action_dim, hidden_dim, use_batch_norm=use_batch_norm
        ).to(device)
        
        self.critic_2_target = SACCriticNetwork(
            state_dim, action_dim, hidden_dim, use_batch_norm=use_batch_norm
        ).to(device)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers with gradient clipping
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)
        
        # Initialize automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim  # Set target entropy to -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action using the policy network.
        
        Args:
            state: Current state array
            evaluate: Whether to evaluate deterministically
            
        Returns:
            action: Selected action array
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            state = state.to(self.device, non_blocking=True).unsqueeze(0)
            
            if evaluate:
                _, mean = self.actor(state)
                return mean.cpu().numpy()[0]
            else:
                action, _ = self.actor(state)
                return action.cpu().numpy()[0]
    
    def update_parameters(self) -> Dict[str, float]:
        """
        Update the parameters of all networks using SAC update rules.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # Sample batch and move to device
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Update critics and actor in parallel if possible
        critic_loss_1, critic_loss_2 = self._update_critics(batch)
        actor_loss, log_probs = self._update_actor(batch)
        
        # Update alpha if automatic entropy tuning is enabled
        if self.automatic_entropy_tuning:
            alpha_loss = self._update_alpha(log_probs)
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = 0
        
        # Update target networks
        self._update_targets()
        
        return {
            "critic_1_loss": critic_loss_1.item(),
            "critic_2_loss": critic_loss_2.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            "alpha": self.alpha
        }
    
    def _update_critics(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update critic networks with improved stability."""
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                next_actions, next_log_probs = self.actor(batch['next_states'])
                next_q1 = self.critic_1_target(batch['next_states'], next_actions)
                next_q2 = self.critic_2_target(batch['next_states'], next_actions)
                next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs.unsqueeze(-1)
                target_q = batch['rewards'].unsqueeze(-1) + (1 - batch['dones'].unsqueeze(-1)) * self.gamma * next_q
            
            # Compute both critics in parallel
            current_q1 = self.critic_1(batch['states'], batch['actions'])
            current_q2 = self.critic_2(batch['states'], batch['actions'])
            
            critic_1_loss = F.mse_loss(current_q1, target_q.detach())
            critic_2_loss = F.mse_loss(current_q2, target_q.detach())
            
            # Update both critics with gradient clipping
            self.critic_1_optimizer.zero_grad(set_to_none=True)
            self.critic_2_optimizer.zero_grad(set_to_none=True)
            
            critic_1_loss.backward()
            critic_2_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.gradient_clip)
            
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()
            
            return critic_1_loss, critic_2_loss
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update actor network with improved stability."""
        with torch.cuda.amp.autocast():
            actions, log_probs = self.actor(batch['states'])
            
            # Compute Q-values in parallel
            q1 = self.critic_1(batch['states'], actions)
            q2 = self.critic_2(batch['states'], actions)
            min_q = torch.min(q1, q2)
            
            actor_loss = (self.alpha * log_probs - min_q).mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            
            self.actor_optimizer.step()
            
            return actor_loss, log_probs
    
    def _update_alpha(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Update temperature parameter with improved stability."""
        with torch.cuda.amp.autocast():
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.gradient_clip)
            
            self.alpha_optimizer.step()
            
            self.alpha = torch.clamp(self.log_alpha.exp(), 0.01, 1.0).item()
            
            return alpha_loss
    
    def _update_targets(self) -> None:
        """Update target networks using soft update with improved efficiency."""
        with torch.no_grad():
            for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
            
            for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
    
    def save(self, path: str) -> None:
        """Save model state dictionaries."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None
        }, path)
    
    def load(self, path: str) -> None:
        """Load model state dictionaries."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer']) 