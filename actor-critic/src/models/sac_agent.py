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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hidden_dim: int = 256,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        learning_rate: float = 3e-4,
        automatic_entropy_tuning: bool = True
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Initialize networks
        self.actor = SACActorNetwork(state_dim, hidden_dim).to(device)
        self.critic_1 = SACCriticNetwork(state_dim, hidden_dim).to(device)
        self.critic_2 = SACCriticNetwork(state_dim, hidden_dim).to(device)
        
        # Initialize target networks
        self.critic_1_target = SACCriticNetwork(state_dim, hidden_dim).to(device)
        self.critic_2_target = SACCriticNetwork(state_dim, hidden_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, buffer_size, device)
        
        # Initialize automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -np.prod((1,)).item()  # -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action using the current policy.
        
        Args:
            state: Current state observation
            evaluate: Whether to evaluate deterministically
            
        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # Use mean action for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state)
                return torch.tanh(mean).cpu().numpy()[0]
        else:
            # Sample action for training
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]
    
    def update_parameters(self) -> Dict[str, float]:
        """
        Update the parameters of all networks using SAC update rules.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Update critics
        critic_loss_1, critic_loss_2 = self._update_critics(batch)
        
        # Update actor
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
        """Update critic networks."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(batch["next_states"])
            
            # Target Q-values
            next_q1_1, next_q1_2 = self.critic_1_target(batch["next_states"], next_actions)
            next_q2_1, next_q2_2 = self.critic_2_target(batch["next_states"], next_actions)
            next_q = torch.min(torch.min(next_q1_1, next_q1_2), torch.min(next_q2_1, next_q2_2)) - self.alpha * next_log_probs
            target_q = batch["rewards"] + (1 - batch["dones"]) * self.gamma * next_q
        
        # Current Q-values
        current_q1_1, current_q1_2 = self.critic_1(batch["states"], batch["actions"])
        current_q2_1, current_q2_2 = self.critic_2(batch["states"], batch["actions"])
        
        # Compute critic losses
        critic_loss_1 = F.mse_loss(current_q1_1, target_q) + F.mse_loss(current_q1_2, target_q)
        critic_loss_2 = F.mse_loss(current_q2_1, target_q) + F.mse_loss(current_q2_2, target_q)
        
        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()
        
        return critic_loss_1, critic_loss_2
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update actor network."""
        actions, log_probs = self.actor.sample(batch["states"])
        
        # Compute actor loss
        q1_1, q1_2 = self.critic_1(batch["states"], actions)
        q2_1, q2_2 = self.critic_2(batch["states"], actions)
        min_q = torch.min(torch.min(q1_1, q1_2), torch.min(q2_1, q2_2))
        
        actor_loss = (self.alpha * log_probs - min_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss, log_probs
    
    def _update_alpha(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Update temperature parameter."""
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss
    
    def _update_targets(self) -> None:
        """Update target networks using soft update rule."""
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, path: str) -> None:
        """Save model parameters."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            },
            path
        )
    
    def load(self, path: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(path)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"]) 