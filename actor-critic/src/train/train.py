import torch
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..models.sac_agent import SACAgent
from ..env.trading_env import TradingEnvironment
from ..utils.risk_management import RiskParams

def train_agent(
    config_path: str,
    price_data: pd.DataFrame,
    features: pd.DataFrame,
    save_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """
    Train SAC agent on trading environment.
    
    Args:
        config_path: Path to training config file
        price_data: Historical price data
        features: Engineered feature data
        save_dir: Directory to save models and logs
        device: Device to train on
        
    Returns:
        metrics: Dictionary of training metrics
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = TradingEnvironment(
        price_data=price_data,
        features=features,
        risk_params=RiskParams(**config["risk_params"]),
        window_size=config["window_size"],
        commission=config["commission"]
    )
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    agent = SACAgent(
        state_dim=state_dim,
        device=device,
        **config["agent_params"]
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(save_dir / "logs")
    
    # Initialize metrics
    metrics = {
        "episode_rewards": [],
        "portfolio_values": [1.0],  # Start with initial portfolio value
        "sharpe_ratios": [],
        "max_drawdowns": [],
        "critic_losses": [],
        "actor_losses": [],
        "alpha_losses": [],
        "alphas": []
    }
    
    # Training loop
    total_steps = 0
    for episode in tqdm(range(config["num_episodes"])):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Select action
            if total_steps < config["start_training_after_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent
            update_metrics = None
            if total_steps >= config["start_training_after_steps"]:
                update_metrics = agent.update_parameters()
            
            # Log metrics
            if update_metrics:
                for key, value in update_metrics.items():
                    writer.add_scalar(f"train/{key}", value, total_steps)
                    if key in ["critic_1_loss", "critic_2_loss"]:
                        metrics["critic_losses"].append(value)
                    elif key == "actor_loss":
                        metrics["actor_losses"].append(value)
                    elif key == "alpha_loss":
                        metrics["alpha_losses"].append(value)
                    elif key == "alpha":
                        metrics["alphas"].append(value)
            
            # Update portfolio value history
            metrics["portfolio_values"].append(float(info["portfolio_value"]))
            
            if done:
                break
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Save intermediate model
            if total_steps % config["save_interval"] == 0:
                agent.save(save_dir / f"model_step_{total_steps}.pt")
        
        # Log episode metrics
        metrics["episode_rewards"].append(episode_reward)
        
        # Calculate Sharpe ratio
        portfolio_values = np.array(metrics["portfolio_values"])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
            metrics["sharpe_ratios"].append(sharpe)
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max()
        metrics["max_drawdowns"].append(max_drawdown)
        
        # Log episode metrics
        writer.add_scalar("episode/reward", episode_reward, episode)
        writer.add_scalar("episode/portfolio_value", info["portfolio_value"], episode)
        writer.add_scalar("episode/sharpe_ratio", sharpe, episode)
        writer.add_scalar("episode/max_drawdown", max_drawdown, episode)
        
        # Early stopping based on Sharpe ratio
        if len(metrics["sharpe_ratios"]) > config["early_stopping_window"]:
            recent_sharpe = np.mean(metrics["sharpe_ratios"][-config["early_stopping_window"]:])
            if recent_sharpe > config["target_sharpe"]:
                print(f"Reached target Sharpe ratio of {config['target_sharpe']}")
                break
    
    # Save final model
    agent.save(save_dir / "model_final.pt")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        "episode": range(len(metrics["episode_rewards"])),
        "reward": metrics["episode_rewards"],
        "sharpe_ratio": metrics["sharpe_ratios"],
        "max_drawdown": metrics["max_drawdowns"],
        "portfolio_value": metrics["portfolio_values"][1:len(metrics["episode_rewards"])+1]  # Align with episodes
    })
    metrics_df.to_csv(save_dir / "training_metrics.csv", index=False)
    
    return metrics

def evaluate_agent(
    model_path: str,
    price_data: pd.DataFrame,
    features: pd.DataFrame,
    risk_params: Optional[RiskParams] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Evaluate trained agent on test data.
    
    Args:
        model_path: Path to saved model
        price_data: Historical price data
        features: Engineered feature data
        risk_params: Risk management parameters
        device: Device to evaluate on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Initialize environment
    env = TradingEnvironment(
        price_data=price_data,
        features=features,
        risk_params=risk_params
    )
    
    # Initialize agent with same hidden dimension as training (64)
    state_dim = env.observation_space.shape[0]
    agent = SACAgent(
        state_dim=state_dim,
        device=device,
        hidden_dim=64  # Match training hidden dimension
    )
    agent.load(model_path)
    
    # Evaluation loop
    state, _ = env.reset()
    done = False
    portfolio_values = [1.0]
    positions = []
    
    while not done:
        # Select action deterministically
        action = agent.select_action(state, evaluate=True)
        
        # Take step in environment
        next_state, _, done, _, info = env.step(action)
        
        state = next_state
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["position"])
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    metrics = {
        "final_portfolio_value": portfolio_values[-1],
        "total_return": (portfolio_values[-1] - 1.0) * 100,
        "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252),
        "max_drawdown": ((np.maximum.accumulate(portfolio_values) - portfolio_values) / 
                        np.maximum.accumulate(portfolio_values)).max(),
        "avg_position": np.mean(np.abs(positions)),
        "position_changes": np.sum(np.abs(np.diff(positions)))
    }
    
    return metrics 