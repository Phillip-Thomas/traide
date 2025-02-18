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
from ..utils.logger import TrainingLogger

def train_agent(
    config_path: str,
    price_data: pd.DataFrame,
    features: pd.DataFrame,
    save_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    disable_progress: bool = False
) -> Dict[str, List[float]]:
    """
    Train SAC agent on trading environment.
    
    Args:
        config_path: Path to training config file
        price_data: Historical price data
        features: Engineered feature data
        save_dir: Directory to save models and logs
        device: Device to train on
        disable_progress: Whether to disable progress bars (useful for parallel runs)
        
    Returns:
        metrics: Dictionary of training metrics
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create save directory and logs subdirectory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
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
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **config["agent_params"]
    )
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=save_dir,
        experiment_name="sac_training",
        config=config
    )
    
    # Training loop
    total_steps = 0
    for episode in tqdm(range(config["num_episodes"]), disable=disable_progress, desc=f"Training on {device}", position=1, leave=False):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Collect initial batch of random transitions
        if episode == 0:
            for _ in range(config["start_training_after_steps"]):
                action = env.action_space.sample()
                next_state, reward, done, _, info = env.step(action)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state if not done else env.reset()[0]
                total_steps += 1
        
        while True:
            # Select action
            with torch.no_grad():
                action = agent.select_action(state, evaluate=False)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.replay_buffer) > config["agent_params"]["batch_size"]:
                update_metrics = agent.update_parameters()
                logger.log_training_step(total_steps, update_metrics)
            
            episode_reward += reward
            state = next_state
            total_steps += 1
            episode_steps += 1
            
            # Save intermediate model
            if total_steps % config["save_interval"] == 0:
                agent.save(save_dir / f"model_step_{total_steps}.pt")
            
            if done:
                break
        
        # Log episode metrics
        metrics = {
            "reward": episode_reward,
            "portfolio_value": info["portfolio_value"],
            "sharpe_ratio": calculate_sharpe_ratio(env),
            "max_drawdown": calculate_max_drawdown(env)
        }
        logger.log_episode(episode, metrics)
        
        # Evaluate agent
        if episode % config["eval_interval"] == 0:
            eval_metrics = evaluate_agent(
                agent=agent,
                env=env,
                n_episodes=config["eval_episodes"]
            )
            logger.log_evaluation(eval_metrics, step=total_steps)
            
            # Early stopping based on Sharpe ratio
            if len(logger.metrics["sharpe_ratios"]) > config["early_stopping_window"]:
                recent_sharpe = np.mean(logger.metrics["sharpe_ratios"][-config["early_stopping_window"]:])
                if recent_sharpe > config["target_sharpe"]:
                    print(f"Reached target Sharpe ratio of {config['target_sharpe']}")
                    break
    
    # Save final model and metrics
    agent.save(save_dir / "model_final.pt")
    logger.close()
    
    return logger.metrics

def evaluate_agent(
    agent: Optional[SACAgent] = None,
    model_path: Optional[str] = None,
    env: Optional[TradingEnvironment] = None,
    price_data: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
    risk_params: Optional[RiskParams] = None,
    n_episodes: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Evaluate trained agent on test data.
    
    Args:
        agent: Trained agent (optional if model_path provided)
        model_path: Path to saved model (optional if agent provided)
        env: Trading environment (optional if price_data and features provided)
        price_data: Historical price data (optional if env provided)
        features: Engineered feature data (optional if env provided)
        risk_params: Risk management parameters (optional)
        n_episodes: Number of evaluation episodes
        device: Device to evaluate on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if agent is None and model_path is None:
        raise ValueError("Either agent or model_path must be provided")
    
    if env is None and (price_data is None or features is None):
        raise ValueError("Either env or price_data and features must be provided")
    
    # Initialize environment if not provided
    if env is None:
        env = TradingEnvironment(
            price_data=price_data,
            features=features,
            risk_params=risk_params
        )
    
    # Initialize agent if not provided
    if agent is None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            hidden_dim=64  # Match training hidden dimension
        )
        agent.load(model_path)
    
    # Evaluation loop
    portfolio_values = []
    positions = []
    returns = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_portfolio = [1.0]
        episode_positions = []
        
        while not done:
            # Select action deterministically
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, _, done, _, info = env.step(action)
            
            state = next_state
            episode_portfolio.append(info["portfolio_value"])
            episode_positions.append(info["positions"])
        
        portfolio_values.extend(episode_portfolio)
        positions.extend(episode_positions)
        returns.extend(np.diff(episode_portfolio) / episode_portfolio[:-1])
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    positions = np.array(positions)
    returns = np.array(returns)
    
    metrics = {
        "final_portfolio_value": portfolio_values[-1],
        "total_return": (portfolio_values[-1] - 1.0) * 100,
        "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252),
        "max_drawdown": ((np.maximum.accumulate(portfolio_values) - portfolio_values) / 
                        np.maximum.accumulate(portfolio_values)).max(),
        "avg_position": np.mean(np.abs(positions)),
        "position_changes": np.sum(np.abs(np.diff(positions, axis=0)))
    }
    
    return metrics

def calculate_sharpe_ratio(env: TradingEnvironment) -> float:
    """Calculate Sharpe ratio from environment returns."""
    returns = np.array(env.returns_history)
    if len(returns) < 2:
        return 0.0
    return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

def calculate_max_drawdown(env: TradingEnvironment) -> float:
    """Calculate maximum drawdown from environment."""
    return env.risk_manager.current_drawdown 