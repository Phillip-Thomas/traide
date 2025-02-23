import torch
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
import sys
import logging
from tqdm.auto import tqdm  # Changed to tqdm.auto for better terminal compatibility
import copy

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
    disable_progress: bool = False,
    log_level: str = "INFO",
    episode_pbar: Optional[tqdm] = None
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
        log_level: Logging level (e.g. "INFO", "WARNING", "ERROR")
        episode_pbar: Optional external progress bar for episodes
        
    Returns:
        metrics: Dictionary of training metrics
    """
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create save directory and logs subdirectory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = save_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger with specified log level
        logger = TrainingLogger(
            log_dir=save_dir,
            experiment_name="sac_training",
            config=config,
            log_level=log_level
        )
        
        logger.logger.info("Initializing environment...")
        # Initialize environment
        env = TradingEnvironment(
            price_data=price_data,
            features=features,
            window_size=config["window_size"],
            commission=config["commission"],
            **config.get("env_params", {})  # Add environment parameters
        )
        
        logger.logger.info("Initializing agent...")
        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **config["agent_params"]
        )
        
        # Training loop
        total_steps = 0
        logger.logger.info("Starting training loop...")
        
        # Initialize metric tracking
        episode_metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
            "avg_positions": [],
            "total_costs": [],
            "volatilities": []
        }
        
        # Use external progress bar if provided, otherwise create our own
        if episode_pbar is not None:
            episode_iterator = range(config["num_episodes"])
        else:
            episode_pbar = tqdm(
                range(config["num_episodes"]),
                desc="Training episodes",
                disable=disable_progress,  # Use the disable_progress parameter
                dynamic_ncols=True,
                mininterval=1.0
            )
            episode_iterator = episode_pbar
        
        print("\nStarting training loop with configuration:")
        print(f"Number of episodes: {config['num_episodes']}")
        print(f"Window size: {config['window_size']}")
        print(f"Commission: {config['commission']}")
        print(f"Device: {device}\n")

        for episode in episode_iterator:
            print(f"\nStarting episode {episode}")
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps_in_episode = 0
            episode_costs = 0
            episode_positions = []
            
            while not done:
                # Select action
                with torch.no_grad():
                    action = agent.select_action(state, evaluate=False)

                try:
                    next_state, reward, done, _, info = env.step(action)
                    # print(f"Step {steps_in_episode}: Action={action[0]:.3f}, Reward={reward:.3f}, Portfolio={info['portfolio_value']:.3f}")
                    
                except Exception as e:
                    logger.logger.error(f"Error during environment step: {str(e)}")
                    break
                
                # Store transition in replay buffer
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train agent
                if len(agent.replay_buffer) > config["agent_params"]["batch_size"]:
                    try:
                        update_metrics = agent.update_parameters()
                        # Only log training metrics periodically
                        if total_steps % 1000 == 0:
                            logger.log_training_step(total_steps, update_metrics)
                    except Exception as e:
                        logger.logger.error(f"Error during agent update: {str(e)}")
                        break
                
                # Track episode statistics
                episode_reward += reward
                episode_costs += info['costs']
                episode_positions.append(abs(info['position']))
                
                state = next_state
                total_steps += 1
                steps_in_episode += 1
                
                # Update progress bar more frequently
                if steps_in_episode % 10 == 0:
                    postfix_dict = {
                        'reward': f'{episode_reward:.2f}',
                        'value': f'{info["portfolio_value"]:.2f}',
                        'pos': f'{abs(info["position"]):.2f}',
                        'vol': f'{info.get("volatility", 0):.2%}'
                    }
                    if episode_pbar:
                        episode_pbar.set_postfix(postfix_dict, refresh=True)
                        episode_pbar.update(0)
                
                # Save intermediate model
                if total_steps % config["save_interval"] == 0:
                    agent.save(save_dir / f"model_step_{total_steps}.pt")
                
                # Log detailed metrics
                if steps_in_episode % 100 == 0:
                    logger.logger.info(
                        f"Episode {episode} Step {steps_in_episode}: "
                        f"Portfolio Value: {info['portfolio_value']:.4f}, "
                        f"Return: {info['step_return']:.4f}%, "
                        f"Position: {abs(info['position']):.3f}, "
                        f"Volatility: {info.get('volatility', 0):.2%}"
                    )
            
            # Calculate episode statistics
            avg_position = np.mean(episode_positions) if episode_positions else 0
            final_portfolio_value = info['portfolio_value']
            volatility = info.get('volatility', 0)
            
            # Update episode metrics
            episode_metrics["episode_rewards"].append(episode_reward)
            episode_metrics["portfolio_values"].append(final_portfolio_value)
            episode_metrics["sharpe_ratios"].append(episode_reward / (volatility + 1e-6))
            episode_metrics["max_drawdowns"].append(max(0.0, 1.0 - final_portfolio_value))
            episode_metrics["avg_positions"].append(avg_position)
            episode_metrics["total_costs"].append(episode_costs)
            episode_metrics["volatilities"].append(volatility)
            
            # Log episode summary
            metrics = {
                "reward": episode_reward,
                "portfolio_value": final_portfolio_value,
                "sharpe_ratio": episode_reward / (volatility + 1e-6),
                "max_drawdown": max(0.0, 1.0 - final_portfolio_value),
                "avg_position": avg_position,
                "total_cost": episode_costs,
                "volatility": volatility
            }
            logger.log_episode(episode, metrics)
            
            # Update progress bar
            if episode_pbar:
                episode_pbar.set_postfix({
                    'reward': f'{episode_reward:.2f}',
                    'value': f'{final_portfolio_value:.2f}',
                    'pos': f'{avg_position:.2f}',
                    'vol': f'{volatility:.2%}'
                }, refresh=True)
                episode_pbar.update(1)
            
            # Clear GPU memory periodically
            if device.startswith('cuda') and episode % 10 == 0:
                torch.cuda.empty_cache()
        
        # Save final model and metrics
        agent.save(save_dir / "model_final.pt")
        logger.close()
        
        return episode_metrics
        
    except Exception as e:
        logger.logger.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        # Clean up resources
        if 'env' in locals():
            del env
        if 'agent' in locals():
            del agent
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

def evaluate_agent(
    env: TradingEnvironment,
    agent: SACAgent,
    n_episodes: int = 5,
    deterministic: bool = True,
    logger: Optional[TrainingLogger] = None
) -> Dict[str, float]:
    """Evaluate agent performance over multiple episodes."""
    try:
        # Create a separate evaluation environment
        eval_env = TradingEnvironment(
            price_data=env.price_data.copy(),
            features=env.features.copy(),
            risk_params=env.risk_manager.params,
            window_size=env.window_size,
            commission=env.commission
        )
        
        # Create evaluation copy of agent
        eval_agent = copy.deepcopy(agent)
        
        # Set networks to evaluation mode
        networks = [eval_agent.actor, eval_agent.critic_1, eval_agent.critic_2]
        for net in networks:
            if hasattr(net, 'eval'):
                net.eval()
        
        metrics = {
            'final_portfolio_value': [],
            'total_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'avg_position': [],
            'position_changes': []
        }
        
        for episode in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            positions = []
            returns = []
            portfolio_values = [1.0]  # Track portfolio value history
            
            while not done:
                with torch.no_grad():
                    # Get action and scale it down for evaluation
                    action = eval_agent.select_action(state, evaluate=True)  # Force deterministic
                    action = action * 0.5  # Scale down actions for more conservative evaluation
                
                next_state, reward, done, _, info = eval_env.step(action)
                
                # Track metrics
                positions.append(info['current_positions'])
                portfolio_values.append(info['portfolio_value'])
                if 'period_return' in info:
                    returns.append(info['period_return'])
                
                state = next_state
                
                if logger and len(portfolio_values) % 100 == 0:
                    logger.logger.debug(
                        f"Eval Episode {episode} Step {len(portfolio_values)}: "
                        f"Portfolio Value: {portfolio_values[-1]:.4f}"
                    )
            
            # Calculate episode metrics
            positions = np.array(positions)
            portfolio_values = np.array(portfolio_values)
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_dd = np.max(drawdown)
            
            metrics['final_portfolio_value'].append(portfolio_values[-1])
            metrics['total_return'].append((portfolio_values[-1] - 1.0) * 100)
            metrics['sharpe_ratio'].append(calculate_sharpe_ratio(eval_env))
            metrics['max_drawdown'].append(max_dd)
            metrics['avg_position'].append(np.mean(np.abs(positions)))
            metrics['position_changes'].append(np.sum(np.abs(np.diff(positions, axis=0))))
            
            if logger:
                logger.logger.info(
                    f"Evaluation Episode {episode} - "
                    f"Final Value: {portfolio_values[-1]:.4f}, "
                    f"Max DD: {max_dd:.2%}, "
                    f"Avg Pos: {np.mean(np.abs(positions)):.4f}"
                )
        
        # Set networks back to training mode
        for net in networks:
            if hasattr(net, 'train'):
                net.train()
                
        # Calculate final metrics with proper averaging
        final_metrics = {}
        for key in metrics:
            values = metrics[key]
            if not values:
                continue
                
            # Remove any None or NaN values
            valid_values = [v for v in values if v is not None and not np.isnan(v)]
            if not valid_values:
                final_metrics[key] = 0.0
                continue
                
            # Calculate mean and standard deviation
            mean_value = float(np.mean(valid_values))
            std_value = float(np.std(valid_values))
            
            # Store both mean and std for each metric
            final_metrics[key] = mean_value
            final_metrics[f"{key}_std"] = std_value
        
        return final_metrics
        
    except Exception as e:
        if logger:
            logger.logger.error(f"Error in evaluate_agent: {str(e)}")
        else:
            logging.error(f"Error in evaluate_agent: {str(e)}")
        raise

def calculate_sharpe_ratio(env: TradingEnvironment) -> float:
    """
    Calculate risk-adjusted Sharpe ratio from environment returns.
    Uses excess returns over risk-free rate and handles different trading frequencies.
    """
    try:
        returns = np.array(env.returns_history)
        if len(returns) < 2:
            return 0.0
            
        # Risk-free rate (assuming 2% annual rate)
        annual_rf_rate = 0.02
        # Convert to per-period rate (assuming daily trading)
        period_rf_rate = annual_rf_rate / 252
        
        # Calculate excess returns
        excess_returns = returns - period_rf_rate
        
        # Annualization factor (√252 for daily data)
        annual_factor = np.sqrt(252)
        
        # Calculate annualized mean and std of excess returns
        mean_excess_return = np.mean(excess_returns)
        return_std = np.std(excess_returns)
        
        # Handle edge cases
        if return_std < 1e-6 or np.isnan(return_std) or np.isnan(mean_excess_return):
            return 0.0
            
        # Calculate annualized Sharpe ratio
        sharpe = (mean_excess_return * annual_factor) / (return_std * np.sqrt(annual_factor))
        
        # Clip extreme values
        return float(np.clip(sharpe, -10.0, 10.0))
        
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0.0

def calculate_max_drawdown(env: TradingEnvironment) -> float:
    """Calculate maximum drawdown from portfolio value history."""
    try:
        portfolio_values = np.array(env.risk_manager.equity_curve)
        if len(portfolio_values) < 2:
            return 0.0
            
        # Calculate running maximum
        peak = np.maximum.accumulate(portfolio_values)
        # Calculate drawdowns
        drawdowns = (peak - portfolio_values) / peak
        # Get maximum drawdown
        max_dd = float(np.max(drawdowns))
        
        return max_dd if not np.isnan(max_dd) else 0.0
        
    except Exception as e:
        logging.error(f"Error calculating max drawdown: {str(e)}")
        return 0.0 