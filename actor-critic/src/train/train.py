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
from tqdm.auto import tqdm  # Changed to tqdm.auto for better terminal compatibility

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

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
    episode_pbar: Optional[tqdm] = None,
    tracer: Optional[trace.Tracer] = None
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
    # Initialize logger at the start
    logger = None
    if tracer is None:
        tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("train_agent") as span:
        try:
            # Create save directory and logs subdirectory
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = save_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize logger with specified log level
            logger = TrainingLogger(
                log_dir=save_dir,
                experiment_name="sac_training",
                config=config,
                log_level=log_level
            )
            
            with tracer.start_as_current_span("initialize_environment") as env_span:
                logger.logger.info("Initializing environment...")
                # Initialize environment
                env = TradingEnvironment(
                    price_data=price_data,
                    features=features,
                    risk_params=RiskParams(**config["risk_params"]),
                    window_size=config["window_size"],
                    commission=config["commission"]
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
            
            # Use deque with maxlen for histories
            from collections import deque
            
            # Modify environment to use bounded histories
            env.returns_history = deque(maxlen=10000)  # Or whatever reasonable limit
            
            # Training loop
            total_steps = 0
            logger.logger.info("Starting training loop...")
            
            # Use external progress bar if provided, otherwise create our own
            if episode_pbar is not None:
                episode_iterator = range(config["num_episodes"])
            else:
                episode_pbar = tqdm(
                    range(config["num_episodes"]),
                    desc=f"Training on {device}",
                    disable=disable_progress,
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=1.0
                )
                episode_iterator = episode_pbar
            
            for episode in episode_iterator:
                with tracer.start_as_current_span("episode") as episode_span:
                    state, _ = env.reset()
                    episode_reward = 0
                    episode_steps = 0

                    episode_span.set_attribute("episode", episode)
                    
                    # Add debugging information every 10 episodes
                    # if episode % 10 == 0:
                    #     print(f"\nEpisode {episode} Debug Info:")
                    #     print(f"Portfolio Value: {env.portfolio_value:.4f}")
                    #     print(f"Current Positions: {env.current_positions}")
                    #     # Store the action we're about to take instead of looking for last_action
                    #     with torch.no_grad():
                    #         action = agent.select_action(state, evaluate=False)
                    #     print(f"Next Action: {action}")
                    #     print(f"Recent Rewards: {env.recent_rewards[-5:] if hasattr(env, 'recent_rewards') else 'N/A'}")
                    
                    # Collect initial batch of random transitions
                    if episode == 0:
                        with tracer.start_as_current_span("collect_initial_transitions") as collect_span:
                            logger.logger.info("Collecting initial random transitions...")
                            initial_transitions = 0
                            max_attempts = config["start_training_after_steps"] * 2
                            
                            # Progress bar for initial transitions with minimal settings
                            with tqdm(
                                total=config["start_training_after_steps"],
                                desc="Initial transitions",
                                disable=disable_progress,
                                file=sys.stdout,  # Explicitly write to stdout
                                dynamic_ncols=True,  # Adapt to terminal width
                                mininterval=1.0,  # Update at most once per second
                            ) as pbar:
                                while initial_transitions < config["start_training_after_steps"] and max_attempts > 0:
                                    try:
                                        action = env.action_space.sample()
                                        next_state, reward, done, _, info = env.step(action)
                                        
                                        # Add transition to buffer
                                        agent.replay_buffer.add(state, action, reward, next_state, done)
                                        initial_transitions += 1
                                        
                                        # Force progress bar update
                                        pbar.update(1)
                                        pbar.refresh()
                                        
                                        # Update state
                                        if done:
                                            state, _ = env.reset()
                                        else:
                                            state = next_state
                                            
                                        total_steps += 1
                                        max_attempts -= 1
                                        
                                    except Exception as e:
                                        logger.logger.error(f"Error during initial transition collection: {str(e)}")
                                        max_attempts -= 1
                                        continue
                            
                            if initial_transitions < config["start_training_after_steps"]:
                                logger.logger.error(f"Failed to collect enough initial transitions. Only got {initial_transitions}")
                                raise RuntimeError("Failed to collect enough initial transitions")
                            
                            logger.logger.info(f"Successfully collected {initial_transitions} initial transitions")
                    
                    
                    # Training steps
                    steps_in_episode = 0
                    with tracer.start_as_current_span("training_steps") as training_span:
                        while True:
                            # Select action
                            with torch.no_grad():
                                action = agent.select_action(state, evaluate=False)
                            
                            # Debugging: Print the selected action
                            # print(f"Selected Action: {action}")
                            
                            # Take step in environment
                            try:
                                next_state, reward, done, _, info = env.step(action)
                                
                                # Debugging: Print the reward and portfolio value after taking the action
                                # print(f"Action Taken: {action}, Reward: {reward:.4f}, New Portfolio Value: {info['portfolio_value']:.4f}")
                                
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
                            
                            episode_reward += reward
                            state = next_state
                            total_steps += 1
                            episode_steps += 1
                            steps_in_episode += 1
                            
                            if steps_in_episode % 20 == 0:  
                                postfix_dict = {
                                    'reward': f'{episode_reward:.2f}',
                                    'value': f'{info["portfolio_value"]:.2f}',
                                    'steps': steps_in_episode
                                }
                                if episode_pbar:
                                    episode_pbar.set_postfix(postfix_dict, refresh=True)
                            
                            # Save intermediate model
                            if total_steps % config["save_interval"] == 0:
                                agent.save(save_dir / f"model_step_{total_steps}.pt")
                            
                            if done:
                                break
                        
                        # Final progress bar update for the episode
                        if episode_pbar:
                            episode_pbar.set_postfix({
                                'reward': f'{episode_reward:.2f}',
                                'value': f'{info["portfolio_value"]:.2f}',
                                'steps': steps_in_episode
                            }, refresh=True)
                            episode_pbar.update(1)  # Increment episode counter
                        
                        # Log episode metrics
                        metrics = {
                            "reward": episode_reward,
                            "portfolio_value": info["portfolio_value"],
                            "sharpe_ratio": calculate_sharpe_ratio(env),
                            "max_drawdown": calculate_max_drawdown(env)
                        }
                        logger.log_episode(episode, metrics)
                        
                        # Evaluate agent periodically
                        if episode % config["eval_interval"] == 0:
                            eval_metrics = evaluate_agent(
                                agent=agent,
                                env=env,
                                n_episodes=config["eval_episodes"],
                                tracer=tracer
                            )
                            logger.log_evaluation(eval_metrics, step=total_steps)
                            
                            # Early stopping based on Sharpe ratio
                            if len(logger.metrics["sharpe_ratios"]) > config["early_stopping_window"]:
                                recent_sharpe = np.mean(logger.metrics["sharpe_ratios"][-config["early_stopping_window"]:])
                                if recent_sharpe > config["target_sharpe"]:
                                    logger.logger.info(f"Reached target Sharpe ratio of {config['target_sharpe']}")
                                    break
                        
                        # Clear unnecessary tensors from GPU
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()
            
            # Save final model and metrics
            agent.save(save_dir / "model_final.pt")
            logger.close()
            
            return logger.metrics
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
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
    agent: Optional[SACAgent] = None,
    model_path: Optional[str] = None,
    env: Optional[TradingEnvironment] = None,
    price_data: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
    risk_params: Optional[RiskParams] = None,
    n_episodes: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tracer: Optional[trace.Tracer] = None
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
    with tracer.start_as_current_span("evaluate_agent") as span:
        try:
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
                # fail
                raise ValueError("Agent is not provided")
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                agent = SACAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    device=device,
                    hidden_dim=64  # Match training hidden dimension
                )
                agent.load(model_path)
            
            print("Evaluating agent...")
            
            # Evaluation loop
            portfolio_values = []
            positions = []
            returns = []
            
            for _ in range(n_episodes):
                with tracer.start_as_current_span("eval_episode") as episode_span:
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
                        episode_positions.append(info["current_positions"])
                    
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
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

def calculate_sharpe_ratio(env: TradingEnvironment) -> float:
    """Calculate Sharpe ratio from environment returns."""
    returns = np.array(env.returns_history)
    if len(returns) < 2:
        return 0.0
    return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

def calculate_max_drawdown(env: TradingEnvironment) -> float:
    """Calculate maximum drawdown from environment."""
    return env.risk_manager.current_drawdown 