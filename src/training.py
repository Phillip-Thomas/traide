import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from datetime import datetime
import shutil
from pathlib import Path
import json
import numpy as np
import math
import random

from .models import (
    OptimizedDQN, ReplayBuffer, compute_q_loss,
    select_actions_gpu, create_optimized_tensors
)
from .environment import (
    SimpleTradeEnvGPU, create_environments_batch_parallel,
    step_environments_chunk
)
from .utils import get_device, get_optimal_batch_size, setup_distributed, cleanup_distributed

# Define curriculum stages globally
CURRICULUM_STAGES = [
    {
        'name': 'Basic',  # Simple trends, low volatility periods
        'episode_length_range': (50, 100),  # Shorter episodes
        'volatility_threshold': 0.01,  # Only low volatility periods
        'reward_scale': 1.5,  # Higher rewards to encourage learning
        'required_profit': 0.05  # Need 5% profit to advance
    },
    {
        'name': 'Intermediate',  # Mixed trends, medium volatility
        'episode_length_range': (100, 200),
        'volatility_threshold': 0.02,
        'reward_scale': 1.2,
        'required_profit': 0.10
    },
    {
        'name': 'Advanced',  # All market conditions
        'episode_length_range': (200, 371),  # Full length episodes
        'volatility_threshold': None,  # No filtering
        'reward_scale': 1.0,
        'required_profit': None
    }
]

def train_step(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    """Execute a single training step"""
    # Sample batch from memory
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Get current Q values and auxiliary outputs
    current_q_values, (features, parallel_outputs) = policy_net(states)
    
    # Get Q values for chosen actions
    state_action_values = current_q_values.gather(1, actions.unsqueeze(1))
    
    with torch.no_grad():
        # Get max Q values for next states from target model
        next_q_values, _ = target_net(next_states)
        next_state_values = next_q_values.max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = rewards + gamma * next_state_values * (~dones)
    
    # Compute Huber loss for Q-values
    q_loss = torch.nn.functional.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )
    
    # Add regularization loss for auxiliary outputs
    aux_loss = 0.0001 * (
        features.pow(2).mean() +  # L2 regularization on features
        sum(output.pow(2).mean() for output in parallel_outputs)  # L2 regularization on parallel outputs
    )
    
    # Combine losses
    loss = q_loss + aux_loss
    
    # Optimize the model
    loss.backward()
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Update target network periodically
    if hasattr(policy_net, 'module'):  # Handle DDP case
        policy_net_state_dict = policy_net.module.state_dict()
        target_net_state_dict = target_net.module.state_dict()
    else:
        policy_net_state_dict = policy_net.state_dict()
        target_net_state_dict = target_net.state_dict()
    
    for target_param, policy_param in zip(target_net_state_dict.values(), policy_net_state_dict.values()):
        target_param.data.copy_(0.005 * policy_param.data + 0.995 * target_param.data)
    
    return loss.item()

def train_dqn(rank, world_size, train_data_dict, val_data_dict, input_size, window_size, n_episodes=1000, gamma=0.99):
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(rank)
        else:
            device = torch.device("cpu")

        # Initialize process group only if using distributed
        is_distributed = world_size > 1 and device.type == "cuda"
        if is_distributed:
            setup_distributed(rank, world_size)

        print(f"\n{'='*50}")
        print(f"Starting Training on GPU {rank}")
        print(f"{'='*50}")

        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        # Initialize batch sizes
        unique_samples_per_ticker = 371  # 397 records - 26 window size
        num_tickers = 12
        total_unique_samples = unique_samples_per_ticker * num_tickers
        
        # Set batch sizes based on available data
        batch_size = min(4096, total_unique_samples // 4)  # Keep batch size <= 25% of unique samples
        env_batch_size = min(32, total_unique_samples // 100)  # Keep parallel envs <= 1% of unique samples
        replay_buffer_size = min(100000, total_unique_samples * 2)  # Keep buffer size reasonable
        
        print(f"\nData Constraints:")
        print(f"  Unique samples per ticker: {unique_samples_per_ticker}")
        print(f"  Total unique samples: {total_unique_samples}")
        
        print(f"\nBatch Configuration:")
        print(f"  Training Batch Size: {batch_size}")
        print(f"  Environment Batch Size: {env_batch_size}")
        print(f"  Replay Buffer Size: {replay_buffer_size}")
        print(f"  Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")

        # Initialize curriculum learning variables
        current_stage = 0
        stage_episodes = 0
        stage_profits = []
        min_episodes_per_stage = 50  # Minimum episodes before considering advancement
        
        def should_advance_stage(stage_profits, current_stage):
            if len(stage_profits) < min_episodes_per_stage:
                return False
            
            # Calculate recent performance
            recent_profits = stage_profits[-20:]  # Last 20 episodes
            avg_profit = sum(recent_profits) / len(recent_profits)
            
            # Check if we meet advancement criteria
            stage_config = CURRICULUM_STAGES[current_stage]
            if stage_config['required_profit'] is None:
                return False  # Final stage
            
            return avg_profit >= stage_config['required_profit']
        
        print("\nCurriculum Learning Configuration:")
        for i, stage in enumerate(CURRICULUM_STAGES):
            print(f"\nStage {i+1}: {stage['name']}")
            print(f"  Episode Length: {stage['episode_length_range']}")
            print(f"  Volatility Threshold: {stage['volatility_threshold']}")
            print(f"  Reward Scale: {stage['reward_scale']}x")
            print(f"  Required Profit: {stage['required_profit']*100 if stage['required_profit'] else 'N/A'}%")
        
        # Initialize environments for first stage
        env_pool = create_curriculum_environments(
            train_data_dict,
            env_batch_size,
            CURRICULUM_STAGES[current_stage],
            device,
            window_size
        )
        print(f"Created {len(env_pool)} environments")

        # Initialize models
        policy_net = OptimizedDQN(input_size).to(device)
        target_net = OptimizedDQN(input_size).to(device)

        if is_distributed:
            policy_net = DDP(policy_net, device_ids=[rank], find_unused_parameters=True)
            target_net = DDP(target_net, device_ids=[rank], find_unused_parameters=True)

        # Initialize optimizer with a lower learning rate
        learning_rate = 0.0001
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        
        # Try to load the latest checkpoint
        start_episode = 0
        best_profit = float('-inf')
        best_path = checkpoint_dir / "best_model.pt"
        latest_path = checkpoint_dir / "latest_model.pt"
        
        if latest_path.exists():
            print("\nLoading latest checkpoint...")
            checkpoint = torch.load(latest_path, map_location=device)
            if is_distributed:
                policy_net.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            current_stage = checkpoint.get('curriculum_stage', 0)  # Load curriculum stage
            stage_profits = checkpoint.get('stage_profits', [])  # Load stage profits
            print(f"Resuming from episode {start_episode} (Stage: {CURRICULUM_STAGES[current_stage]['name']})")
            
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            best_profit = checkpoint['profit']
            print(f"Previous best profit: {best_profit*100:.2f}%")
        
        # Initialize target network with policy network's state
        if is_distributed:
            target_net.module.load_state_dict(policy_net.module.state_dict())
        else:
            target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # Set target network to evaluation mode

        memory = ReplayBuffer(replay_buffer_size, input_size, device)
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes

        # Modified epsilon decay strategy
        epsilon_start = 0.95
        epsilon_end = 0.05
        epsilon_warmup = int(0.1 * n_episodes)  # 10% of episodes for warmup
        epsilon_decay = int(0.6 * n_episodes)   # 60% of episodes for decay
        
        def get_epsilon(episode):
            if episode < epsilon_warmup:
                return epsilon_start
            elif episode < (epsilon_warmup + epsilon_decay):
                # Exponential decay during decay phase
                progress = (episode - epsilon_warmup) / epsilon_decay
                return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-5 * progress)
            else:
                return epsilon_end

        # Initialize episode tracking variables
        episode_reward = 0
        trades_this_episode = 0
        total_profit = 0
        total_loss = 0
        total_steps = 0  # Initialize total steps counter

        for episode in range(start_episode, n_episodes):
            episode_start = time.time()
            
            # Check if we should advance to next curriculum stage
            if should_advance_stage(stage_profits, current_stage):
                current_stage = min(current_stage + 1, len(CURRICULUM_STAGES) - 1)
                stage_episodes = 0
                stage_profits = []
                print(f"\nAdvancing to {CURRICULUM_STAGES[current_stage]['name']} stage!")
            
            stage_config = CURRICULUM_STAGES[current_stage]
            reward_scale = stage_config['reward_scale']

            # Get epsilon for this episode
            epsilon = get_epsilon(episode)

            # Process all environments in the pool
            pool_start = time.time()
            print(f"\nProcessing environment pool:")
            print(f"{'-'*40}")
            
            # Reset environments with random starting points
            states = torch.stack([env.reset(random_start=True) for env in env_pool])
            if states.device != device:
                states = states.to(device)
            
            step_count = 0
            pool_done = False
            
            # Track pool statistics
            pool_reward = 0
            pool_trades = 0
            pool_profit = 0
            pool_loss = 0
            training_steps = 0

            # Print episode configuration
            print("\nEpisode Configuration:")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Start Indices: {[env.start_idx for env in env_pool]}")
            print(f"  Episode Lengths: {[env.episode_length for env in env_pool]}")

            while not pool_done:
                step_count += 1
                total_steps += 1
                
                # Get actions from policy network
                with torch.no_grad():
                    # Ensure states are properly batched
                    if len(states.shape) == 1:
                        states = states.unsqueeze(0)
                    q_values, _ = policy_net(states)  # Unpack the tuple, we only need q_values
                    actions = select_actions_gpu(q_values, epsilon, device)
                
                # Step environments and get results
                next_states, rewards, dones = step_environments_chunk((env_pool, actions))
                
                # Apply curriculum reward scaling
                rewards = rewards * reward_scale
                
                # Ensure tensors are on correct device
                if next_states.device != device:
                    next_states = next_states.to(device)
                
                # Store transitions in replay buffer
                memory.push_batch(states, actions, rewards, next_states, dones)
                
                # Update statistics
                step_reward = rewards.sum().item()
                step_trades = (actions != 0).sum().item()
                profit_mask = rewards > 0
                step_profit = rewards[profit_mask].sum().item() if profit_mask.any() else 0
                
                pool_reward += step_reward
                pool_trades += step_trades
                pool_profit += step_profit
                
                # Training step if we have enough samples
                if len(memory) >= batch_size:
                    training_steps += 1
                    loss = train_step(policy_net, target_net, memory, optimizer, batch_size, gamma, device)
                    pool_loss += loss
                    
                    if training_steps % 100 == 0:
                        avg_loss = pool_loss / training_steps
                        print(f"\nTraining Status (Step {step_count}):")
                        print(f"  Avg Loss: {avg_loss:.6f}")
                        print(f"  Pool Reward: {pool_reward:.2f}")
                        print(f"  Pool Profit: {pool_profit:.2f}")
                        print(f"  Pool Trades: {pool_trades}")
                        print(f"  Memory Size: {len(memory)}")
                
                # Print step statistics periodically
                if step_count % 100 == 0:
                    steps_per_sec = step_count / (time.time() - pool_start)
                    print(f"\nStep {step_count}:")
                    print(f"  Reward: {step_reward:.4f}")
                    print(f"  Trades: {step_trades}")
                    print(f"  Profit: {step_profit:.4f}")
                    print(f"  Steps/sec: {steps_per_sec:.2f}")
                
                # Update states and check done condition
                states = next_states
                pool_done = dones.all().item()

                # Save checkpoint periodically
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save latest checkpoint
                    latest_checkpoint = {
                        'episode': episode,
                        'model_state_dict': policy_net.state_dict() if not is_distributed else policy_net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'profit': pool_profit,
                        'reward': pool_reward,
                        'trades': pool_trades,
                        'loss': avg_loss,
                        'epsilon': epsilon,
                        'curriculum_stage': current_stage,
                        'stage_profits': stage_profits
                    }
                    torch.save(latest_checkpoint, latest_path)
                    print(f"\nSaved latest checkpoint")
                    
                    # Save timestamped checkpoint
                    checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.pt"
                    torch.save(latest_checkpoint, checkpoint_path)
                    print(f"Saved backup checkpoint to {checkpoint_path}")
                    
                    last_save_time = current_time
            
            # Update episode statistics
            episode_reward += pool_reward
            trades_this_episode += pool_trades
            total_profit += pool_profit
            total_loss += pool_loss
            
            # Print pool summary
            pool_time = time.time() - pool_start
            avg_loss = pool_loss / max(1, training_steps)
            
            # Calculate per-environment statistics
            trades_per_env = pool_trades / len(env_pool)
            avg_episode_length = sum(env.episode_length for env in env_pool) / len(env_pool)
            trade_frequency = trades_per_env / avg_episode_length if avg_episode_length > 0 else 0
            
            print(f"\nPool Complete:")
            print(f"  Time: {pool_time:.2f}s")
            print(f"  Total Steps: {step_count}")
            print(f"  Total Reward: {pool_reward:.2f}")
            print(f"  Total Profit: {pool_profit*100:.2f}%")
            print(f"\nTrading Activity:")
            print(f"  Total Trades: {pool_trades}")
            print(f"  Trades per Environment: {trades_per_env:.1f}")
            print(f"  Avg Episode Length: {avg_episode_length:.1f} periods (15-min candles)")
            print(f"  Trade Frequency: {trade_frequency*100:.1f}% of periods")
            print(f"  Trading Rate: {trades_per_env/(avg_episode_length*0.25):.1f} trades per hour")
            print(f"\nTraining:")
            print(f"  Avg Loss: {avg_loss:.6f}")
            print(f"  Epsilon: {epsilon:.3f} ({epsilon*100:.1f}% random actions)")

            # Save best model if we have a new best
            if pool_profit > best_profit:
                best_profit = pool_profit
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_checkpoint = {
                    'episode': episode,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'profit': best_profit,
                    'reward': pool_reward,
                    'trades': pool_trades,
                    'loss': avg_loss
                }
                best_path = checkpoint_dir / "best_model.pt"
                torch.save(best_checkpoint, best_path)
                print(f"\nNew best model saved! Profit: {best_profit:.2f}")

    except Exception as e:
        print(f"Error on GPU {rank}: {str(e)}")
        raise e
    finally:
        if is_distributed:
            cleanup_distributed()
        if device.type == "cuda":
            torch.cuda.empty_cache()

def validate_model(model, env, device):
    state = env.reset()
    done = False
    initial_value = env.cash
    trades = []
    entry_price = None
    accumulated_reward = 0

    # Calculate buy and hold return
    buy_and_hold_shares = initial_value / env.raw_data['Close'][env.window_size]
    buy_and_hold_value = buy_and_hold_shares * env.raw_data['Close'][-1]
    buy_and_hold_return = (buy_and_hold_value - initial_value) / initial_value

    while not done:
        state_tensor = (
            torch.from_numpy(np.array(state))  # shape (241,)
            .float()
            .unsqueeze(0)                      # -> (1, 241)
            .unsqueeze(0)                      # -> (1, 1, 241)
            .to(device)
        )

        valid_actions = env.get_valid_actions()
        q_values = model(state_tensor)[0]

        q_values, _ = model(state_tensor)  # shape = [1, 3]
        q_values = q_values.squeeze(0)     # now shape = [3]

        mask = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            mask[a] = q_values[a]
        action = torch.argmax(mask).item()

        if action == 1 and env.position == 0:
            entry_price = env.raw_data['Close'][env.idx]
        elif action == 2 and env.position == 1:
            exit_price = env.raw_data['Close'][env.idx]
            trades.append((exit_price - entry_price) / entry_price)
            entry_price = None

        state, reward, done = env.step(action)
        accumulated_reward += reward

    final_value = env.cash if env.position == 0 else env.shares * env.raw_data['Close'][env.idx]
    return {
        'profit': (final_value - initial_value) / initial_value,
        'num_trades': len(trades),
        'avg_return': np.mean(trades) if trades else 0,
        'win_rate': np.mean([t > 0 for t in trades]) if trades else 0,
        'accumulated_reward': accumulated_reward,
        'buy_and_hold_return': buy_and_hold_return
    }

def save_experiment(model, optimizer, metrics, seed, avg_loss, current_file_path):
    """
    Save experiment results including model checkpoint, current code state, and metrics log.

    Args:
        model: The PyTorch model
        optimizer: The optimizer
        metrics: Dictionary containing performance metrics
        seed: The random seed used
        avg_loss: The average training loss
        current_file_path: Path to the current running script
    """
    # Create timestamp for experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # 1. Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'seed': seed,
        'timestamp': timestamp,
    }
    checkpoint_path = os.path.join(experiment_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)

    # 2. Copy current model.py file
    try:
        shutil.copy2(current_file_path, os.path.join(experiment_dir, "model.py"))
    except Exception as e:
        print(f"Warning: Could not copy model.py file: {e}")

    # 3. Create and save detailed log
    log_content = f"""Experiment Log - {timestamp}
        ===============================

        Model Performance Metrics
        -----------------------
        Average Excess Return: {metrics['excess_return']:.2f}%
        Average Profit: {metrics['profit']*100:.2f}%
        Average Win Rate: {metrics['win_rate']*100:.1f}%
        Average Number of Trades: {metrics.get('num_trades', 'N/A')}
        Accumulated Reward: {metrics.get('accumulated_reward', 'N/A')}

        Training Parameters
        ------------------
        Initial Seed: {seed}
        Learning Rate: {optimizer.param_groups[0]['lr']:.8f}
        Average Loss: {avg_loss:.8f}

        Per-Ticker Performance
        ---------------------
        """

    # Add per-ticker metrics if available
    if 'per_ticker_metrics' in metrics:
        for ticker, ticker_metrics in metrics['per_ticker_metrics'].items():
            log_content += f"\n{ticker}:\n"
            log_content += f"  Profit: {ticker_metrics['profit']*100:.2f}%\n"
            log_content += f"  Win Rate: {ticker_metrics['win_rate']*100:.1f}%\n"
            log_content += f"  Trades: {ticker_metrics['num_trades']}\n"
            log_content += f"  Excess Return: {(ticker_metrics['profit'] - ticker_metrics['buy_and_hold_return'])*100:.2f}%\n"

    # Save log file
    log_path = os.path.join(experiment_dir, "experiment_log.txt")
    with open(log_path, "w") as f:
        f.write(log_content)

    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Log: {log_path}")
    print(f"  - Model Code: {os.path.join(experiment_dir, 'model.py')}")

    return experiment_dir

def save_new_global_best(model, optimizer, metrics, global_best_profit):
    # We assume metrics['profit'] > global_best_profit

    # Make a timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build checkpoint object
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': timestamp,
    }

    # 2a) Save to a timestamped folder
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[SAVE] New best checkpoint => {ckpt_path}")

    # 2b) Also overwrite the best_checkpoint.pt
    torch.save(checkpoint, "checkpoints/best_checkpoint.pt")
    print(f"[SAVE] Overwrote global best => checkpoints/best_checkpoint.pt")

    return ckpt_path

def create_curriculum_environments(train_data_dict, batch_size, stage_config, device, window_size):
    min_length, max_length = stage_config['episode_length_range']
    volatility_threshold = stage_config['volatility_threshold']
    
    # Filter periods based on volatility if threshold is set
    if volatility_threshold is not None:
        filtered_data = {}
        for ticker, data in train_data_dict.items():
            # Calculate rolling volatility
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            # Select low volatility periods
            valid_indices = volatility <= volatility_threshold
            if sum(valid_indices) > max_length:
                filtered_data[ticker] = {
                    col: data[col][valid_indices] 
                    for col in data.columns
                }
            else:
                filtered_data[ticker] = data  # Use all data if not enough low vol periods
    else:
        filtered_data = train_data_dict
    
    # Create environments with curriculum-specific settings
    env_pool = create_environments_batch_parallel(
        filtered_data,
        window_size,
        batch_size,
        device
    )
    
    # Modify each environment's episode length to be within the curriculum range
    for env in env_pool:
        env.episode_length = random.randint(min_length, max_length)
    
    return env_pool 
