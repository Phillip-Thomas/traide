import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from models.model import DQN
from environments.trading_env import SimpleTradeEnv
import random
import numpy as np
import math
import copy
import time
from memory.replay_buffer import PrioritizedReplayBuffer
from utils.save_utils import save_experiment, save_last_checkpoint, load_checkpoint
from utils.logger import TrainingLogger
from utils.visualizer import TrainingVisualizer
import multiprocessing as mp
from functools import partial
import torch.multiprocessing as tmp
from queue import Empty
from threading import Event

# Set multiprocessing start method to spawn for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

def compute_loss(q_values, next_q_values, batch, gamma):
    """
    Compute the loss with proper tensor dimensions.
    All input tensors should have same batch size.
    """
    batch_size = q_values.size(0)
    
    # Double Q-learning
    next_actions = next_q_values.max(1)[1]
    next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    target_q = batch['rewards'] + gamma * (1 - batch['dones']) * next_q
    
    # Current Q-values
    q_value = q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
    
    # Huber loss with importance sampling weights
    # Ensure weights tensor matches batch size
    weights = batch['weights'].expand(batch_size)
    loss = F.smooth_l1_loss(q_value, target_q.detach(), reduction='none')
    weighted_loss = (loss * weights).mean()
    
    return weighted_loss, loss

def train_batch(policy_net, target_net, optimizer, batch, batch_size, gamma, device, memory):
    """
    Modified to accept preprocessed batch directly and ensure GPU training
    """
    try:
        torch.cuda.set_device(device)
        
        # Get shapes
        batch_size, seq_len, feature_size = batch['states'].shape
        
        # Process sequences in chunks
        chunk_size = min(32, batch_size)
        total_loss = 0
        num_chunks = 0
        
        # Process chunks
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_slice = slice(i, end_idx)
            
            # Move data to GPU
            states = batch['states'][chunk_slice].reshape(-1, feature_size).to(device, non_blocking=True)
            next_states = batch['next_states'][chunk_slice].reshape(-1, feature_size).to(device, non_blocking=True)
            actions = batch['actions'][chunk_slice].reshape(-1).to(device, non_blocking=True)
            rewards = batch['rewards'][chunk_slice].reshape(-1).to(device, non_blocking=True)
            dones = batch['dones'][chunk_slice].reshape(-1).to(device, non_blocking=True)
            weights = batch['weights'][chunk_slice].repeat_interleave(seq_len).to(device, non_blocking=True)
            
            # N-step returns
            n_step = 3
            discounted_rewards = rewards.clone()
            for i in range(1, n_step):
                discounted_rewards += (gamma ** i) * rewards.roll(-i, dims=0) * (1 - dones.roll(-i, dims=0))

            with torch.amp.autocast('cuda'):
                # Forward pass on GPU
                with torch.no_grad():
                    next_q_values, _ = policy_net(next_states)
                    next_actions = next_q_values.max(1)[1]
                    target_q_values, _ = target_net(next_states)
                    next_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets = discounted_rewards + (gamma ** n_step) * (1 - dones) * next_q

                # Training step on GPU
                q_values, _ = policy_net(states)
                if random.random() < 0.1:
                    noise = torch.randn_like(q_values, device=device) * 0.1
                    q_values += noise
                
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_value, targets, reduction='none')
                weighted_loss = (loss * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += weighted_loss.item()
            num_chunks += 1

            # Update priorities (move to CPU for numpy operations)
            sequence_losses = loss.view(-1, seq_len).mean(dim=1)
            memory.update_priorities(batch['indices'][chunk_slice], sequence_losses.detach().cpu().numpy())

            # Clear GPU memory
            del states, next_states, actions, rewards, dones, weights, q_values, loss
            del next_q_values, next_actions, target_q_values, next_q, targets
            torch.cuda.empty_cache()

        torch.cuda.synchronize(device)
        return total_loss / num_chunks if num_chunks > 0 else 0

    except Exception as e:
        print(f"\nError in train_batch: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def run_env_episode_worker(env, policy_net_state_dict, epsilon, temperature, input_size, device_idx):
    """
    Worker process to run environment episodes
    """
    try:
        device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        transitions, reward, trades = run_env_episode(
            env=env,
            policy_net_state_dict=policy_net_state_dict,
            epsilon=epsilon,
            temperature=temperature,
            input_size=input_size,
            device=device
        )
        return transitions, reward, trades
    except Exception as e:
        print(f"Error in episode worker: {str(e)}")
        return None

def preprocess_batch_worker(memory, batch_size, seq_len, device_idx):
    """
    Worker process to prepare batches for GPU training
    """
    try:
        # Sample batch on CPU
        batch = memory.sample(batch_size, seq_len=seq_len, device='cpu')
        
        # Ensure all tensors are on CPU and detached
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].detach().cpu()
            elif isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key]).float()
        
        return device_idx, batch
    except Exception as e:
        print(f"Error in preprocess worker: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_process_pool(num_workers):
    """
    Create a process pool with proper CUDA initialization settings
    """
    ctx = mp.get_context('spawn')
    return ctx.Pool(num_workers)

def train_step_parallel(policy_nets, target_nets, optimizers, memory, batch_size, gamma, devices):
    """
    Run a single training step in parallel across all available GPUs with multi-CPU data preparation
    """
    try:
        # Use more CPU workers for preprocessing
        num_workers = min(mp.cpu_count(), 8)  # Use up to 8 CPU cores
        sub_batch_size = batch_size // (len(devices) * 2)  # Smaller batches for more parallelism
        
        # Process batches in parallel with more workers
        with create_process_pool(num_workers) as pool:
            # Submit more preprocessing jobs for better CPU utilization
            preprocessing_results = []
            for dev_idx in range(len(devices) * 2):  # Double the number of batches
                result = pool.apply_async(
                    preprocess_batch_worker,
                    (memory, sub_batch_size, 10, dev_idx % len(devices))
                )
                preprocessing_results.append(result)
            
            # Process results as they become available
            gpu_losses = []
            for result in preprocessing_results:
                try:
                    device_idx, batch = result.get(timeout=30.0)
                    if batch is None:
                        print("Skipping None batch")
                        continue
                        
                    device = devices[device_idx]
                    torch.cuda.set_device(device)
                    
                    # Move batch to GPU with non-blocking transfers
                    gpu_batch = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    
                    # Ensure models are on correct device
                    policy_nets[device_idx] = policy_nets[device_idx].to(device)
                    target_nets[device_idx] = target_nets[device_idx].to(device)
                    
                    policy_nets[device_idx].train()
                    target_nets[device_idx].train()
                    
                    loss = train_batch(
                        policy_nets[device_idx],
                        target_nets[device_idx],
                        optimizers[device_idx],
                        gpu_batch,
                        sub_batch_size,
                        gamma,
                        device,
                        memory
                    )
                    
                    if loss > 0:
                        gpu_losses.append(loss)
                    
                    # Asynchronous GPU operations
                    torch.cuda.current_stream(device).synchronize()
                    
                    # Clean up GPU memory
                    del gpu_batch
                    torch.cuda.empty_cache()
                    
                except mp.TimeoutError:
                    print(f"Timeout waiting for preprocessing result")
                    continue
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Average losses and synchronize models
        if gpu_losses:
            avg_loss = np.mean(gpu_losses)
            
            # Synchronize models using the first GPU as reference
            with torch.cuda.device(devices[0]), torch.no_grad():
                reference_state_dict = policy_nets[0].state_dict()
                
                # Update all other models
                for dev_idx in range(1, len(devices)):
                    torch.cuda.set_device(devices[dev_idx])
                    device_state_dict = {
                        k: v.to(devices[dev_idx], non_blocking=True) 
                        for k, v in reference_state_dict.items()
                    }
                    policy_nets[dev_idx].load_state_dict(device_state_dict)
                    target_nets[dev_idx].load_state_dict(device_state_dict)
                    torch.cuda.synchronize(devices[dev_idx])
            
            return avg_loss
        return 0
        
    except Exception as e:
        print(f"\nError in train_step_parallel: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def get_optimizer(model, lr=1e-4):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Initial restart interval
        T_mult=2,  # Multiply interval by 2 after each restart
        eta_min=1e-6
    )
    
    return optimizer, scheduler

def validate_model(model, env, device):
    """
    Validate the model on a single environment with memory optimization
    """
    state = env.reset()
    done = False
    initial_value = env.cash
    trades = []
    entry_price = None
    accumulated_reward = 0
    
    with torch.no_grad():
        while not done:
            # Convert state to tensor and move to device
            state_tensor = torch.from_numpy(np.array(state)).float().to(device)
            
            # Get valid actions and Q-values
            valid_actions = env.get_valid_actions()
            q_values, _ = model(state_tensor)  # Returns (1, n_actions)
            q_values = q_values.squeeze(0)  # Convert to (n_actions)
            
            # Mask invalid actions
            mask = torch.full_like(q_values, float('-inf'))
            for a in valid_actions:
                mask[a] = q_values[a]
            action = torch.argmax(mask).item()
            
            # Track trades
            if action == 1 and env.position == 0:
                entry_price = env.close_prices[env.idx]
            elif action == 2 and env.position == 1:
                exit_price = env.close_prices[env.idx]
                trades.append((exit_price - entry_price) / entry_price)
                entry_price = None
            
            # Take step in environment
            state, reward, done = env.step(action)
            accumulated_reward += reward
            
            # Clear unnecessary tensors
            del state_tensor, q_values, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate final portfolio value
    final_value = env.cash if env.position == 0 else env.shares * env.close_prices[env.idx]
    buy_and_hold_return = (env.close_prices[-1] - env.close_prices[env.window_size]) / env.close_prices[env.window_size]
    
    return {
        'profit': (final_value - initial_value) / initial_value,
        'num_trades': len(trades),
        'avg_return': np.mean(trades) if trades else 0,
        'win_rate': np.mean([t > 0 for t in trades]) if trades else 0,
        'accumulated_reward': accumulated_reward,
        'buy_and_hold_return': buy_and_hold_return
    }

def store_transitions_worker(transitions_chunk, device_idx):
    """
    Worker process to store transitions in memory with proper CPU/GPU handling
    """
    try:
        device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        # Process transitions on CPU first
        processed_transitions = []
        for transition in transitions_chunk:
            # Ensure tensors are on CPU and detached
            if isinstance(transition[0], torch.Tensor):
                state = transition[0].detach().cpu().numpy()
                next_state = transition[3].detach().cpu().numpy()
            else:
                state = transition[0]
                next_state = transition[3]
            
            processed_transitions.append((
                state,
                transition[1],  # action
                transition[2],  # reward
                next_state,
                transition[4]   # done
            ))
        return processed_transitions
    except Exception as e:
        print(f"Error in store transitions worker: {str(e)}")
        return None

def parallel_store_transitions(all_transitions, num_workers, devices):
    """
    Store transitions in parallel across multiple processes with proper CPU/GPU handling
    """
    if not all_transitions:
        return []
        
    # Calculate optimal chunk size
    chunk_size = max(1, min(len(all_transitions) // num_workers, 1000))
    chunks = [all_transitions[i:i + chunk_size] for i in range(0, len(all_transitions), chunk_size)]
    
    results = []
    with create_process_pool(num_workers) as pool:
        for idx, chunk in enumerate(chunks):
            device_idx = idx % len(devices)
            result = pool.apply_async(store_transitions_worker, (chunk, device_idx))
            results.append(result)
            
        stored_transitions = []
        for result in results:
            try:
                transitions = result.get(timeout=30.0)
                if transitions is not None:
                    # Keep transitions as numpy arrays until needed
                    stored_transitions.extend(transitions)
            except Exception as e:
                print(f"Error collecting stored transitions: {str(e)}")
                continue
                
    return stored_transitions

def run_env_episode(env, policy_net_state_dict, epsilon, temperature, input_size, device='cpu', max_steps=1000):
    """
    Run a single episode in an environment with proper CPU/GPU handling
    """
    try:
        # Create a new policy network instance
        policy_net = DQN(input_size).to(device)
        policy_net.load_state_dict(policy_net_state_dict)
        policy_net.eval()
        
        state = env.reset()
        if state is None:
            raise ValueError("Environment reset returned None state")
            
        state_size = len(state)
        if state_size != input_size:
            raise ValueError(f"State size mismatch. Expected {input_size}, got {state_size}")
            
        done = False
        steps = 0
        transitions = []
        episode_reward = 0
        trades_info = []
        entry_price = None
        
        while not done and steps < max_steps:
            try:
                # Convert state to tensor and move to device
                state_tensor = torch.from_numpy(np.array(state)).float().to(device)
                
                if random.random() < epsilon:
                    action = random.choice(env.get_valid_actions())
                else:
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        q_values, _ = policy_net(state_tensor.unsqueeze(0))
                        q_values = q_values.squeeze() / temperature
                        valid_actions = env.get_valid_actions()
                        mask = torch.full_like(q_values, float('-inf'))
                        for a in valid_actions:
                            mask[a] = q_values[a]
                        action = torch.argmax(mask).item()
                
                # Track trades
                if action == 1 and env.position == 0:
                    entry_price = env.close_prices[env.idx]
                elif action == 2 and env.position == 1:
                    exit_price = env.close_prices[env.idx]
                    trades_info.append((exit_price - entry_price) / entry_price)
                    entry_price = None
                    
                next_state, reward, done = env.step(action)
                if next_state is None:
                    raise ValueError("Environment step returned None state")
                
                # Store state as numpy array to avoid CUDA IPC issues
                transitions.append((
                    state_tensor.detach().cpu().numpy(),
                    action,
                    reward,
                    np.array(next_state, dtype=np.float32),
                    done
                ))
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Clean up GPU tensors
                del state_tensor
                torch.cuda.empty_cache()
                    
            except Exception as step_error:
                print(f"Error during environment step: {str(step_error)}")
                raise
        
        # Clean up
        del policy_net
        torch.cuda.empty_cache()
            
        return transitions, episode_reward, trades_info

    except Exception as e:
        print(f"Error in worker process: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def calculate_std(rewards):
    """
    Standalone function for calculating standard deviation
    """
    return np.std(rewards) if rewards else 0.0

def process_validation_worker(policy_net_state_dict, val_env, device_idx, input_size):
    """
    Worker process to run validation on a single environment
    """
    try:
        device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        policy_net = DQN(input_size).to(device)
        policy_net.load_state_dict(policy_net_state_dict)
        return validate_model(policy_net, val_env, device)
    except Exception as e:
        print(f"Error in validation worker: {str(e)}")
        return None

def parallel_validation(policy_net, val_envs, devices, input_size):
    """
    Run validation in parallel across multiple processes
    """
    results = []
    with create_process_pool(len(devices)) as pool:
        for idx, (val_ticker, val_env) in enumerate(val_envs.items()):
            device_idx = idx % len(devices)
            result = pool.apply_async(
                process_validation_worker,
                (policy_net.state_dict(), val_env, device_idx, input_size)
            )
            results.append((val_ticker, result))
            
        val_metrics_all = []
        for val_ticker, result in results:
            try:
                metrics = result.get(timeout=30.0)
                if metrics is not None:
                    val_metrics_all.append((val_ticker, metrics))
            except Exception as e:
                print(f"Error collecting validation result: {str(e)}")
                
    return val_metrics_all

def create_trade_env(data, window_size):
    """Helper function to create trading environment"""
    return SimpleTradeEnv(data, window_size=window_size)

def train_dqn(train_data_dict, val_data_dict, input_size, n_episodes=1000, batch_size=32, gamma=0.99, 
              initial_best_profit=float('-inf'), initial_best_excess=float('-inf')):
    
    print("\nInitializing training...")
    logger = TrainingLogger()
    
    # Set up multi-GPU if available
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} CUDA devices")
        for i in range(n_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            # Set device properties for better performance
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
        main_device = devices[0]
    else:
        print("Using CPU for training")
        devices = [torch.device('cpu')]
        main_device = devices[0]
        
    # Enable cuDNN benchmarking and deterministic operations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for better performance
    
    # Initialize networks for each GPU
    print("Initializing networks...")
    policy_nets = []
    target_nets = []
    optimizers = []
    
    for device in devices:
        policy_net = DQN(input_size).to(device)
        target_net = DQN(input_size).to(device)
        
        if len(policy_nets) > 0:  # Copy weights from first model
            policy_net.load_state_dict(policy_nets[0].state_dict())
            target_net.load_state_dict(target_nets[0].state_dict())
            
        target_net.load_state_dict(policy_net.state_dict())
        optimizer, _ = get_optimizer(policy_net, lr=3e-4)
        
        policy_nets.append(policy_net)
        target_nets.append(target_net)
        optimizers.append(optimizer)
        
        print(f"Initialized network on {device}")
    
    # Use a larger memory buffer for better sampling
    memory = PrioritizedReplayBuffer(100000)  # Increased from 50000
    
    # Training state
    window_size = 48
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 100
    best_val_profit = initial_best_profit
    best_excess_return = initial_best_excess
    best_model = None
    last_avg_loss = 0
    
    # Environment setup with parallel initialization
    print("Setting up environments...")
    num_cpu_workers = min(mp.cpu_count(), 16)  # Use up to 16 CPU cores
    with create_process_pool(num_cpu_workers) as pool:
        # Initialize environments in parallel
        train_env_futures = [
            pool.apply_async(create_trade_env, (data, window_size))
            for data in train_data_dict.values()
        ]
        val_env_futures = [
            pool.apply_async(create_trade_env, (data, window_size))
            for data in val_data_dict.values()
        ]
        
        train_envs = {
            ticker: future.get() 
            for ticker, future in zip(train_data_dict.keys(), train_env_futures)
        }
        val_envs = {
            ticker: future.get() 
            for ticker, future in zip(val_data_dict.keys(), val_env_futures)
        }
    
    print(f"Created {len(train_envs)} training environments and {len(val_envs)} validation environments")
    
    # Training loop state
    episodes_without_improvement = 0
    reset_count = 0
    max_resets = 30
    
    try:
        print("\nStarting training loop...")
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            episode_start = time.time()
            
            # Sample tickers for this episode
            batch_size_env = min(len(devices) * 4, batch_size)  # Ensure each GPU gets at least 4 environments
            episode_tickers = random.sample(list(train_envs.keys()), k=min(batch_size_env, len(train_envs)))
            print(f"Selected tickers: {episode_tickers}")
            
            # Calculate epsilon and temperature
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                     math.exp(-1. * episode / epsilon_decay)
            temperature = max(0.5, 1.0 - episode / n_episodes)
            
            # Distribute tickers across GPUs
            all_transitions = []
            episode_reward = 0
            trades_info = []
            
            # Ensure even distribution across GPUs
            tickers_per_device = math.ceil(len(episode_tickers) / len(devices))
            device_tickers = [[] for _ in devices]
            for i, ticker in enumerate(episode_tickers):
                device_idx = i % len(devices)
                device_tickers[device_idx].append(ticker)
            
            # Process environments in parallel with better CPU utilization
            num_env_workers = min(mp.cpu_count(), 32)  # Use up to 32 CPU cores for environment processing
            with create_process_pool(num_env_workers) as pool:
                futures = []
                
                # Submit environment processing jobs in smaller batches
                for dev_idx, device_ticker_list in enumerate(device_tickers):
                    for ticker in device_ticker_list:
                        future = pool.apply_async(
                            run_env_episode_worker,
                            (train_envs[ticker], 
                             policy_nets[dev_idx].state_dict(),
                             epsilon, temperature, input_size, dev_idx)
                        )
                        futures.append(future)
                
                # Collect results with better error handling
                collected_transitions = []
                collected_rewards = []
                collected_trades = []
                
                for future in futures:
                    try:
                        result = future.get(timeout=30.0)
                        if result is not None:
                            transitions, reward, trades = result
                            collected_transitions.extend(transitions)
                            collected_rewards.append(reward)
                            collected_trades.extend(trades)
                    except mp.TimeoutError:
                        print(f"Timeout waiting for environment result")
                        continue
                    except Exception as e:
                        print(f"Error collecting result: {str(e)}")
                        continue
                
                all_transitions = collected_transitions
                episode_reward = sum(collected_rewards)
                trades_info = collected_trades
            
            print(f"Collected {len(all_transitions)} total transitions")
            
            # Store experience and train with parallel processing
            if all_transitions:
                print("Storing transitions in memory...")
                num_store_workers = min(mp.cpu_count(), 16)  # Use up to 16 CPU cores for storage
                stored_transitions = parallel_store_transitions(all_transitions, num_store_workers, devices)
                if stored_transitions:
                    # Process transitions in parallel batches
                    batch_size = 1000  # Process in smaller batches
                    for i in range(0, len(stored_transitions), batch_size):
                        batch = stored_transitions[i:i + batch_size]
                        processed_transitions = []
                        for state, action, reward, next_state, done in batch:
                            processed_transitions.append((
                                torch.from_numpy(state).float(),  # Store on CPU
                                action,
                                reward,
                                torch.from_numpy(next_state).float(),  # Store on CPU
                                done
                            ))
                        memory.push_episode(processed_transitions)

            if len(memory) >= batch_size:
                print("Training on collected experience...")
                losses = []
                
                # Multiple training steps per episode with parallel processing
                num_train_workers = min(mp.cpu_count(), 8)  # Use up to 8 CPU cores for training
                train_steps = 4
                
                for step in range(train_steps):
                    print(f"Training step {step + 1}/{train_steps}")
                    loss = train_step_parallel(
                        policy_nets,
                        target_nets,
                        optimizers,
                        memory,
                        batch_size,
                        gamma,
                        devices
                    )
                    if loss > 0:
                        losses.append(loss)
                        
                if losses:
                    last_avg_loss = np.mean(losses)
                    print(f"Average loss: {last_avg_loss:.6f}")
            
            # Calculate episode statistics in parallel
            episode_duration = time.time() - episode_start
            with create_process_pool(min(4, mp.cpu_count())) as pool:
                returns_std = pool.apply_async(
                    calculate_std,
                    ([t[2] for t in all_transitions],)
                ).get(timeout=30.0)
            
            print(f"Episode completed in {episode_duration:.2f} seconds")
            
            # Log episode results
            print("Logging episode results...")
            logger.log_episode(
                episode_num=episode,
                returns=episode_reward,
                length=len(all_transitions),
                std_dev=returns_std,
                priority=memory.priorities[-1] if memory.priorities else 0,
                epsilon=epsilon,
                loss=last_avg_loss,
                trades_info=trades_info
            )
            
            # Periodic validation and model updates
            if episode % 5 == 0:
                target_nets[0].load_state_dict(policy_nets[0].state_dict())
                policy_nets[0].eval()
                
                # Validate in parallel
                val_metrics_results = parallel_validation(policy_nets[0], val_envs, devices, input_size)
                
                # Calculate average metrics
                val_metrics_all = [metrics for _, metrics in val_metrics_results]
                avg_profit = np.mean([m['profit'] for m in val_metrics_all])
                avg_win_rate = np.mean([m['win_rate'] for m in val_metrics_all])
                avg_num_trades = np.mean([m['num_trades'] for m in val_metrics_all])
                avg_excess_return = np.mean([(m['profit'] - m['buy_and_hold_return']) * 100 
                                           for m in val_metrics_all])
                
                # Log validation results
                validation_metrics = {
                    'profit': avg_profit,
                    'win_rate': avg_win_rate,
                    'num_trades': int(avg_num_trades),
                    'excess_return': avg_excess_return,
                    'per_ticker_metrics': {t: m for t, m in val_metrics_results}
                }
                logger.log_validation(episode, validation_metrics)
                
                # Model selection and early stopping
                if avg_excess_return > best_excess_return:
                    logger.log_model_save(episode, "checkpoints/best_model.pt", validation_metrics)
                    best_excess_return = avg_excess_return
                    best_val_profit = avg_profit
                    best_model = copy.deepcopy(policy_nets[0])
                    episodes_without_improvement = 0
                    
                    # Save experiment
                    save_experiment(
                        model=best_model,
                        optimizer=optimizers[0],
                        metrics=validation_metrics
                    )
                else:
                    episodes_without_improvement += 1
                
                # Log learning rate
                logger.log_training_update(
                    episode,
                    learning_rate=optimizers[0].param_groups[0]['lr'],
                    grad_norm=torch.nn.utils.clip_grad_norm_(policy_nets[0].parameters(), max_norm=1.0).item()
                )
                
                # Reset on plateau
                if episodes_without_improvement >= 50:  # patience
                    reset_count += 1
                    if reset_count >= max_resets:
                        print("Maximum resets reached. Stopping training.")
                        break
                    
                    if best_model is not None:
                        for dev_idx in range(len(devices)):
                            policy_nets[dev_idx].load_state_dict(best_model.state_dict())
                            target_nets[dev_idx].load_state_dict(best_model.state_dict())
                    
                    optimizer, _ = get_optimizer(policy_nets[0], lr=3e-4 * (0.9 ** reset_count))
                    memory = PrioritizedReplayBuffer(100000)
                    epsilon = epsilon_start
                    episodes_without_improvement = 0
                    
                    logger.log_training_update(episode, optimizer.param_groups[0]['lr'])
                    print(f"\nReset {reset_count}/{max_resets} complete.")
                    continue
                
                policy_nets[0].train()
                optimizers[0].step()
    
        # Generate final visualizations and report
        visualizer = TrainingVisualizer(logger.metrics_file)
        visualizer.plot_training_progress()
        visualizer.plot_trade_distribution()
        visualizer.generate_summary_report()
        
        return {
            'final_model': policy_nets[0],
            'best_model': best_model,
            'training_summary': logger.get_summary_stats()
        }

    except Exception as e:
        print(f"\nError in train_dqn: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'final_model': policy_nets[0],
            'best_model': best_model,
            'training_summary': logger.get_summary_stats()
        }
