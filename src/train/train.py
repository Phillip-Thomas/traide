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

def train_batch(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    try:
        batch = memory.sample(batch_size, seq_len=10, device=device)
        if batch is None:
            return 0

        # Get shapes
        batch_size, seq_len, feature_size = batch['states'].shape
        
        # Process sequences in parallel
        states = batch['states'].reshape(-1, feature_size)
        next_states = batch['next_states'].reshape(-1, feature_size)
        actions = batch['actions'].reshape(-1)
        rewards = batch['rewards'].reshape(-1)
        dones = batch['dones'].reshape(-1)

        # N-step returns for better reward propagation
        n_step = 3
        discounted_rewards = rewards.clone()
        for i in range(1, n_step):
            discounted_rewards += (gamma ** i) * rewards.roll(-i, dims=0) * (1 - dones.roll(-i, dims=0))

        # Double Q-learning with target network
        with torch.no_grad():
            # Get actions from policy network
            next_q_values, _ = policy_net(next_states)
            next_actions = next_q_values.max(1)[1]
            
            # Get Q-values from target network
            target_q_values, _ = target_net(next_states)
            next_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target with discounted rewards
            targets = discounted_rewards + (gamma ** n_step) * (1 - dones) * next_q

        # Current Q-values with added noise for exploration
        q_values, _ = policy_net(states)
        if random.random() < 0.1:  # Add noise occasionally
            noise = torch.randn_like(q_values) * 0.1
            q_values += noise
            
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Prioritized replay with importance sampling
        weights = batch['weights'].repeat_interleave(seq_len)
        
        # Huber loss for stability
        loss = F.smooth_l1_loss(q_value, targets, reduction='none')
        weighted_loss = (loss * weights).mean()

        # Optimize with gradient clipping
        optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        # Update priorities using mean sequence loss
        sequence_losses = loss.view(batch_size, seq_len).mean(dim=1)
        memory.update_priorities(batch['indices'], sequence_losses.detach().cpu().numpy())

        return weighted_loss.item()

    except Exception as e:
        print(f"\nError in train_batch: {type(e).__name__}: {str(e)}")
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
    Validate the model on a single environment
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

def train_dqn(train_data_dict, val_data_dict, input_size, n_episodes=1000, batch_size=32, gamma=0.99, 
              initial_best_profit=float('-inf'), initial_best_excess=float('-inf')):
    
    # Initialize logger
    logger = TrainingLogger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Initialize networks and optimizer
    policy_net = DQN(input_size).to(device)
    target_net = DQN(input_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer, scheduler = get_optimizer(policy_net, lr=3e-4)
    memory = PrioritizedReplayBuffer(50000)
    
    # Training state
    window_size = 48
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 100
    best_val_profit = initial_best_profit
    best_excess_return = initial_best_excess
    best_model = None
    last_avg_loss = 0
    
    # Environment setup
    train_envs = {ticker: SimpleTradeEnv(data, window_size=window_size) 
                  for ticker, data in train_data_dict.items()}
    val_envs = {ticker: SimpleTradeEnv(data, window_size=window_size)
                for ticker, data in val_data_dict.items()}
    
    # Training loop state
    episodes_without_improvement = 0
    reset_count = 0
    max_resets = 30
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # Sample tickers for this episode
        episode_tickers = random.sample(list(train_envs.keys()), k=min(batch_size, len(train_envs)))
        all_transitions = []
        episode_reward = 0
        trades_info = []
        
        # Collect experience
        for ticker in episode_tickers:
            env = train_envs[ticker]
            state = env.reset()
            done = False
            steps = 0
            max_steps = 1000
            
            while not done and steps < max_steps:
                # Epsilon-greedy with temperature
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                         math.exp(-1. * episode / epsilon_decay)
                temperature = max(0.5, 1.0 - episode / n_episodes)
                
                state_tensor = torch.from_numpy(np.array(state)).float().to(device)
                
                if random.random() < epsilon:
                    action = random.choice(env.get_valid_actions())
                else:
                    with torch.no_grad():
                        q_values, _ = policy_net(state_tensor)
                        # Add unsqueeze to ensure proper dimensions
                        if len(q_values.shape) == 1:
                            q_values = q_values.unsqueeze(0)
                        q_values = q_values.squeeze()  # Ensure shape is [num_actions]
                        q_values = q_values / temperature
                        valid_actions = env.get_valid_actions()
                        # Create mask with correct size
                        mask = torch.full((3,), float('-inf'), device=device)  # Assuming 3 actions: 0,1,2
                        for a in valid_actions:
                            mask[a] = q_values[a]
                        action = torch.argmax(mask).item()
                
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                # Track trade information
                if action in [1, 2]:  # Buy or Sell
                    current_price = env.close_prices[env.idx]
                    if action == 2 and env.position == 1:  # Sell
                        profit = (current_price - env.entry_price) / env.entry_price * 100
                        trades_info.append(profit)
                
                all_transitions.append((state, action, reward, next_state, done))
                state = next_state
                steps += 1
            
        # Store experience and train
        if all_transitions:
            memory.push_episode(all_transitions)
            
        if len(memory) >= batch_size:
            losses = []
            for _ in range(4):  # Multiple training steps per episode
                loss = train_batch(policy_net, target_net, optimizer, memory, 
                                 batch_size, gamma, device)
                if loss > 0:  # Only track valid losses
                    losses.append(loss)
            if losses:
                last_avg_loss = np.mean(losses)
        
        # Calculate episode statistics
        episode_duration = time.time() - episode_start
        returns_std = np.std([t[2] for t in all_transitions])  # Standard deviation of rewards
        
        # Log episode results
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
            target_net.load_state_dict(policy_net.state_dict())
            policy_net.eval()
            
            # Validate
            val_metrics_all = []
            for val_ticker, val_env in val_envs.items():
                val_metrics = validate_model(policy_net, val_env, device)
                val_metrics_all.append(val_metrics)
            
            # Calculate average metrics
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
                # 'per_ticker_metrics': {t: m for t, m in zip(val_envs.keys(), val_metrics_all)}
            }
            logger.log_validation(episode, validation_metrics)
            
            # Model selection and early stopping
            if avg_excess_return > best_excess_return:
                logger.log_model_save(episode, "checkpoints/best_model.pt", validation_metrics)
                best_excess_return = avg_excess_return
                best_val_profit = avg_profit
                best_model = copy.deepcopy(policy_net)
                episodes_without_improvement = 0
                
                # Save experiment
                save_experiment(
                    model=best_model,
                    optimizer=optimizer,
                    metrics=validation_metrics
                )
            else:
                episodes_without_improvement += 1
            
            # Log learning rate
            logger.log_training_update(
                episode,
                learning_rate=optimizer.param_groups[0]['lr'],
                grad_norm=torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0).item()
            )
            
            # Reset on plateau
            if episodes_without_improvement >= 50:  # patience
                reset_count += 1
                if reset_count >= max_resets:
                    print("Maximum resets reached. Stopping training.")
                    break
                
                if best_model is not None:
                    policy_net.load_state_dict(best_model.state_dict())
                    target_net.load_state_dict(best_model.state_dict())
                
                optimizer, scheduler = get_optimizer(policy_net, lr=3e-4 * (0.9 ** reset_count))
                memory = PrioritizedReplayBuffer(50000)
                epsilon = epsilon_start
                episodes_without_improvement = 0
                
                logger.log_training_update(episode, optimizer.param_groups[0]['lr'])
                print(f"\nReset {reset_count}/{max_resets} complete.")
                continue
            
            policy_net.train()
            scheduler.step()
    
    # Generate final visualizations and report
    visualizer = TrainingVisualizer(logger.metrics_file)
    visualizer.plot_training_progress()
    visualizer.plot_trade_distribution()
    visualizer.generate_summary_report()
    
    return {
        'final_model': policy_net,
        'best_model': best_model,
        'training_summary': logger.get_summary_stats()
    }