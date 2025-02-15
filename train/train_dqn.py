import random
import copy
import os
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np
import math
from model import DQN, ReplayBuffer, train_batch, validate_model, save_new_global_best


# Update the train_dqn function to use the new checkpoint saving
def train_dqn(train_data_dict, val_data_dict, input_size, n_episodes=1000, batch_size=32, gamma=0.99, optimizer=None, initial_best_profit=float('-inf'), initial_best_excess=float('-inf')):

    def quick_evaluate_seed(seed, train_data_dict, val_data_dict, input_size):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize model and run one episode
        policy_net = DQN(input_size).to(device)
        
        # Run validation
        val_metrics_all = []
        for val_ticker, val_env in val_envs.items():
            val_metrics = validate_model(policy_net, val_env, device)
            val_metrics_all.append(val_metrics)
        
        avg_excess_return = np.mean([(m['profit'] - m['buy_and_hold_return']) * 100 for m in val_metrics_all])
        return avg_excess_return

    def find_good_seed(threshold=-10.0, max_attempts=100):
        print("Searching for promising seed...")
        for attempt in range(max_attempts):
            seed = random.randint(0, 2**32)
            excess_return = quick_evaluate_seed(seed, train_data_dict, val_data_dict, input_size)
            print(f"Seed {seed}: {excess_return:.2f}% excess return")
            
            if excess_return > threshold:
                print(f"Found promising seed {seed} with {excess_return:.2f}% excess return")
                return seed
                
        print(f"No seed found above threshold after {max_attempts} attempts")
        return None
    
    good_seed = find_good_seed()
    if good_seed:
        random.seed(good_seed)
        torch.manual_seed(good_seed)
        np.random.seed(good_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_size).to(device)
    target_net = DQN(input_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    if optimizer is None:
        print(f"optimizer: new")
        optimizer = optim.Adam(policy_net.parameters(), lr=0.0000005)
    else:
        print(f"optimizer: {optimizer}")
        optimizer = optimizer

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    memory = ReplayBuffer(100000)
    window_size = 48
    epsilon = 1.0
    best_val_profit = initial_best_profit
    best_excess_return = initial_best_excess  # Add this line
    best_model = None

    # Create environments for all tickers
    train_envs = {
        ticker: SimpleTradeEnv(data, window_size=window_size) 
        for ticker, data in train_data_dict.items()
    }
    val_envs = {
        ticker: SimpleTradeEnv(data, window_size=window_size)
        for ticker, data in val_data_dict.items()
    }
    
    # List to store average excess return for each episode
    episode_excess_returns = []

    # Add patience tracking
    patience = 50  # Episodes to wait before reset
    episodes_without_improvement = 0
    reset_count = 0
    max_resets = 50  # Maximum number of resets before stopping

    for episode in range(n_episodes):
        # Randomly select a ticker for this episode
        ticker = random.choice(list(train_envs.keys()))
        train_env = train_envs[ticker]
                               
        state = train_env.reset()
        done = False

        episode_transitions = []
        
        while not done:
            state_tensor = (
                torch.from_numpy(np.array(state))  # shape (241,)
                .float()
                .unsqueeze(0)                      # -> (1, 241)
                .unsqueeze(0)                      # -> (1, 1, 241)
                .to(device)
            )

            valid_actions = train_env.get_valid_actions()

            if random.random() < epsilon:
                # random among valid actions
                action = random.choice(valid_actions)
            else:
                # mask out invalid actions from Q-values
                q_values, _ = policy_net(state_tensor)  # shape [1,3]
                q_values = q_values.squeeze(0)         # shape [3]
                # Then apply your mask
                mask = torch.full_like(q_values, float('-inf'))
                for a in valid_actions:
                    mask[a] = q_values[a]     

                action = torch.argmax(mask).item()
            
            next_state, reward, done = train_env.step(action)
            episode_transitions.append((state, action, reward, next_state, done))
            state = next_state
                
        memory.push_episode(episode_transitions)

        if len(memory) >= batch_size:
                train_batch(policy_net, target_net, optimizer, memory, batch_size, gamma, device)

        
        if episode % 1 == 0:

            target_net.load_state_dict(policy_net.state_dict())
            policy_net.eval()
            
            print(f"\nEpisode {episode + 1}")

            # Validate on all tickers
            val_metrics_all = []
            for val_ticker, val_env in val_envs.items():
                val_metrics = validate_model(policy_net, val_env, device)
                val_metrics_all.append(val_metrics)
                # print(f"\n{val_ticker} Metrics:")
                # print(f"  Profit: {val_metrics['profit']*100:.2f}%")
                # print(f"  Win Rate: {val_metrics['win_rate']*100:.1f}%")
                # print(f"  Trades: {val_metrics['num_trades']}")
                # print(f"  Avg Return: {val_metrics['avg_return']*100:.2f}%")
                # print(f"  Buy & Hold Return: {val_metrics['buy_and_hold_return']*100:.2f}%")
            
            # Calculate average metrics across all tickers
            avg_profit = np.mean([m['profit'] for m in val_metrics_all])
            avg_win_rate = np.mean([m['win_rate'] for m in val_metrics_all])
            avg_num_trades = np.mean([m['num_trades'] for m in val_metrics_all])
            # the average percent difference between profit and buy and hold return
            avg_excess_return = np.mean([(m['profit'] - m['buy_and_hold_return']) * 100 for m in val_metrics_all])
            avg_accumulated_reward = np.mean([m['accumulated_reward'] for m in val_metrics_all])

            print(f"\nAverage Metrics Across All Tickers:")
            print(f"  Accumulated Reward: {avg_accumulated_reward}")
            print(f"  Profit: {avg_profit*100:.2f}%")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%")
            print(f"  Trades: {avg_num_trades}")
            print(f"  Avg Excess Return: {avg_excess_return:.2f}%")
            
            # Save this episode's avg excess retur
            episode_excess_returns.append(avg_excess_return)


            if avg_excess_return > best_excess_return:
                best_val_profit = avg_profit
                best_excess_return = avg_excess_return
                best_model = copy.deepcopy(policy_net)
                episodes_without_improvement = 0 
                save_new_global_best(best_model, optimizer, {
                    'profit': avg_profit,
                    'win_rate': avg_win_rate,
                    'excess_return': avg_excess_return,  # Add this line
                    'per_ticker_metrics': {t: m for t, m in zip(val_envs.keys(), val_metrics_all)}
                }, best_val_profit)
                print(f"\nNew best model! Episode {episode + 1}, Excess Return: {avg_excess_return:.2f}%")
            else:
                episodes_without_improvement += 1
            
            if episodes_without_improvement >= patience:
                reset_count += 1
                print(f"\nNo improvement for {patience} episodes. Performing reset {reset_count}/{max_resets}")
                
                if reset_count >= max_resets:
                    print("Maximum resets reached. Stopping training.")
                    break
                    
                # Soft reset: Keep best weights but reset other training components
                if best_model is not None:
                    policy_net.load_state_dict(best_model.state_dict())
                    target_net.load_state_dict(best_model.state_dict())
                
                # Reset training components
                epsilon = 1.0  # Reset exploration
                memory = ReplayBuffer(100000)  # Fresh replay buffer
                optimizer = optim.Adam(policy_net.parameters(), lr=0.0000005)  # Fresh optimizer
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
                
                # Reset counter
                episodes_without_improvement = 0
                
                # Change random seed
                new_seed = random.randint(0, 1000000000)
                random.seed(new_seed)
                torch.manual_seed(new_seed)
                np.random.seed(new_seed)
                
                print(f"Reset complete. New seed: {new_seed}")
                continue

            policy_net.train()


        
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995  # More gradual decay

        # Replace the epsilon update logic with:
        if episode <10:  # Force more exploration in first few episodes
            epsilon = 1.0
        else:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                    math.exp(-1. * (episode - 10) / 20)

        scheduler.step()
    
    # After training, write all episode excess returns to a single file with a timestamp in the filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join("results", f"episode_excess_returns_{timestamp}.txt")
    with open(results_file, "w") as f:
        for ep, excess in enumerate(episode_excess_returns, start=1):
            f.write(f"Episode {ep}: Average Excess Return: {excess:.2f}%\n")
    print(f"Saved episode results to {results_file}")

    return {'final_model': policy_net, 'best_model': best_model}
