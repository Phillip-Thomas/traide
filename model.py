import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
from datetime import datetime
from typing import List, Tuple
import random
import copy

def get_market_data_multi(tickers=None, period="730d", interval="1h"):
    """
    Fetch data for multiple tickers
    """
    if tickers is None:
        tickers = [
            "SPY",
            "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "EEM", "GLD", "TLT",
            "VTI", "VOO", "IWB", "IJH", "IJR", "EFA", "VNQ", "DBC", "USO", "AGG",
            "BND", "LQD", "MBB", "TIP", "SHY", "IEI", "IJNK", "EMB", "IAU", "GDX",
            "SLV", "PDBC", "ICLN", "KRE", "XLY", "XLC", "XLI", "XLB", "XLV", "XLU",
            "XBI", "SMH", "SOXX", "VGT", "ARKK", "ARKG", "ARKQ", "ARKW", "TAN", "ICF"
        ]

    
    data_dict = {}
    for symbol in tickers:
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if not df.empty and len(df) > 100:  # Ensure sufficient data
                print(f"Successfully fetched {len(df)} records for {symbol}")
                data_dict[symbol] = df
            else:
                print(f"Insufficient data for {symbol}")
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
    
    return data_dict

class DQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        
        self.value_net = nn.Sequential(
            nn.Linear(16, 3)
        )
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, h=None):
        B, T, D = x.shape
        x_2d = x.view(B*T, D)
        features_2d = self.feature_net(x_2d)
        features_3d = features_2d.view(B, T, 32)
        
        lstm_out, (h_n, c_n) = self.lstm(features_3d, h)
        out = lstm_out[:, -1, :]
        
        return self.value_net(out), (h_n, c_n)


class SimpleTradeEnv:
    def __init__(self, data, window_size=10):
        self.data = data
        
        # Store raw prices for Open, High, Low, and Close
        self.raw_open = self.data['Open'].values
        self.raw_high = self.data['High'].values
        self.raw_low = self.data['Low'].values
        self.raw_close = self.data['Close'].values
        self.raw_volume = self.data['Volume'].values
        self.window_size = window_size
        
        # Normalize all prices using the Close price at the starting point
        scale = self.raw_close[window_size] if self.raw_close[window_size] != 0 else 1.0
        self.open_prices = self.raw_open / scale
        self.high_prices = self.raw_high / scale
        self.low_prices = self.raw_low / scale
        self.close_prices = self.raw_close / scale

        scale_vol = self.raw_volume[window_size] if self.raw_volume[window_size] != 0 else 1.0
        self.volume = self.raw_volume / scale_vol
        
        self.reset()

    def reset(self):
        self.idx = self.window_size
        self.position = 0
        self.cash = 1.0  # Normalized initial cash
        self.shares = 0
        self.entry_price = None
        self.max_portfolio_value = self.cash
        self.baseline_price = self.open_prices[self.idx - self.window_size]
        return self._get_state()
    
    def get_valid_actions(self):
        """
        Returns a list of valid action indices given the current position.
        """
        if self.position == 0:
            return [0, 1]  # HOLD, BUY
        else:
            return [0, 2]  # HOLD, SELL

    def _get_state(self):
        # Create windows for each price type over the specified window_size
        window_open = self.open_prices[self.idx - self.window_size:self.idx]
        window_high = self.high_prices[self.idx - self.window_size:self.idx]
        window_low  = self.low_prices[self.idx - self.window_size:self.idx]
        window_close = self.close_prices[self.idx - self.window_size:self.idx]
        window_volume = self.volume[self.idx - self.window_size:self.idx]
        position_info = np.array([float(self.position)])
        
        # Concatenate all raw data windows into a single state vector
        state = np.concatenate([window_open, window_high, window_low, window_close, window_volume, position_info])
        return state


    def _get_reward(self, action):
        reward = 0
        if (action == 1 and self.position == 1) or (action == 2 and self.position == 0):
            return -0.001

        current_price = self.close_prices[self.idx]
        buying_fee = 0.001
        selling_fee = 0.0015
        
        # Calculate trend over multiple timeframes
        short_window = min(10, self.idx)
        long_window = min(20, self.idx)
        
        short_trend = (current_price - self.close_prices[self.idx - short_window]) / self.close_prices[self.idx - short_window]
        long_trend = (current_price - self.close_prices[self.idx - long_window]) / self.close_prices[self.idx - long_window]
        
        # Trend confidence (agreement between timeframes)
        trend_confidence = 1.0 + abs(short_trend * long_trend)
        
        if action == 1 and self.position == 0:  # Buy
            reward = -buying_fee
            if short_trend > 0 and long_trend > 0:
                reward += 0.0005 * trend_confidence
            
        elif action == 2 and self.position == 1:  # Sell
            effective_sale = current_price * (1 - selling_fee)
            profit = (effective_sale - self.entry_price) / self.entry_price
            
            # Scale profit based on holding duration
            holding_duration = self.idx - self.entry_idx
            duration_scale = min(1.5, 1.0 + (holding_duration / 100))
            
            reward = profit * trend_confidence * duration_scale
            
            if short_trend < 0 and long_trend < 0:
                reward += 0.0005 * trend_confidence
                
        else:  # Hold
            # Penalize holding against trend more strongly in strong trends
            if self.position == 1:
                if short_trend < -0.001 and long_trend < -0.001:
                    reward = -0.0002 * trend_confidence
            else:
                if short_trend > 0.001 and long_trend > 0.001:
                    reward = -0.0002 * trend_confidence

        return reward

    def step(self, action):


        # Enforce that a buy is only allowed if we're not already in a position.
        if action == 1:
            if self.position == 0:  # Only execute buy if not already bought
                # print("buy")
                self.position = 1
                self.entry_price = self.close_prices[self.idx]
                self.shares = self.cash / self.close_prices[self.idx]
                self.cash = 0
            else:
                # If already bought and action == 1, treat it as hold (do nothing)
                action = 0

        elif action == 2:
            if self.position == 1:  # Only execute sell if we have a position
                # print("sell")
                self.position = 0
                self.cash = self.shares * self.close_prices[self.idx]
                self.shares = 0
                self.entry_price = None
            else:
                # If not in a position and action == 2, treat it as hold
                action = 0

        # For hold action (action == 0) or if buy/sell were overridden, do nothing.

        self.idx += 1
        if self.idx >= len(self.close_prices):
            if self.position == 1:
                self.cash = self.shares * self.close_prices[-1]
                self.shares = 0
                self.position = 0
            return self._get_state(), 0, True

        reward = self._get_reward(action)
        portfolio_value = self.cash if self.position == 0 else self.shares * self.close_prices[self.idx]
        self.max_portfolio_value = max(portfolio_value, self.max_portfolio_value)
        
        return self._get_state(), reward, False

    
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.episodes = []
        self.capacity = capacity

    def __len__(self):
        return len(self.episodes)
    
    def push_episode(self, episode):
        """An episode is a list of (state, action, reward, next_state, done) tuples."""
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)
    
    def sample(self, batch_size, seq_len):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
        
        # Keep sampling until we have batch_size sequences
        while len(batch_states) < batch_size:
            # Calculate total return for each episode
            episode_returns = []
            for episode in self.episodes:
                total_return = sum(transition[2] for transition in episode)  # Sum rewards
                episode_returns.append(total_return)
                
            # Convert to probabilities
            max_return = max(episode_returns)
            min_return = min(episode_returns)
            if max_return == min_return:
                probs = [1.0 / len(self.episodes)] * len(self.episodes)
            else:
                # Normalize to [0,1] and add small epsilon
                probs = [(r - min_return) / (max_return - min_return) + 0.01 for r in episode_returns]
                # Normalize to sum to 1
                total = sum(probs)
                probs = [p/total for p in probs]
                
            # Sample episode based on returns
            episode_idx = random.choices(range(len(self.episodes)), weights=probs, k=1)[0]
            episode = self.episodes[episode_idx]
            
            if len(episode) < seq_len:
                continue

            start = random.randint(0, len(episode) - seq_len)
            seq = episode[start : start + seq_len]
            states, actions, rewards, next_states, dones = zip(*seq)
            batch_states.append(np.array(states))
            batch_actions.append(np.array(actions))
            batch_rewards.append(np.array(rewards))
            batch_next_states.append(np.array(next_states))
            batch_dones.append(np.array(dones))
        
        return (torch.FloatTensor(np.array(batch_states)),
                torch.LongTensor(np.array(batch_actions)),
                torch.FloatTensor(np.array(batch_rewards)),
                torch.FloatTensor(np.array(batch_next_states)),
                torch.FloatTensor(np.array(batch_dones)))



def train_batch(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
   # Track losses for averaging
   losses = []
   for _ in range(16):  # Do 16 training iterations
       seq_states, seq_actions, seq_rewards, seq_next_states, seq_dones = \
           memory.sample(batch_size, seq_len=4)
       
       seq_states = seq_states.to(device)
       seq_actions = seq_actions.to(device)
       seq_rewards = seq_rewards.to(device)
       seq_next_states = seq_next_states.to(device)
       seq_dones = seq_dones.to(device)

       # produce Q-values for only the last step:
       current_q_values, _ = policy_net(seq_states)  # shape [B, 3]
       
       # gather using the final action:
       last_actions = seq_actions[:, -1]  # shape [B]
       current_q_values = current_q_values.gather(1, last_actions.unsqueeze(1))
       
       with torch.no_grad():
           # same logic for next_state Q
           next_q_values, _ = policy_net(seq_next_states)
           last_next_actions = next_q_values.max(1)[1].unsqueeze(1)
           # or do the same with target_net
           next_q_values_target, _ = target_net(seq_next_states)
           max_q_of_next = next_q_values_target.gather(1, last_next_actions)
           
           last_dones = seq_dones[:, -1].unsqueeze(1)
           max_q_of_next[last_dones.bool()] = 0.0
           
           last_rewards = seq_rewards[:, -1].unsqueeze(1)
           target_q_values = last_rewards + gamma * max_q_of_next
       
       loss = F.smooth_l1_loss(current_q_values, target_q_values)
       if torch.isnan(loss):
           print('Warning: NaN loss, skipping update')
           continue  # Skip this iteration entirely
       
       losses.append(loss.item())
       optimizer.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(policy_net.parameters(), .1)
       optimizer.step()

   if losses:  # Only calculate average if we have valid losses
       avg_loss = sum(losses) / len(losses)
       print(f"Average Loss: {avg_loss:.8f}")
       return avg_loss
   else:
       print("Warning: No valid losses in batch")
       return 0.0

# Update the train_dqn function to use the new checkpoint saving
def train_dqn(train_data_dict, val_data_dict, input_size, n_episodes=1000, batch_size=32, gamma=0.99, optimizer=None, initial_best_profit=float('-inf'), initial_best_excess=float('-inf')):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_size).to(device)
    target_net = DQN(input_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    if optimizer is None:
        print(f"optimizer: new")
        optimizer = optim.Adam(policy_net.parameters(), lr=0.000005)
    else:
        print(f"optimizer: {optimizer}")
        optimizer = optimizer

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

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
                save_new_global_best(best_model, optimizer, {
                    'profit': avg_profit,
                    'win_rate': avg_win_rate,
                    'excess_return': avg_excess_return,  # Add this line
                    'per_ticker_metrics': {t: m for t, m in zip(val_envs.keys(), val_metrics_all)}
                }, best_val_profit)
                print(f"\nNew best model! Episode {episode + 1}, Excess Return: {avg_excess_return:.2f}%")
            
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

def validate_model(model, env, device):
    state = env.reset()
    done = False
    initial_value = env.cash
    trades = []
    entry_price = None
    accumulated_reward = 0
    
    # Calculate buy and hold return
    buy_and_hold_shares = initial_value / env.close_prices[env.window_size]
    buy_and_hold_value = buy_and_hold_shares * env.close_prices[-1]
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
            entry_price = env.close_prices[env.idx]
        elif action == 2 and env.position == 1:
            exit_price = env.close_prices[env.idx]
            trades.append((exit_price - entry_price) / entry_price)
            entry_price = None
            
        state, reward, done = env.step(action)
        accumulated_reward += reward
    
    final_value = env.cash if env.position == 0 else env.shares * env.close_prices[env.idx]
    return {
        'profit': (final_value - initial_value) / initial_value,
        'num_trades': len(trades),
        'avg_return': np.mean(trades) if trades else 0,
        'win_rate': np.mean([t > 0 for t in trades]) if trades else 0,
        'accumulated_reward': accumulated_reward,
        'buy_and_hold_return': buy_and_hold_return
    }

def get_state_size(window_size):
    """
    With raw data and position:
      - OHLC: window_size * 4
      - Volume: window_size
      - Position: 1
    Total: window_size * 5 + 1
    """
    return window_size * 5 + 1



# ---------------------------------------------
# 1) Load the all-time best model at startup
# ---------------------------------------------
BEST_CHECKPOINT_FILE = os.path.join("checkpoints", "best_checkpoint.pt")

def load_global_best():
    if os.path.exists(BEST_CHECKPOINT_FILE):
        best_checkpoint = torch.load(BEST_CHECKPOINT_FILE)
        best_profit = best_checkpoint['metrics']['profit']
        best_excess_return = best_checkpoint['metrics'].get('excess_return', float('-inf'))  # Add this line
        print(f"[INFO] Found global best checkpoint with profit={best_profit * 100:.3f}%, excess_return={best_excess_return:.3f}%")
        return best_checkpoint, best_profit, best_excess_return  # Modified return
    else:
        return None, float('-inf'), float('-inf')  # Modified return


# ---------------------------------------------
# 2) Save new best model ONLY if it beats all-time best
# ---------------------------------------------
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
    torch.save(checkpoint, BEST_CHECKPOINT_FILE)
    print(f"[SAVE] Overwrote global best => {BEST_CHECKPOINT_FILE}")

    return ckpt_path


def main():
    full_data_dict = get_market_data_multi()

    if not full_data_dict:
        print("No data available for training. Exiting.")
        return

    # Split data for each ticker
    train_data_dict = {}
    val_data_dict = {}
    for ticker, data in full_data_dict.items():
        split_idx = int(len(data) * 0.7)
        train_data_dict[ticker] = data[:split_idx].copy()
        val_data_dict[ticker] = data[split_idx:].copy()

    window_size = 48
    input_size = get_state_size(window_size)
    print(f"Input size: {input_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size).to(device)

    # Load the best checkpoint and its metrics
    best_checkpoint, best_profit, best_excess_return = load_global_best() 
    if best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])
        initial_best_profit = best_profit
        initial_best_excess = best_excess_return  # Add this line
    else:
        initial_best_profit = float('-inf')
        initial_best_excess = float('-inf')  # Add this line
    
    print(f"Initial Best Profit: {initial_best_profit*100:.2f}%")
    print(f"Initial Best Excess Return: {initial_best_excess:.2f}%")  # Add this line

    # Train with awareness of historical best performance
    results = train_dqn(
        train_data_dict, 
        val_data_dict, 
        input_size,
        n_episodes=1000,
        batch_size=4,
        gamma=0.99,
        optimizer=None,
        initial_best_profit=initial_best_profit,
        initial_best_excess=initial_best_excess  # Add this line
    )
    
    # print("\nFinal Results:")
    # final_metrics = validate_model(results['final_model'], SimpleTradeEnv(val_data), device)
    # best_metrics = validate_model(results['best_model'], SimpleTradeEnv(val_data), device)
    
    # print(f"Final Model - Profit: {final_metrics['profit']*100:.2f}%, Trades: {final_metrics['num_trades']}")
    # print(f"Best Model  - Profit: {best_metrics['profit']*100:.2f}%, Trades: {best_metrics['num_trades']}")
    # if initial_best_profit != float('-inf'):
    #     print(f"Previous Best - Profit: {initial_best_profit*100:.2f}%")

if __name__ == "__main__":
    main()