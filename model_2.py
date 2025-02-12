import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple
import random
import copy

def get_market_data(symbol="SPY", period="60d", interval="15m"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        print(f"Successfully fetched {len(df)} records for {symbol}")
        return df if not df.empty else None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

class DQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        
        self.value_net = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        features = self.feature_net(x).unsqueeze(1)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(features)
        return self.value_net(lstm_out[:, -1, :])

class SimpleTradeEnv:
    def __init__(self, data, window_size=10):
        self.data = data
        self.prices = self.data['Close'].values
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.idx = self.window_size
        self.position = 0
        self.cash = 1000
        self.shares = 0
        self.entry_price = None
        self.max_portfolio_value = self.cash
        return self._get_state()

    def get_state_size(window_size):
        """Calculate input size for the neural network"""
        price_window = window_size  # Price history window
        position_indicator = 1      # Binary indicator for current position
        technical_features = 3      # Additional features
        state_size = window_size + technical_features + position_indicator
        return state_size

    def _get_state(self):
        # Price window
        window = self.prices[self.idx - self.window_size:self.idx]
        window_returns = np.diff(window) / window[:-1]  # Price returns
        
        # Technical indicators (simple ones for demonstration)
        volatility = np.std(window_returns)
        momentum = np.mean(window_returns[-3:])  # 3-period momentum
        trend = (window[-1] - window[0]) / window[0]
        
        # Combine all features
        technical_features = np.array([volatility, momentum, trend])
        position_feature = np.array([float(self.position)])
        
        return np.concatenate([window, technical_features, position_feature])

    def _get_reward(self, action):
        reward = 0.0
        if action == 1 and self.position == 0:  # Buy
            reward -= 0.01 * self.cash  # 1% transaction fee
        elif action == 2 and self.position == 1:  # Sell
            sell_value = self.prices[self.idx] * self.shares
            profit = sell_value - (self.entry_price * self.shares)
            reward = profit - (0.01 * sell_value)  # Profit minus 1% fee
        return reward

    def step(self, action):
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = self.prices[self.idx]
            self.shares = self.cash / self.prices[self.idx]
            self.cash = 0
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.cash = self.shares * self.prices[self.idx]
            self.shares = 0
            self.entry_price = None

        self.idx += 1
        if self.idx >= len(self.prices):
            if self.position == 1:
                self.cash = self.shares * self.prices[-1]
                self.shares = 0
                self.position = 0
            return self._get_state(), 0, True

        reward = self._get_reward(action)
        portfolio_value = self.cash if self.position == 0 else self.shares * self.prices[self.idx]
        self.max_portfolio_value = max(portfolio_value, self.max_portfolio_value)
        
        return self._get_state(), reward, False

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (torch.FloatTensor(states), torch.LongTensor(actions),
                torch.FloatTensor(rewards), torch.FloatTensor(next_states),
                torch.FloatTensor(dones))
        
    def __len__(self):
        return len(self.buffer)

def train_batch(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    states, actions = states.to(device), actions.to(device)
    rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)

    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
    
    with torch.no_grad():
        next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = target_net(next_states).gather(1, next_actions)
        next_q_values[dones.bool().unsqueeze(1)] = 0.0
        target_q_values = rewards.unsqueeze(1) + gamma * next_q_values

    loss = F.smooth_l1_loss(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# Update the train_dqn function to use the new checkpoint saving
def train_dqn(train_data, val_data, input_size, n_episodes=1000, batch_size=32, gamma=0.99):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_size).to(device)
    target_net = DQN(input_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)
    memory = ReplayBuffer(100000)
    
    window_size = 10
    train_env = SimpleTradeEnv(train_data, window_size=window_size)
    val_env = SimpleTradeEnv(val_data, window_size=window_size)
    
    epsilon = 1.0
    best_val_profit = float('-inf')
    best_model = None
    
    for episode in range(n_episodes):
        state = train_env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(np.array([state])).to(device)
            action = random.randrange(3) if random.random() < epsilon else \
                    policy_net(state_tensor).max(1)[1].item()
            
            next_state, reward, done = train_env.step(action)
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            
            if len(memory) >= batch_size:
                train_batch(policy_net, target_net, optimizer, memory, batch_size, gamma, device)
        
        if episode % 1 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            policy_net.eval()
            val_metrics = validate_model(policy_net, val_env, device)
            print(f"Metrics:")  
            print(f"  Profit: {val_metrics['profit']*100:.2f}%")
            print(f"  Number of Trades: {val_metrics['num_trades']}")
            print(f"  Accumulated Reward: {val_metrics['accumulated_reward']:.2f}")
            print(f"  Buy & Hold Return: {val_metrics['buy_and_hold_return']*100:.2f}%")
            print(f"  Average Trade Return: {val_metrics['avg_return']*100:.2f}%")
            print(f"  Win Rate: {val_metrics['win_rate']*100:.1f}%")
            
            if val_metrics['profit'] > best_val_profit:
                best_model = copy.deepcopy(policy_net)
                global_best_profit = val_metrics['profit']
                save_new_global_best(best_model, optimizer, val_metrics, global_best_profit)
                print(f"\nNew best model! Episode {episode + 1}")

            policy_net.train()
        
        epsilon = max(0.01, epsilon * 0.985)
    
    return {'final_model': policy_net, 'best_model': best_model}

def validate_model(model, env, device):
    state = env.reset()
    done = False
    initial_value = env.cash
    trades = []
    entry_price = None
    accumulated_reward = 0
    
    # Calculate buy and hold return
    buy_and_hold_shares = initial_value / env.prices[env.window_size]
    buy_and_hold_value = buy_and_hold_shares * env.prices[-1]
    buy_and_hold_return = (buy_and_hold_value - initial_value) / initial_value
    
    while not done:
        state_tensor = torch.FloatTensor([state]).to(device)
        action = model(state_tensor).max(1)[1].item()
        
        if action == 1 and env.position == 0:
            entry_price = env.prices[env.idx]
        elif action == 2 and env.position == 1:
            exit_price = env.prices[env.idx]
            trades.append((exit_price - entry_price) / entry_price)
            entry_price = None
            
        state, reward, done = env.step(action)
        accumulated_reward += reward
    
    final_value = env.cash if env.position == 0 else env.shares * env.prices[env.idx]
    return {
        'profit': (final_value - initial_value) / initial_value,
        'num_trades': len(trades),
        'avg_return': np.mean(trades) if trades else 0,
        'win_rate': np.mean([t > 0 for t in trades]) if trades else 0,
        'accumulated_reward': accumulated_reward,
        'buy_and_hold_return': buy_and_hold_return
    }

def get_state_size(window_size):
    """Calculate input size for the neural network"""
    price_window = window_size  # Price history window
    position_indicator = 1      # Binary indicator for current position
    technical_features = 3      # Additional features
    state_size = window_size + technical_features + position_indicator
    return state_size

# ---------------------------------------------
# 1) Load the all-time best model at startup
# ---------------------------------------------
BEST_CHECKPOINT_FILE = os.path.join("checkpoints", "best_checkpoint.pt")

def load_global_best():
    if os.path.exists(BEST_CHECKPOINT_FILE):
        best_checkpoint = torch.load(BEST_CHECKPOINT_FILE)
        best_profit = best_checkpoint['metrics']['profit']
        print(f"[INFO] Found global best checkpoint with profit={best_profit:.3f}")
        return best_checkpoint, best_profit
    else:
        return None, float('-inf')


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
    full_data = get_market_data(symbol="SPY", period="730d", interval="1h")
    if full_data is None or len(full_data) < 11:
        print("Not enough data for training. Exiting.")
        return

    split_idx = int(len(full_data) * 0.7)
    train_data = full_data[:split_idx].copy()
    val_data = full_data[split_idx:].copy()

    window_size = 10
    input_size = get_state_size(window_size)
    print(f"Input size: {input_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size).to(device)
    
    # Load the best checkpoint and its metrics
    best_checkpoint, best_profit =  load_global_best()
    if best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])
        initial_best_profit = best_profit
    else:
        initial_best_profit = float('-inf')
    
    # Train with awareness of historical best performance
    results = train_dqn(train_data, val_data, input_size, n_episodes=50, batch_size=64)
    
    print("\nFinal Results:")
    final_metrics = validate_model(results['final_model'], SimpleTradeEnv(val_data), device)
    best_metrics = validate_model(results['best_model'], SimpleTradeEnv(val_data), device)
    
    print(f"Final Model - Profit: {final_metrics['profit']*100:.2f}%, Trades: {final_metrics['num_trades']}")
    print(f"Best Model  - Profit: {best_metrics['profit']*100:.2f}%, Trades: {best_metrics['num_trades']}")
    if initial_best_profit != float('-inf'):
        print(f"Previous Best - Profit: {initial_best_profit*100:.2f}%")

if __name__ == "__main__":
    main()