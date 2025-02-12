import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass
from plot import plot_candlestick_signals  # <-- Your candlestick plotting function

# ---------------------------------------------------------------------
# Trading configuration – adding long-term features (with '_long') and new VWAP features.
# ---------------------------------------------------------------------

@dataclass
class TradingConfig:
    """Configuration parameters for trading simulation."""
    initial_capital: float = 100.0
    feature_columns: List[str] = None

    def __post_init__(self):
        if self.feature_columns is None:
            short_features = [
                'Lag_Return', 'Return_5', 'Return_10', 'Return_20',
                'SMA_5', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
                'Volatility_5', 'Volatility_10', 'Volatility_20',
                'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B',
                'Ichimoku_diff', 'Above_Kumo', 'Below_Kumo', 'Kumo_Strength',
                'HA_Open', 'HA_Close', 'HA_High', 'HA_Low', 'HA_Close_diff',
                'HA_Trend', 'HA_Size', 'HA_Body', 'HA_Upper_Shadow', 'HA_Lower_Shadow',
                'Price_to_SMA20', 'Price_to_Kijun',
                # New VWAP features:
                'VWAP', 'Price_to_VWAP', 'VWAP_Slope'
            ]
            long_features = [f + '_long' for f in short_features]
            self.feature_columns = short_features + long_features

# ---------------------------------------------------------------------
# Data Fetching and Normalization Functions
# ---------------------------------------------------------------------

def normalize_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize the columns of a yfinance DataFrame."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    column_map = {
        'Adj Close': 'Close',
        'adj close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    df.rename(columns=column_map, inplace=True)
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df

def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin Ashi candles."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    expected_cols = ['Open', 'High', 'Low', 'Close']
    col_mapping = {col.lower(): col for col in expected_cols}
    df.rename(columns=col_mapping, inplace=True)
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")
    df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors='coerce')
    df['HA_Close'] = df[expected_cols].mean(axis=1)
    df['HA_Open'] = pd.Series(np.nan, index=df.index)
    df.loc[df.index[0], 'HA_Open'] = df[['Open', 'Close']].iloc[0].mean()
    for i in range(1, len(df)):
        df.iloc[i, df.columns.get_loc('HA_Open')] = (
            df.iloc[i-1, df.columns.get_loc('HA_Open')] +
            df.iloc[i-1, df.columns.get_loc('HA_Close')]
        ) / 2
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    return df

# ---------------------------------------------------------------------
# Feature Engineering Function (with new VWAP features)
# ---------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical analysis features with Heikin Ashi, Ichimoku and VWAP."""
    df = df.copy()
    df['Return'] = df['Close'].pct_change().fillna(0.0)
    df['Lag_Return'] = df['Return'].shift(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    df['Return_20'] = df['Close'].pct_change(20)
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Volatility_5'] = df['Return'].rolling(window=5, min_periods=1).std()
    df['Volatility_10'] = df['Return'].rolling(window=10, min_periods=1).std()
    df['Volatility_20'] = df['Return'].rolling(window=20, min_periods=1).std()
    high_rolling = df['High'].rolling(window=9, min_periods=9)
    low_rolling = df['Low'].rolling(window=9, min_periods=9)
    df['Tenkan_Sen'] = (high_rolling.max() + low_rolling.min()) / 2
    high_rolling_26 = df['High'].rolling(window=26, min_periods=26)
    low_rolling_26 = df['Low'].rolling(window=26, min_periods=26)
    df['Kijun_Sen'] = (high_rolling_26.max() + low_rolling_26.min()) / 2
    df['Senkou_Span_A'] = (df['Tenkan_Sen'] + df['Kijun_Sen']) / 2
    high_rolling_52 = df['High'].rolling(window=52, min_periods=52)
    low_rolling_52 = df['Low'].rolling(window=52, min_periods=52)
    df['Senkou_Span_B'] = (high_rolling_52.max() + low_rolling_52.min()) / 2
    df['Ichimoku_diff'] = df['Tenkan_Sen'] - df['Kijun_Sen']
    df['Above_Kumo'] = ((df['Close'] > df['Senkou_Span_A']) &
                         (df['Close'] > df['Senkou_Span_B'])).astype(float)
    df['Below_Kumo'] = ((df['Close'] < df['Senkou_Span_A']) &
                         (df['Close'] < df['Senkou_Span_B'])).astype(float)
    df['Kumo_Strength'] = df['Senkou_Span_A'] - df['Senkou_Span_B']
    ha_df = compute_heikin_ashi(df)
    ha_columns = ['HA_Open', 'HA_Close', 'HA_High', 'HA_Low']
    df[ha_columns] = ha_df[ha_columns]
    df['HA_Close_diff'] = df['HA_Close'] - df['Close']
    df['HA_Trend'] = (df['HA_Close'] > df['HA_Open']).astype(float)
    df['HA_Size'] = df['HA_High'] - df['HA_Low']
    df['HA_Body'] = abs(df['HA_Close'] - df['HA_Open'])
    df['HA_Upper_Shadow'] = df['HA_High'] - df[['HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Lower_Shadow'] = df[['HA_Open', 'HA_Close']].min(axis=1) - df['HA_Low']
    df['Price_to_SMA20'] = df['Close'] / df['SMA_20'] - 1
    df['Price_to_Kijun'] = df['Close'] / df['Kijun_Sen'] - 1

    # -------------------------------
    # New VWAP Features
    # -------------------------------
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).rolling(window=15, min_periods=1).sum() / \
                 df['Volume'].rolling(window=15, min_periods=1).sum()
    df['Price_to_VWAP'] = df['Close'] / df['VWAP'] - 1
    df['VWAP_Slope'] = df['VWAP'].pct_change(fill_method=None)
    df.drop(columns=['Typical_Price'], inplace=True)

    if 'Volume' in df.columns:
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Trend'] = (df['Volume'] > df['Volume_SMA_5']).astype(float)

    return df.dropna()

# ---------------------------------------------------------------------
# New Function: Fetch and Merge Multi–Timeframe Data
# ---------------------------------------------------------------------

def fetch_multi_timeframe_data_for_tickers(
    tickers: List[str],
    period_short: str = "59d",
    interval_short: str = "15m",
    period_long: str = "2y",
    interval_long: str = "1d"
) -> List[pd.DataFrame]:
    """
    For each ticker, fetch short timeframe data (15m) and long timeframe data (daily),
    compute features for each timeframe, then merge the long timeframe features into the short timeframe data
    using merge_asof (backward fill).
    """
    dataframes = []
    for ticker in tickers:
        try:
            # Short timeframe data
            df_short = yf.download(ticker, period=period_short, interval=interval_short, progress=False)
            if df_short.empty:
                print(f"Warning: No short timeframe data received for {ticker}")
                continue
            df_short = normalize_yfinance_columns(df_short, ticker)
            df_short['Ticker'] = ticker
            df_short.sort_index(inplace=True)
            df_short_features = create_features(df_short)
            df_short_features.sort_index(inplace=True)

            # Long timeframe data
            df_long = yf.download(ticker, period=period_long, interval=interval_long, progress=False)
            if df_long.empty:
                print(f"Warning: No long timeframe data received for {ticker}")
                continue
            df_long = normalize_yfinance_columns(df_long, ticker)
            df_long['Ticker'] = ticker
            df_long.sort_index(inplace=True)
            df_long_features = create_features(df_long)
            df_long_features.sort_index(inplace=True)

            # Merge long timeframe features into short timeframe data using merge_asof
            merged = pd.merge_asof(df_short_features, df_long_features,
                                   left_index=True, right_index=True,
                                   direction='backward',
                                   suffixes=('', '_long'))
            dataframes.append(merged)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            continue

    if not dataframes:
        raise ValueError("No valid multi-timeframe data received for any tickers")
    return dataframes

# ---------------------------------------------------------------------
# Build Multiticker Dataset (using multi–timeframe merged data)
# ---------------------------------------------------------------------

def build_multiticker_dataset(ticker_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Build combined dataset from multiple tickers."""
    processed_dfs = []
    for df in ticker_dfs:
        df['Return'] = df['Close'].pct_change().fillna(0.0)
        df['NextReturn'] = df['Return'].shift(-1).fillna(0.0)
        processed_dfs.append(df.dropna())
    return pd.concat(processed_dfs, axis=0, ignore_index=True)

# ---------------------------------------------------------------------
# Trading Environment for RL (Single-Unit Simulation)
# ---------------------------------------------------------------------

class TradingEnvironment:
    def __init__(self, features: np.ndarray, returns: np.ndarray, initial_capital: float = 100.0, 
                 transaction_cost: float = 0.001, reward_scaling: float = 1.0):
        self.features = features
        self.returns = returns
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        done = False
        reward = 0.0
        r = self.returns[self.current_step]
        
        # Calculate multiple timeframe trends
        if self.current_step >= 20:
            trends = {
                'short': np.mean(self.returns[self.current_step-5:self.current_step]),
                'medium': np.mean(self.returns[self.current_step-10:self.current_step]),
                'long': np.mean(self.returns[self.current_step-20:self.current_step])
            }
            trend_alignment = (np.sign(trends['short']) == np.sign(trends['medium']) == np.sign(trends['long']))
        else:
            trends = {'short': 0, 'medium': 0, 'long': 0}
            trend_alignment = False

        previous_position = self.position
        
        if self.position == 0:  # No position
            if action == 1:  # Buy
                self.capital *= (1 - self.transaction_cost)
                self.position = 1
                self.entry_price = self.capital
                
                # Reward for buying in strong uptrend
                if trend_alignment and trends['short'] > 0:
                    reward += 2.0  # Increased reward for good entry
                elif not trend_alignment and trends['short'] < 0:
                    reward -= 1.0  # Penalty for buying in downtrend
        
        else:  # Holding position
            # Calculate running P&L
            current_pnl = (self.capital * (1 + r) - self.entry_price) / self.entry_price
            reward += current_pnl * 5.0  # Continuous reward/penalty for position performance
            
            if action == 2:  # Sell
                self.capital *= (1 + r)  # Apply return before selling
                self.capital *= (1 - self.transaction_cost)
                
                # Calculate final P&L for this trade
                final_pnl = (self.capital - self.entry_price) / self.entry_price
                
                # Reward based on profit/loss
                if final_pnl > 0:
                    reward += final_pnl * 10.0  # Bigger reward for profitable trades
                else:
                    reward += final_pnl * 5.0   # Smaller penalty for losses
                    
                # Additional reward for good exit
                if trend_alignment and trends['short'] < 0:
                    reward += 2.0  # Reward for selling in downtrend
                
                self.position = 0
        
        # Penalty for holding through adverse movement
        if self.position == 1 and r < -0.005:  # -0.5% threshold
            reward -= abs(r) * 3.0
        
        # Trading frequency penalty
        if previous_position != self.position:
            reward -= 0.1  # Small fixed penalty for trading
        
        # Terminal state handling
        self.current_step += 1
        if self.current_step >= len(self.features):
            done = True
            # Extra penalty for holding position at end
            if self.position == 1:
                reward -= 1.0
            next_state = np.zeros_like(self.features[0])
        else:
            next_state = self.features[self.current_step]
            
        return next_state, reward * self.reward_scaling, done
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0  # 0 = no position, 1 = long position
        self.entry_price = 0.0
        return self.features[self.current_step]  # Return initial state

# ---------------------------------------------------------------------
# Vectorized Trading Environment
# ---------------------------------------------------------------------

class VectorizedTradingEnvironment:
    def __init__(self, envs: List[TradingEnvironment]):
        self.envs = envs
        self.n_envs = len(envs)

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            ns, r, d = env.step(action)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
        # Convert to arrays; for dones, we can use 1 for done and 0 for not done.
        return np.stack(next_states), np.array(rewards), np.array(dones)

# ---------------------------------------------------------------------
# Policy Network for RL (using LayerNorm for single-sample support)
# ---------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        # Separate feature processing paths
        self.market_features = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        
        # Final decision layers
        self.decision = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 3)  # 3 actions: hold, buy, sell
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process features
        features = self.market_features(x)
        
        # Add sequence dimension if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
            
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Final decision
        logits = self.decision(attn_out.squeeze(1))
        return logits

# ---------------------------------------------------------------------
# RL Training Loop using REINFORCE with Checkpoint Saving, Resume Support, and Vectorized Environments
# ---------------------------------------------------------------------
def compute_advantages(returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Compute advantages using returns and value predictions with error checking
    """
    if returns.dim() != values.dim():
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        if values.dim() == 1:
            values = values.unsqueeze(1)
    
    if returns.shape != values.shape:
        raise ValueError(f"Shape mismatch: returns shape {returns.shape} != values shape {values.shape}")
    
    advantages = returns - values
    if advantages.shape[0] > 1:  # Only normalize if we have more than one sample
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def compute_returns(rewards: torch.Tensor, masks: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns for each timestep with error checking
    """
    if rewards.dim() != masks.dim():
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if masks.dim() == 1:
            masks = masks.unsqueeze(1)
            
    if rewards.shape != masks.shape:
        raise ValueError(f"Shape mismatch: rewards shape {rewards.shape} != masks shape {masks.shape}")
    
    returns = []
    R = 0
    
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    
    returns = torch.stack(returns)
    if returns.shape[0] > 1:  # Only normalize if we have more than one sample
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def batches(data: range, batch_size: int):
    """
    Generate batches from range of indices
    """
    indices = list(data)
    np.random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            
            nn.Linear(32, 1)  # Single value output
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  
    
def train_policy(network: nn.Module, optimizer: optim.Optimizer, env: VectorizedTradingEnvironment,
                n_episodes: int = 500, gamma: float = 0.99, epsilon: float = 0.2,
                checkpoint_freq: int = 10, checkpoint_dir: str = "checkpoints", 
                device: torch.device = torch.device("cpu")):
    """Enhanced training using PPO and saving the latest checkpoint"""
    network.train()
    rewards_history = []
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create value network for PPO
    value_net = ValueNetwork(network.input_dim).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.0001)
    
    # Add entropy scheduling
    initial_entropy_coef = 0.1
    final_entropy_coef = 0.01
    
    start_episode = 0
    print("Starting fresh training run")
    
    for episode in range(start_episode, n_episodes, env.n_envs):
        # Store episode data
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        episode_dones = []
        
        # Reset environment
        states = env.reset()
        done = False
        
        # Calculate current entropy coefficient
        progress = episode / n_episodes
        entropy_coef = initial_entropy_coef * (1 - progress) + final_entropy_coef * progress
        
        while not done:
            state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = network(state_tensor)
                values = value_net(state_tensor)
                action_probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            # Take actions in environment
            next_states, rewards, dones = env.step(actions.cpu().numpy())
            
            # Store transition
            episode_states.append(state_tensor)
            episode_actions.append(actions)
            episode_rewards.append(torch.tensor(rewards, dtype=torch.float32).to(device))
            episode_values.append(values.squeeze())
            episode_log_probs.append(log_probs)
            episode_dones.append(torch.tensor(dones, dtype=torch.bool).to(device))
            
            # Update states
            states = next_states
            done = all(dones)
        
        # Convert episode data to tensors
        episode_states = torch.cat(episode_states)
        episode_actions = torch.cat(episode_actions)
        episode_rewards = torch.cat(episode_rewards)
        episode_values = torch.cat(episode_values)
        episode_log_probs = torch.cat(episode_log_probs)
        episode_dones = torch.cat(episode_dones)
        
        # Compute returns and advantages
        returns = compute_returns(episode_rewards.unsqueeze(1), 
                                (~episode_dones).unsqueeze(1), 
                                gamma)
        advantages = compute_advantages(returns, episode_values.unsqueeze(1))
        
        # PPO update with multiple epochs
        for _ in range(20):  # Increased from 10
            batch_size = 64
            indices = torch.randperm(len(episode_states))
            
            for start_idx in range(0, len(episode_states), batch_size):
                # Get batch indices
                batch_idx = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                batch_states = episode_states[batch_idx]
                batch_actions = episode_actions[batch_idx]
                batch_log_probs = episode_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Policy update
                logits = network(batch_states)
                new_dist = torch.distributions.Categorical(torch.softmax(logits, dim=1))
                new_log_probs = new_dist.log_prob(batch_actions)
                
                # Calculate policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value update
                value_pred = value_net(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                # Entropy bonus with scheduling
                entropy_loss = -entropy_coef * new_dist.entropy().mean()
                
                # Total loss with adjusted coefficients
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                # Optimization step with increased gradient clipping
                optimizer.zero_grad()
                value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=10)
                optimizer.step()
                value_optimizer.step()
        
        # Calculate average episode reward
        avg_reward = episode_rewards.mean().item()
        rewards_history.append(avg_reward)
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Entropy Coef: {entropy_coef:.3f}")
        
        # Save checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode + 1}.pt")
            torch.save({
                'episode': episode + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'avg_reward': avg_reward,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
# Deterministic Test Episode Function
# ---------------------------------------------------------------------

def run_deterministic_test(network: nn.Module, env: VectorizedTradingEnvironment,
                           config: TradingConfig, device: torch.device) -> None:
    network.eval()
    # Run one deterministic episode per environment and average the results.
    states = env.reset()
    done_flags = np.zeros(env.n_envs, dtype=bool)
    equity_curves = [[env.envs[i].capital] for i in range(env.n_envs)]
    while not all(done_flags):
        state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        logits = network(state_tensor)
        actions = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().tolist()
        states, rewards, done_flags = env.step(actions)
        for i in range(env.n_envs):
            env.envs[i].capital  # updated inside step
            equity_curves[i].append(env.envs[i].capital)
    final_capitals = [env.envs[i].capital for i in range(env.n_envs)]
    avg_final_equity = np.mean(final_capitals)
    total_profit = avg_final_equity - config.initial_capital
    print("\nTest Episode Results (Deterministic):")
    print(f"Avg Final Equity: ${avg_final_equity:.2f}")
    print(f"Avg Total Profit: ${total_profit:.2f}")
    # Compute drawdowns for each environment and average
    drawdowns = []
    for curve in equity_curves:
        running_max = np.maximum.accumulate(curve)
        dd = (np.array(curve) - running_max) / running_max
        drawdowns.append(dd.min() * 100)
    avg_drawdown = np.mean(drawdowns)
    print(f"Avg Maximum Drawdown: {avg_drawdown:.2f}%")

# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TradingConfig()
    tickers = [
        "AAPL",  
        # "MSFT", "AMZN", "GOOG", "META",
        # "NVDA", "TSLA", "BRK-B", "JNJ", "V",
        # "PG", "UNH", "HD", "DIS", "KO",
        # "PFE", "CSCO", "INTC", "ORCL", "WMT",
        # "SPY", "QQQ", "DIA", "IWM", "GLD",
        # "SLV", "BTC-USD", "ETH-USD", "VTI"
    ]

    # Fetch multi-timeframe data:
    # - Short timeframe: 59 days of 15-minute data
    # - Long timeframe: 2 years of daily data
    ticker_dfs = []
    for ticker in tickers:
        try:
            df = fetch_multi_timeframe_data_for_tickers(
                [ticker],
                period_short="59d",
                interval_short="15m",
                period_long="2y",
                interval_long="1d"
            )
            if df:
                ticker_dfs.append(df[0])
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue

    combined_df = build_multiticker_dataset(ticker_dfs)
    print(f"\nInitial Data Statistics:")
    print(f"Combined dataset shape: {combined_df.shape}")

    # For the RL environment, use the features and the short-term returns.
    features = combined_df[config.feature_columns].values
    returns = combined_df['Return'].values  # Use per-bar return

    # Create multiple copies of the TradingEnvironment for vectorized simulation
    n_envs = 16  # Increased from 8
    envs = [TradingEnvironment(
        features=features, 
        returns=returns, 
        initial_capital=config.initial_capital,
        transaction_cost=0.001,
        reward_scaling=10.0  # Increased reward scaling
    ) for _ in range(n_envs)]
    vec_env = VectorizedTradingEnvironment(envs)

    # Create the policy network with higher learning rate
    policy_net = PolicyNetwork(input_dim=len(config.feature_columns))
    policy_net.to(device)
    optimizer = optim.AdamW(
        policy_net.parameters(), 
        lr=0.0005,  # Increased learning rate
        weight_decay=0.001  # Reduced weight decay
    )

    # Train with modified parameters
    train_policy(
        policy_net, 
        optimizer, 
        vec_env, 
        n_episodes=1000,  # More episodes
        gamma=0.99,
        epsilon=0.2,
        checkpoint_freq=10, 
        checkpoint_dir="checkpoints", 
        device=device
    )

    # Run a deterministic test episode.
    run_deterministic_test(policy_net, vec_env, config, device)

if __name__ == "__main__":
    main()
