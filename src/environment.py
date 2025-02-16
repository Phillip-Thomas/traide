import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import List, Tuple
from .technical_indicators import (
    calculate_rsi_gpu,
    calculate_macd_gpu,
    calculate_bollinger_bands_gpu
)
import pandas as pd

def get_state_size(window_size):
    """Calculate exact state size including yield ETF-specific features"""
    # Price data (OHLCV)
    price_size = window_size * 5  # Open, High, Low, Close, Volume

    # Technical indicators
    tech_size = (
        window_size +      # RSI
        window_size * 3 +  # MACD, Signal, Histogram
        window_size * 3    # BB Upper, Middle, Lower
    )

    # Yield ETF specific features
    yield_etf_size = (
        window_size * 3 +  # Underlying asset correlation features (ratio, spread, correlation)
        3                  # Dividend timing features (next_div, days_to_div, hours_to_div)
    )

    # Position info
    position_info_size = 4  # Position, Cash, Shares, Entry Price

    total_size = price_size + tech_size + yield_etf_size + position_info_size

    return total_size

class SimpleTradeEnvGPU:
    def __init__(self, data=None, normalized_data=None, indicators=None, window_size=10, device=None):
        self.window_size = window_size
        self.device = device if device is not None else torch.device('cuda')
        self.max_steps = 1000  # Add max steps limit
        self.step_count = 0    # Add step counter

        if normalized_data is not None and indicators is not None:
            # Move data to specified device immediately
            self.raw_data = {
                k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v).to(self.device)
                for k, v in normalized_data.items()
            }
            self.indicators = {
                k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v).to(self.device)
                for k, v in indicators.items()
            }
        else:
            # Create tensors on CPU first
            self.raw_data = {
                'Open': torch.tensor(data['Open'].to_numpy(), dtype=torch.float32),
                'High': torch.tensor(data['High'].to_numpy(), dtype=torch.float32),
                'Low': torch.tensor(data['Low'].to_numpy(), dtype=torch.float32),
                'Close': torch.tensor(data['Close'].to_numpy(), dtype=torch.float32),
                'Volume': torch.tensor(data['Volume'].to_numpy(), dtype=torch.float32)
            }

            # Normalize data
            scale = self.raw_data['Close'][window_size].item()
            volume_scale = self.raw_data['Volume'][window_size].item()

            for key in self.raw_data:
                self.raw_data[key] = self.raw_data[key] / (volume_scale if key == 'Volume' else scale)

            # Calculate indicators on CPU
            self._precalculate_indicators()

            # Move data to device after all calculations
            self.raw_data = {k: v.to(self.device) for k, v in self.raw_data.items()}
            self.indicators = {k: v.to(self.device) for k, v in self.indicators.items()}

        # Calculate and store state size
        self.state_size = get_state_size(window_size)
        
        # Pre-allocate state buffer on device
        self.state_buffer = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)

        # Initialize position info on device
        self.position = 0
        self.cash = torch.tensor(1.0, device=self.device)
        self.shares = torch.tensor(0.0, device=self.device)
        self.entry_price = None

        # Calculate valid episode lengths
        self.data_length = len(next(iter(self.raw_data.values())))
        self.min_episode_length = 100  # Minimum episode length
        self.max_episode_length = 500  # Maximum episode length
        self.episode_length = None
        self.start_idx = None

        self.reset()

    def _precalculate_indicators(self):
        """Pre-calculate indicators for the environment"""
        self.indicators = {
            'rsi': calculate_rsi_gpu(self.raw_data['Close']),
            'macd': calculate_macd_gpu(self.raw_data['Close'])[0],
            'signal': calculate_macd_gpu(self.raw_data['Close'])[1],
            'hist': calculate_macd_gpu(self.raw_data['Close'])[2],
            'bb_upper': calculate_bollinger_bands_gpu(self.raw_data['Close'])[0],
            'bb_middle': calculate_bollinger_bands_gpu(self.raw_data['Close'])[1],
            'bb_lower': calculate_bollinger_bands_gpu(self.raw_data['Close'])[2]
        }

    def _get_state(self):
        """Get current state tensor directly on GPU"""
        idx = self.idx
        window_start = idx - self.window_size

        # All data should already be on GPU
        price_data = torch.cat([
            self.raw_data[k][window_start:idx]
            for k in ['Open', 'High', 'Low', 'Close', 'Volume']
        ])

        indicator_data = torch.cat([
            self.indicators[k][window_start:idx]
            for k in ['rsi', 'macd', 'signal', 'hist', 'bb_upper', 'bb_middle', 'bb_lower']
        ])

        # Get yield ETF specific features
        correlation_data = torch.cat([
            self.raw_data[k][window_start:idx]
            for k in ['underlying_ratio', 'underlying_spread', 'underlying_correlation']
        ]) if all(k in self.raw_data for k in ['underlying_ratio', 'underlying_spread', 'underlying_correlation']) else torch.zeros(self.window_size * 3, device=self.device)

        # Get dividend timing features
        dividend_data = torch.tensor([
            self.raw_data.get('days_to_dividend', torch.zeros(1, device=self.device))[idx],
            self.raw_data.get('hours_to_dividend', torch.zeros(1, device=self.device))[idx],
            float(self.raw_data.get('next_dividend', pd.NaT) is not pd.NaT)  # Binary indicator if next dividend is known
        ], device=self.device) if 'days_to_dividend' in self.raw_data else torch.zeros(3, device=self.device)

        # Create position info tensor directly on GPU
        position_info = torch.tensor([
            float(self.position),
            self.cash.item() if isinstance(self.cash, torch.Tensor) else float(self.cash),
            self.shares.item() if isinstance(self.shares, torch.Tensor) else float(self.shares),
            self.entry_price.item() if isinstance(self.entry_price, torch.Tensor) and self.entry_price is not None else 0.0
        ], device=self.device)

        # Flatten all features into a single vector
        state = torch.cat([price_data, indicator_data, correlation_data, dividend_data, position_info])
        
        # Validate state size
        expected_size = get_state_size(self.window_size)
        if state.size(0) != expected_size:
            raise ValueError(
                f"State size mismatch. Got {state.size(0)} features but expected {expected_size}.\n"
                f"Price data: {price_data.size(0)} features\n"
                f"Indicator data: {indicator_data.size(0)} features\n"
                f"Correlation data: {correlation_data.size(0)} features\n"
                f"Dividend data: {dividend_data.size(0)} features\n"
                f"Position info: {position_info.size(0)} features"
            )
            
        return state

    def reset(self, random_start=True):
        """Reset environment state with optional random start point"""
        # Calculate valid range for start index
        self.data_length = len(next(iter(self.raw_data.values())))
        
        # Ensure we have enough data for the window
        if self.data_length <= self.window_size:
            raise ValueError(f"Not enough data points ({self.data_length}) for window size {self.window_size}")
        
        # Calculate valid episode lengths
        max_possible_length = self.data_length - self.window_size
        self.min_episode_length = min(100, max_possible_length // 2)  # Ensure min length is reasonable
        self.max_episode_length = min(500, max_possible_length)       # Cap max length
        
        if random_start:
            # Calculate valid range for start index
            max_start = self.data_length - self.max_episode_length - self.window_size
            if max_start <= 0:
                # If we don't have enough data for random starts, just start at window_size
                self.start_idx = self.window_size
            else:
                self.start_idx = self.window_size + torch.randint(0, max_start, (1,)).item()
            
            # Choose episode length between min and max
            max_possible_from_start = min(
                self.max_episode_length,
                self.data_length - self.start_idx
            )
            self.episode_length = torch.randint(
                self.min_episode_length,
                max_possible_from_start + 1,  # +1 because randint is exclusive
                (1,)
            ).item()
        else:
            self.start_idx = self.window_size
            self.episode_length = min(self.max_episode_length, self.data_length - self.window_size)

        self.idx = self.start_idx
        self.step_count = 0
        self.done = False
        self.position = 0
        self.cash = torch.tensor(1.0, device=self.device)
        self.shares = torch.tensor(0.0, device=self.device)
        self.entry_price = None

        return self._get_state()

    def to_gpu(self, device):
        """Move environment data to GPU"""
        self.device = device

        # Move raw data to device
        self.raw_data = {
            k: v.to(device) for k, v in self.raw_data.items()
        }

        # Move indicators to device
        self.indicators = {
            k: v.to(device) for k, v in self.indicators.items()
        }

        # Move state buffer to device
        self.state_buffer = self.state_buffer.to(device)

    def calculate_portfolio_value(self):
        """Calculate total portfolio value in GPU"""
        current_price = self.raw_data['Close'][self.idx]
        if self.position == 0:
            return self.cash
        else:
            return self.shares * current_price

    def step(self, action):
        """Execute step with episode length limit"""
        # Check if already done
        if self.done:
            return self._get_state(), torch.tensor(0.0, device=self.device), True
            
        self.step_count += 1
        
        # Track portfolio value before action
        old_value = self.calculate_portfolio_value()
        
        current_price = self.raw_data['Close'][self.idx]
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.shares = self.cash / current_price
            self.cash = torch.tensor(0.0, device=self.device)
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            self.cash = self.shares * current_price
            self.shares = torch.tensor(0.0, device=self.device)
            self.position = 0
            self.entry_price = None
        
        # Move to next timestep
        self.idx += 1
        
        # Check termination conditions
        steps_remaining = (self.start_idx + self.episode_length) - self.idx
        self.done = (
            self.idx >= len(self.raw_data['Close']) - 1 or  # End of data
            steps_remaining <= 0 or  # Episode length reached
            self.step_count >= self.max_steps  # Max steps reached
        )
        
        # Calculate reward
        new_value = self.calculate_portfolio_value()
        reward = (new_value - old_value) / old_value
        
        return self._get_state(), reward, self.done

    def cleanup(self):
        """Clean up CUDA tensors"""
        self.raw_data = None
        self.indicators = None
        self.state_buffer = None
        torch.cuda.empty_cache()

def create_env_worker(args):
    """Worker function to create a single environment"""
    data, window_size, device = args
    # Create environment with CPU device first
    env = SimpleTradeEnvGPU(data, window_size=window_size, device='cpu')
    # Move to target device after creation
    if device != 'cpu':
        env.to_gpu(device)
    return env

def create_environments_batch_parallel(data_dict, window_size, batch_size, device, num_workers=None):
    """Create environments in parallel batches with caching"""
    print(f"\nInitializing environment pools for {device}...")
    env_pools = []

    for ticker, ticker_data in data_dict.items():
        print(f"\nCreating environments for {ticker}...")
        env_pool = []
        
        # Create environments in parallel on CPU first
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create arguments for each environment
            env_args = [(ticker_data, window_size, 'cpu') for _ in range(batch_size)]
            # Initialize environments in parallel
            for env in executor.map(create_env_worker, env_args):
                # Move to target device after creation
                if device != 'cpu':
                    env.to_gpu(device)
                env_pool.append(env)
        
        env_pools.append(env_pool)
        print(f"Completed pool of {len(env_pool)} environments")

    print(f"Initialized {len(env_pools)} pools")
    return env_pools[0] if env_pools else []  # Return just the first pool for now

def init_environments_parallel(data_dict, window_size, device, env_batch_size):
    """Initialize environments in parallel using process pool"""
    def init_env_worker(args):
        data, device = args
        # Create environment with CPU device first
        env = SimpleTradeEnvGPU(data, window_size=window_size, device='cpu')
        # Move to target device after creation
        env.to_gpu(device)
        return env

    with ProcessPoolExecutor() as executor:
        env_pools = []
        for data in data_dict.values():
            # Create arguments for each environment
            env_args = [(data, device) for _ in range(env_batch_size)]
            # Initialize batch of environments in parallel
            env_batch = list(executor.map(init_env_worker, env_args))
            env_pools.append(env_batch)

    return env_pools

def process_environment_batch(env_pool, shared_memory, device):
    """Process entire environment batch in parallel"""
    # Initialize environments
    states = torch.stack([env._get_state() for env in env_pool])

    while not all(env.done for env in env_pool):
        # Get actions (this runs on GPU)
        with torch.cuda.amp.autocast():
            q_values = policy_net(states)
            valid_action_masks = get_valid_actions_gpu(env_pool, device)
            actions = select_actions_gpu(q_values, epsilon, device)

        # Step environments in parallel chunks
        chunk_size = len(env_pool) // multiprocessing.cpu_count()
        chunks = [
            (env_pool[i:i + chunk_size], actions[i:i + chunk_size])
            for i in range(0, len(env_pool), chunk_size)
        ]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(step_environments_chunk, chunks))

        # Combine results
        next_states = torch.cat([r[0] for r in results])
        rewards = torch.cat([r[1] for r in results])
        dones = torch.cat([r[2] for r in results])

        states = next_states

def step_environments_chunk(chunk_data):
    """Process a chunk of environments in parallel"""
    envs, actions = chunk_data
    next_states = []
    rewards = []
    dones = []
    device = actions.device  # Get device from actions tensor
    
    for env, action in zip(envs, actions):
        try:
            next_state, reward, done = env.step(action.item())
            # Ensure tensors are on correct device
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).to(device)
            elif isinstance(next_state, torch.Tensor):
                next_state = next_state.to(device)
            
            if isinstance(reward, (int, float)):
                reward = torch.tensor(reward, device=device)
            elif isinstance(reward, torch.Tensor):
                reward = reward.to(device)
            
            if isinstance(done, bool):
                done = torch.tensor(done, device=device)
            elif isinstance(done, torch.Tensor):
                done = done.to(device)
                
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        except Exception as e:
            print(f"Error in process_env_batch: {str(e)}")
            raise e
    
    # Stack tensors (already on correct device)
    next_states = torch.stack(next_states)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    
    return next_states, rewards, dones 