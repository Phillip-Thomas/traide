# environment/trading_env.py
import numpy as np

def get_state_size(window_size):
    """
    Calculate input size based on window size and feature count
    Features per timestep:
    - OHLCV (5)
    - Returns (1)
    - SMA (1)
    - Volatility (1)
    - RSI (1)
    Plus:
    - Position (1)
    - Max Drawdown (1)
    - Max Profit (1)
    """
    features_per_timestep = 9  # OHLCV + returns + sma + volatility + rsi
    additional_features = 3    # position + max_drawdown + max_profit
    
    return window_size * features_per_timestep + additional_features

class SimpleTradeEnv:
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size
        
        # Store raw prices
        self.raw_open = self.data['Open'].values
        self.raw_high = self.data['High'].values
        self.raw_low = self.data['Low'].values
        self.raw_close = self.data['Close'].values
        self.raw_volume = self.data['Volume'].values
        
        # Normalize prices using the starting point
        scale = self.raw_close[window_size] if self.raw_close[window_size] != 0 else 1.0
        self.open_prices = self.raw_open / scale
        self.high_prices = self.raw_high / scale
        self.low_prices = self.raw_low / scale
        self.close_prices = self.raw_close / scale
        
        # Normalize volume
        scale_vol = self.raw_volume[window_size] if self.raw_volume[window_size] != 0 else 1.0
        self.volume = self.raw_volume / scale_vol
        
        # Calculate technical indicators
        self.calculate_indicators()
        
        self.reset()

    def get_valid_actions(self):
        if self.position == 0:
            return [0, 1]  # HOLD, BUY
        else:
            return [0, 2]  # HOLD, SELL

    def calculate_indicators(self):
        """Pre-calculate technical indicators for efficiency"""
        prices = self.raw_close
        self.returns = np.zeros_like(prices)
        self.returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Momentum indicators
        window = 20
        self.sma = np.zeros_like(prices)
        self.volatility = np.zeros_like(prices)
        self.rsi = np.zeros_like(prices)
        
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            returns_window = self.returns[i-window:i]
            
            self.sma[i] = np.mean(price_window)
            self.volatility[i] = np.std(returns_window)
            
            # RSI
            gains = np.sum(np.maximum(returns_window, 0))
            losses = np.sum(np.maximum(-returns_window, 0))
            
            if losses == 0:
                self.rsi[i] = 100
            else:
                rs = gains / losses
                self.rsi[i] = 100 - (100 / (1 + rs))
    
    def reset(self):
        self.idx = self.window_size
        self.position = 0
        self.cash = 1.0
        self.shares = 0
        self.entry_price = None
        self.entry_idx = None
        self.max_drawdown = 0
        self.max_profit = 0
        return self._get_state()
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (HOLD), 1 (BUY), or 2 (SELL)
        Returns:
            tuple of (next_state, reward, done)
        """
        # Check if we're at the end of data
        if self.idx >= len(self.close_prices) - 1:  # Need one more price for next state
            if self.position == 1:
                self.cash = self.shares * self.close_prices[self.idx]
                self.shares = 0
                self.position = 0
            return self._get_state(), 0, True
        
        # Execute action
        old_portfolio_value = self.cash if self.position == 0 else self.shares * self.close_prices[self.idx]
        
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = self.close_prices[self.idx]
            self.entry_idx = self.idx
            self.shares = self.cash / self.entry_price
            self.cash = 0
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.cash = self.shares * self.close_prices[self.idx]
            self.shares = 0
            self.entry_price = None
            self.entry_idx = None
        
        # Calculate reward before moving to next state
        reward = self._calculate_reward(action)
        
        # Move to next step
        self.idx += 1
        
        # Check if we've reached the end after moving
        done = self.idx >= len(self.close_prices) - 1
        if done and self.position == 1:
            self.cash = self.shares * self.close_prices[self.idx]
            self.shares = 0
            self.position = 0
        
        # Add penalty for excessive trading
        if action != 0:  # Trading cost
            reward -= 0.001
        
        return self._get_state(), reward, done

    def _calculate_reward(self, action):
        """Calculate reward for the current action."""
        # Safety check for index
        if self.idx >= len(self.close_prices):
            return 0
            
        current_price = self.close_prices[self.idx]
        reward = 0
        
        # Technical indicators for reward shaping
        momentum = self.returns[self.idx]
        volatility = self.volatility[self.idx]
        rsi = self.rsi[self.idx]
        trend = (current_price - self.sma[self.idx]) / (current_price + 1e-6)  # Avoid division by zero
        
        if action == 1:  # Buy
            if self.position == 0:
                # Reward buying in favorable conditions
                reward = 0.001  # Base reward for taking action
                
                # Add rewards for good entry conditions
                if rsi < 30:  # Oversold
                    reward += 0.002
                if trend < -0.02:  # Price below SMA
                    reward += 0.002
                if volatility > np.mean(self.volatility):  # Higher volatility
                    reward += 0.001
                
        elif action == 2:  # Sell
            if self.position == 1:
                # Calculate profit/loss
                profit = (current_price - self.entry_price) / self.entry_price
                holding_time = self.idx - self.entry_idx
                
                # Base reward on profit
                reward = profit
                
                # Add rewards for good exit conditions
                if rsi > 70:  # Overbought
                    reward += 0.002
                if trend > 0.02:  # Price above SMA
                    reward += 0.002
                
                # Scale reward by holding time
                reward *= min(1.0, holding_time / 20)  # Cap at 20 periods
                
                # Update maximum values
                self.max_profit = max(self.max_profit, profit)
                self.max_drawdown = min(self.max_drawdown, profit)
        
        # Small penalty for holding during unfavorable conditions
        elif action == 0 and self.position == 1:
            if rsi > 80 or trend < -0.03:
                reward -= 0.001
        
        return reward

    def _get_state(self):
        """
        Returns state as a 1D array with shape (window_size * features_per_timestep + additional_features,)
        """
        idx = self.idx
        
        # Safety check for window boundaries
        if idx < self.window_size:
            idx = self.window_size
        elif idx >= len(self.close_prices):
            idx = len(self.close_prices) - 1
            
        window = slice(idx - self.window_size, idx)
        
        # Price and volume data
        ohlcv_data = np.column_stack([
            self.open_prices[window],
            self.high_prices[window],
            self.low_prices[window],
            self.close_prices[window],
            self.volume[window]
        ]).flatten()
        
        # Technical indicators
        tech_indicators = np.column_stack([
            self.returns[window],
            self.sma[window],
            self.volatility[window],
            self.rsi[window]
        ]).flatten()
        
        # Additional state information
        position_info = np.array([
            float(self.position),
            self.max_drawdown,
            self.max_profit
        ])
        
        # Combine all features
        state = np.concatenate([
            ohlcv_data,
            tech_indicators,
            position_info
        ])
        
        return state