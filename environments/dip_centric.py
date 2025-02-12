class SimpleTradeEnv:
    def __init__(self, data, window_size=10, indicators=None, max_drawdown=0.15):
        self.data = data
        self.prices = self.data['Close'].values
        self.window_size = window_size
        self.indicators = indicators if indicators is not None else ["vwap"]
        self.max_drawdown = max_drawdown
        self.reset()

    def reset(self):
        self.idx = self.window_size
        self.position = 0  # 0: no position, 1: long
        self.cash = 1000  # Simplified starting capital
        self.shares = 0
        self.entry_price = None
        self.max_portfolio_value = self.cash
        self.last_action = 0  # Track last action for reward calculation
        return self._get_state()

    def _get_state(self):
        # Get the price window
        window = self.prices[self.idx - self.window_size:self.idx]
        # Calculate returns (length = window_size - 1)
        returns = np.diff(window) / window[:-1]
        # Position feature (1)
        position_feature = np.array([float(self.position)])
        
        # Unrealized return (1) if in position; else 0
        if self.position == 1:
            unrealized_return = [(self.prices[self.idx] - self.entry_price) / self.entry_price]
        else:
            unrealized_return = [0.0]
        
        # Compute additional technical indicators using the price window
        window_volumes = self.data['Volume'].values[self.idx - self.window_size:self.idx]
        vwap = compute_VWAP(window, window_volumes)
        rsi = compute_RSI(window)
        macd = compute_MACD(window)
        
        # Incorporate raw volume data:
        # Normalize the volume data by dividing by the mean to keep values on a similar scale.
        norm_volumes = window_volumes / (np.mean(window_volumes) + 1e-6)  # Add small constant to avoid division by zero

        # Concatenate all features:
        # - Price returns: (window_size - 1)
        # - Position feature: 1
        # - Unrealized return: 1
        # - Technical indicators: 3 (vwap, rsi, macd)
        # - Normalized raw volumes: (window_size)
        state = np.concatenate([returns, position_feature, unrealized_return, [vwap, rsi, macd], norm_volumes])
        return state


    def _get_reward(self, prev_value, current_value, action):
        # Base return calculation
        pct_return = (current_value - prev_value) / (prev_value + 1e-9)
        
        # Get RSI and MACD for timing
        rsi = compute_RSI(self.prices[max(0, self.idx - 15):self.idx + 1])
        macd = compute_MACD(self.prices[max(0, self.idx - 27):self.idx + 1])
        
        # Calculate multiple momentum timeframes
        short_window = self.prices[max(0, self.idx - 5):self.idx + 1]
        medium_window = self.prices[max(0, self.idx - 20):self.idx + 1]
        
        short_momentum = (short_window[-1] / short_window[0] - 1) if len(short_window) > 1 else 0
        medium_momentum = (medium_window[-1] / medium_window[0] - 1) if len(medium_window) > 1 else 0
        
        # Enhanced position holding bonus
        holding_bonus = 0.0
        if self.position == 1:
            duration = self.idx - self.entry_idx
            if pct_return > 0:
                holding_bonus = 0.0125 * np.log1p(duration)  # Doubled bonus for holding
        
        # Enhanced buy/sell timing rewards
        timing_reward = 0.0
        if action == 1:  # Buy
            # Strong buy conditions (multiple timeframes)
            if short_momentum < -0.001 and medium_momentum < -0.02 and rsi < 40 and macd < 0:
                timing_reward = 0.1  # Increased reward for strong signals
            elif short_momentum < -0.001 and rsi < 40:
                timing_reward = 0.025  # Small reward for moderate signals
            else:
                timing_reward = -0.005  # Stronger penalty for buying without signals
                
        elif action == 2:  # Sell
            # Strong sell conditions
            if short_momentum > 0.001 and rsi > 60:
                timing_reward = 0.1
            elif self.position == 1:
                if pct_return > 0.02:  # 2% profit target
                    timing_reward = 0.04  # Reward for taking significant profits
                elif pct_return < 0:
                    timing_reward = -0.01  # Penalty for selling at loss
        
        # Base reward scaling
        intermediate_reward = pct_return * 100.0
        
        # Increased transaction costs
        if action in [1, 2]:
            intermediate_reward -= 0.005  # Doubled transaction cost
        
        # Sell completion reward
        if action == 2 and self.position == 1:
            trade_return = (self.prices[self.idx] - self.entry_price) / (self.entry_price + 1e-9)
            trade_reward = trade_return * 100.0
            return trade_reward + intermediate_reward + timing_reward
        
        return intermediate_reward + holding_bonus + timing_reward

    def step(self, action):
        done = False
        prev_value = self.cash if self.position == 0 else self.shares * self.prices[self.idx]
        
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = self.prices[self.idx]
            self.entry_idx = self.idx
            self.shares = self.cash / self.prices[self.idx]
            self.cash = 0
            
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.cash = self.shares * self.prices[self.idx]
            self.shares = 0
            self.entry_price = None
        
        self.idx += 1
        if self.idx >= len(self.prices):
            done = True
            if self.position == 1:
                self.cash = self.shares * self.prices[-1]
                self.shares = 0
                self.position = 0
            return self._get_state(), 0, done
        
        current_value = self.cash if self.position == 0 else self.shares * self.prices[self.idx]
        reward = self._get_reward(prev_value, current_value, action)
        
        # Update maximum portfolio value and check drawdown
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if drawdown > self.max_drawdown:
            done = True
            reward = -1  # Penalty for exceeding drawdown limit
        
        self.last_action = action
        return self._get_state(), reward, done
