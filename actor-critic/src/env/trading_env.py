import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import pandas as pd

class TradingEnvironment(gym.Env):
    """
    Minimal trading environment implementing the OpenAI Gym interface.
    Implements realistic trading mechanics including:
    - P&L calculation based on holding old position during price moves
    - Risk-adjusted rewards
    - Explicit transaction cost modeling
    """
    def __init__(
        self,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int = 100,
        commission: float = 0.001,
        reward_scaling: float = 1.0,
        risk_aversion: float = 0.1,
        vol_lookback: int = 20
    ):
        super().__init__()
        
        self.price_data = price_data
        self.features = features
        self.window_size = window_size
        self.commission = commission
        self.reward_scaling = reward_scaling
        self.risk_aversion = risk_aversion
        self.vol_lookback = vol_lookback
        
        # Set up dimensions
        self.n_assets = 1  # Simplified to single asset
        self.state_dim = len(features.columns) + 1  # Features + position
        
        # Initialize spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()

    def _calculate_state(self) -> np.ndarray:
        """Calculate the current state representation."""
        feature_values = self.features.iloc[self.current_step].values
        state = np.append(feature_values, self.current_position)
        return state.astype(np.float32)
    
    def _calculate_transaction_costs(self, old_position: float, new_position: float, 
                                  current_price: float, portfolio_value: float) -> float:
        """
        Calculate transaction costs with more realistic modeling.
        Includes both fixed commission and slippage based on size.
        """
        position_change = abs(new_position - old_position)
        
        # Base commission
        commission_cost = position_change * self.commission * portfolio_value
        
        return commission_cost 
    
    def _calculate_reward(self, step_return: float, position_size: float, portfolio_value: float) -> float:
        """
        Calculate reward based on returns with stronger emphasis on portfolio value.
        Encourages:
        - Taking larger positions when profitable
        - Maintaining high portfolio value
        - Quick recovery from drawdowns
        """
        # Base reward directly tied to portfolio value change
        base_reward = 200.0 * step_return  # Doubled base reward scaling
        
        # Portfolio value bonus/penalty (asymmetric)
        if portfolio_value >= 1.0:
            # Strong reward for being above water
            value_bonus = 2.0 * (portfolio_value - 1.0)
            base_reward *= (1.0 + value_bonus)
        else:
            # Reduced penalty when underwater to encourage recovery
            value_penalty = 0.5 * (1.0 - portfolio_value)
            base_reward *= (1.0 - value_penalty)
        
        # Position size incentive with minimum threshold
        size_incentive = 0.0
        if position_size >= 0.2:  # Minimum target position size
            if step_return > 0:
                # Strong reward for profitable large positions
                size_incentive = 1.0 * position_size * abs(step_return) * 100
            else:
                # Small penalty for unprofitable large positions
                size_incentive = -0.2 * position_size * abs(step_return) * 100
        else:
            # Significant penalty for too small positions
            size_incentive = -0.5  # Fixed penalty for small positions
        
        # Recovery bonus with increased magnitude
        recovery_bonus = 0.0
        if portfolio_value < 1.0 and step_return > 0:
            # Stronger recovery bonus proportional to how far underwater
            recovery_bonus = 3.0 * step_return * (1.0 - portfolio_value) * 100
        
        # Combine components
        reward = base_reward + size_incentive + recovery_bonus
        
        # Wider reward range for stronger learning signals
        reward = float(np.clip(reward, -5.0, 5.0))
        
        return reward

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with realistic trading mechanics."""
        # Store current values and implement proper order of operations
        old_value = self.portfolio_value
        old_position = self.current_position
        
        # Get current price (known at decision time)
        current_price = self.price_data.iloc[self.current_step, 0]
        
        # Calculate new position based on action (this happens at current price)
        new_position = float(np.clip(action[0] * 0.6, -0.6, 0.6))
        if abs(new_position) < 0.1:
            new_position = 0.0
            
        # Calculate transaction costs for position change at current price
        costs = self._calculate_transaction_costs(
            old_position, new_position, current_price, old_value
        )

        # Move to next step to get the price where P&L is realized
        self.current_step += 1
        next_price = self.price_data.iloc[self.current_step, 0]
        
        # Calculate P&L using price change after position was taken
        price_change_pct = (next_price - current_price) / current_price
        position_pnl = new_position * price_change_pct * old_value  # P&L from new position
        
        # Update portfolio value with minimum value constraint
        self.portfolio_value = max(old_value + position_pnl - costs, 0.01)
        
        # Calculate step return and update history
        step_return = (self.portfolio_value - old_value) / old_value
        self.returns_history.append(step_return)
        
        # Update position for next step
        self.current_position = new_position
        
        # Calculate reward
        reward = self._calculate_reward(
            step_return=step_return,
            position_size=abs(new_position),
            portfolio_value=self.portfolio_value
        )
        
        # Check if done
        done = (self.current_step >= len(self.price_data) - 2 or 
                self.portfolio_value <= 0.90)
        
        info = {
            'portfolio_value': float(self.portfolio_value),
            'step_return': step_return * 100.0,
            'position': float(new_position),
            'pnl': float(position_pnl),
            'costs': float(costs),
            'price_change': price_change_pct * 100.0,
            'current_price': float(current_price),
            'volatility': float(np.std(self.returns_history[-self.vol_lookback:]) if len(self.returns_history) >= self.vol_lookback else 0)
        }
        
        return self._calculate_state(), reward, done, False, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        print("Resetting environment")
        self.current_step = self.window_size
        self.current_position = 0.0
        self.portfolio_value = 1.0
        self.returns_history = []
        
        state = self._calculate_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.current_position
        }
        
        return state, info
