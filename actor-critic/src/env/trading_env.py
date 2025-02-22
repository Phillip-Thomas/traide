import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import pandas as pd
import logging
from collections import deque

from ..utils.risk_management import RiskManager, RiskParams

class TradingEnvironment(gym.Env):
    """
    Trading environment implementing the OpenAI Gym interface.
    Handles market state management, reward calculation, and risk constraints.
    """
    def __init__(
        self,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        risk_params: Optional[RiskParams] = None,
        window_size: int = 50,
        commission: float = 0.001,
    ):
        super().__init__()
        
        self.price_data = price_data
        self.features = features
        self.window_size = window_size
        self.commission = commission
        
        # Determine if we have asset-specific columns or simple columns
        if any('ASSET' in col for col in price_data.columns):
            self.n_assets = len([col for col in price_data.columns if 'close' in col.lower()])
            self.column_format = 'asset_specific'
        else:
            self.n_assets = 1
            self.column_format = 'simple'
        
        # Initialize risk manager
        self.risk_manager = RiskManager(risk_params or RiskParams())
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),  # One action per asset
            dtype=np.float32
        )
        
        # State space includes features and positions for all assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1] + self.n_assets,),  # +n_assets for positions
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = window_size
        self.current_positions = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        self.returns_history = deque(maxlen=20)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def _calculate_state(self) -> np.ndarray:
        """
        Calculate the current state representation.
        
        Returns:
            state: Current state array including features and positions
        """
        try:
            feature_values = self.features.iloc[self.current_step].values
            state = np.append(feature_values, self.current_positions)
            return state.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in _calculate_state: {str(e)}")
            raise
    
    def _calculate_reward(self, old_value: float, new_value: float) -> float:
        """
        Calculate the step reward based on portfolio value change.
        
        Args:
            old_value: Previous portfolio value
            new_value: Current portfolio value
            
        Returns:
            reward: Step reward (Sharpe-like ratio)
        """
        try:
            # Ensure values are valid
            if not (np.isfinite(old_value) and np.isfinite(new_value)):
                self.logger.warning(f"Invalid portfolio values: old={old_value}, new={new_value}")
                return 0.0
                
            if old_value <= 0 or new_value <= 0:
                self.logger.warning(f"Non-positive portfolio values: old={old_value}, new={new_value}")
                return 0.0
            
            # Calculate current return with numerical stability
            current_return = np.clip((new_value - old_value) / (old_value + 1e-8), -1.0, 1.0)
            
            # Add to history if it's a valid number
            if np.isfinite(current_return):
                self.returns_history.append(float(current_return))
            
            # Need at least 2 returns to calculate meaningful statistics
            if len(self.returns_history) < 2:
                return 0.0
            
            # Calculate statistics with numerical stability
            returns = np.array(self.returns_history)
            daily_mean = np.mean(returns)
            daily_std = np.std(returns, ddof=1) + 1e-8  # Add small constant for stability
            
            # Calculate Sharpe ratio using recent returns
            sharpe = np.clip(daily_mean / daily_std, -1.0, 1.0)  # Clip to reasonable range
            
            return float(sharpe)
            
        except Exception as e:
            self.logger.error(f"Error in _calculate_reward: {str(e)}")
            return 0.0  # Return safe value on error
    
    def _log_trade_info(self, old_positions: np.ndarray, new_positions: np.ndarray, 
                        current_prices: np.ndarray, transaction_costs: float) -> None:
        """Log meaningful trade information."""
        position_changes = new_positions - old_positions
        
        # Only log when there are actual position changes
        significant_changes = np.abs(position_changes) > 1e-6
        if np.any(significant_changes):
            trade_details = []
            for i in range(self.n_assets):
                if abs(position_changes[i]) > 1e-6:
                    asset_name = f"ASSET_{i+1}" if self.column_format == 'asset_specific' else "ASSET"
                    trade_size = position_changes[i]
                    notional_value = abs(trade_size * current_prices[i])
                    direction = "BOUGHT" if trade_size > 0 else "SOLD"
                    trade_details.append(
                        f"{direction} {abs(trade_size):.4f} units of {asset_name} "
                        f"(Notional: ${notional_value:.2f})"
                    )
            
            self.logger.info(
                f"Trade executed at step {self.current_step}:\n"
                f"  {'  '.join(trade_details)}\n"
                f"  Transaction costs: ${transaction_costs:.4f}\n"
                f"  New portfolio value: ${self.portfolio_value:.2f}"
            )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Position sizing action from policy
            
        Returns:
            next_state: Next state observation
            reward: Step reward
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        try:
            # Store current portfolio value before updates
            old_value = float(self.portfolio_value)
            
            # Ensure action is numpy array with correct shape and values
            action = np.asarray(action, dtype=np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)  # Handle invalid values
            action = np.clip(action, -1.0, 1.0)  # Ensure action is in valid range
            
            if action.ndim == 0:
                action = np.array([action])
            if action.ndim == 1 and len(action) == 1:
                action = np.repeat(action, self.n_assets)
            if action.shape != (self.n_assets,):
                raise ValueError(f"Action shape {action.shape} does not match number of assets {self.n_assets}")
            
            # Get current and next prices for all assets
            try:
                if self.column_format == 'asset_specific':
                    current_prices = np.array([
                        float(self.price_data.iloc[self.current_step][f'close_ASSET_{i+1}'])
                        for i in range(self.n_assets)
                    ])
                    next_prices = np.array([
                        float(self.price_data.iloc[self.current_step + 1][f'close_ASSET_{i+1}'])
                        for i in range(self.n_assets)
                    ])
                else:
                    current_prices = np.array([float(self.price_data.iloc[self.current_step]['close'])])
                    next_prices = np.array([float(self.price_data.iloc[self.current_step + 1]['close'])])
                
                # Ensure prices are valid
                current_prices = np.nan_to_num(current_prices, nan=1.0, posinf=1.0, neginf=1.0)
                next_prices = np.nan_to_num(next_prices, nan=1.0, posinf=1.0, neginf=1.0)
                
                # Ensure prices are positive
                current_prices = np.maximum(current_prices, 1e-8)
                next_prices = np.maximum(next_prices, 1e-8)
                
            except Exception as e:
                self.logger.error(f"Error getting prices: {str(e)}")
                return self._calculate_state(), 0.0, True, False, {'error': str(e)}
            
            # Calculate returns for each asset with numerical stability
            price_returns = np.clip((next_prices - current_prices) / (current_prices + 1e-8), -1.0, 1.0)
            
            # Get target positions from actions
            target_positions = action
            
            # Store old positions for logging
            old_positions = np.asarray(self.current_positions, dtype=np.float32)
            
            # Calculate volatilities with numerical stability
            current_vols = []
            for i in range(self.n_assets):
                try:
                    if self.current_step < self.window_size:
                        if self.column_format == 'asset_specific':
                            prices = self.price_data.iloc[:self.current_step+1][f'close_ASSET_{i+1}']
                        else:
                            prices = self.price_data.iloc[:self.current_step+1]['close']
                    else:
                        if self.column_format == 'asset_specific':
                            prices = self.price_data.iloc[self.current_step-self.window_size:self.current_step+1][f'close_ASSET_{i+1}']
                        else:
                            prices = self.price_data.iloc[self.current_step-self.window_size:self.current_step+1]['close']
                    
                    # Clean prices
                    prices = np.nan_to_num(prices.values, nan=1.0, posinf=1.0, neginf=1.0)
                    prices = np.maximum(prices, 1e-8)
                    
                    # Calculate returns and volatility
                    returns = np.diff(prices) / (prices[:-1] + 1e-8)
                    returns = np.clip(returns, -1.0, 1.0)  # Clip extreme returns
                    vol = float(np.std(returns) if len(returns) > 0 else 0.01)
                    vol = np.clip(vol, 0.01, 1.0)  # Ensure reasonable volatility range
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating volatility for asset {i}: {str(e)}")
                    vol = 0.01  # Use safe default
                    
                current_vols.append(vol)
            
            current_vols = np.array(current_vols, dtype=np.float32)
            
            # Update positions with risk management
            new_positions = np.zeros(self.n_assets, dtype=np.float32)
            for i in range(self.n_assets):
                new_positions[i] = self.risk_manager.adjust_position_size(
                    float(old_positions[i]),
                    float(target_positions[i]),
                    float(current_vols[i])
                )
            self.current_positions = new_positions
            
            # Calculate position returns and apply to portfolio value with numerical stability
            position_returns = np.sum(self.current_positions * price_returns) * old_value
            position_returns = np.clip(position_returns, -old_value, old_value)  # Prevent extreme values
            
            # Calculate transaction costs
            transaction_costs = sum(
                self.risk_manager.calculate_transaction_cost(float(old_pos), float(new_pos))
                for old_pos, new_pos in zip(old_positions, self.current_positions)
            )
            transaction_costs = np.clip(transaction_costs, 0.0, old_value)  # Prevent extreme costs
            
            # Log trade information
            self._log_trade_info(old_positions, new_positions, current_prices, transaction_costs)
            
            # Update portfolio value with returns and costs
            self.portfolio_value = old_value + position_returns - transaction_costs
            self.portfolio_value = np.maximum(self.portfolio_value, 1e-8)  # Ensure positive value
            
            # Calculate reward
            reward = self._calculate_reward(old_value, self.portfolio_value)
            
            # Prepare next state
            next_state = self._calculate_state()
            
            # Check if episode should end
            done = self.current_step >= len(self.price_data) - 2
            
            # Increment step
            self.current_step += 1
            
            # Prepare info dict
            info = {
                'portfolio_value': float(self.portfolio_value),
                'position_returns': float(position_returns),
                'transaction_costs': float(transaction_costs),
                'current_positions': self.current_positions.tolist(),
                'price_returns': price_returns.tolist(),
                'volatilities': current_vols.tolist()
            }
            
            return next_state, reward, done, False, info
            
        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}")
            # Return safe values on error
            return self._calculate_state(), 0.0, True, False, {'error': str(e)}
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            state: Initial state
            info: Additional information
        """
        try:
            self.logger.info("Resetting environment...")
            super().reset(seed=seed)
            
            self.current_step = self.window_size
            self.current_positions = np.zeros(self.n_assets, dtype=np.float32)
            self.portfolio_value = 1.0
            self.returns_history = deque(maxlen=20)
            
            state = self._calculate_state()
            
            info = {
                "portfolio_value": self.portfolio_value,
                "positions": self.current_positions.tolist()
            }
            
            return state, info
            
        except Exception as e:
            self.logger.error(f"Error in reset: {str(e)}")
            raise 