import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class RiskParams:
    """Risk management parameters"""
    max_position: float = 1.0  # Maximum allowed position size
    max_leverage: float = 1.0  # Maximum allowed leverage
    position_step: float = 0.1  # Minimum position change increment
    max_drawdown: float = 0.15  # Maximum allowed drawdown
    vol_lookback: int = 20  # Volatility lookback period
    vol_target: float = 0.15  # Target annualized volatility
    transaction_cost: float = 0.001  # Per-trade transaction cost

class RiskManager:
    """
    Risk management system for trading operations.
    Handles position sizing, drawdown control, and transaction cost modeling.
    """
    def __init__(self, risk_params: RiskParams):
        self.params = risk_params
        self.current_drawdown = 0.0
        self.peak_value = 1.0
        self.position_history = []
        self.equity_curve = [1.0]
        
    def get_position_limits(self, current_vol: float) -> Tuple[float, float]:
        """
        Calculate position limits based on volatility and risk parameters.
        
        Args:
            current_vol: Current market volatility (can be scalar or pandas Series)
            
        Returns:
            min_position: Minimum allowed position
            max_position: Maximum allowed position
        """
        # Convert to float if pandas Series
        if hasattr(current_vol, 'iloc'):
            current_vol = float(current_vol.iloc[-1])
        elif hasattr(current_vol, 'item'):
            current_vol = float(current_vol.item())
        else:
            current_vol = float(current_vol)
        
        # Scale position limits by volatility
        vol_scalar = self.params.vol_target / max(current_vol, 1e-6)
        max_pos = min(self.params.max_position * vol_scalar, self.params.max_leverage)
        
        return -max_pos, max_pos
    
    def adjust_position_size(
        self,
        current_position: float,
        target_position: float,
        current_vol: float
    ) -> float:
        """
        Adjust target position based on risk constraints.
        
        Args:
            current_position: Current position size
            target_position: Desired target position
            current_vol: Current market volatility
            
        Returns:
            adjusted_position: Risk-adjusted position size
        """
        # Convert inputs to float scalars
        if hasattr(current_position, 'iloc'):
            current_position = float(current_position.iloc[-1])
        elif hasattr(current_position, 'item'):
            current_position = float(current_position.item())
        else:
            current_position = float(current_position)
            
        if hasattr(target_position, 'iloc'):
            target_position = float(target_position.iloc[-1])
        elif hasattr(target_position, 'item'):
            target_position = float(target_position.item())
        else:
            target_position = float(target_position)
        
        # Handle NaN values
        if np.isnan(current_position) or np.isnan(target_position) or np.isnan(current_vol):
            return 0.0  # Default to flat position if we have NaN values
            
        min_pos, max_pos = self.get_position_limits(current_vol)
        
        # Clamp position within limits
        target_position = np.clip(target_position, min_pos, max_pos)
        
        # Quantize position changes to reduce trading frequency
        position_change = target_position - current_position
        if abs(position_change) < self.params.position_step:
            return current_position
            
        # Round to position step size
        steps = round(position_change / self.params.position_step)
        return current_position + steps * self.params.position_step
    
    def calculate_transaction_cost(
        self,
        current_position: float,
        new_position: float
    ) -> float:
        """
        Calculate transaction cost for position change.
        
        Args:
            current_position: Current position size
            new_position: New target position size
            
        Returns:
            cost: Transaction cost for the trade
        """
        position_change = abs(new_position - current_position)
        return position_change * self.params.transaction_cost
    
    def update_drawdown(self, portfolio_value: float) -> None:
        """
        Update drawdown metrics based on new portfolio value.
        
        Args:
            portfolio_value: Current portfolio value
        """
        # Convert to float if pandas Series
        if hasattr(portfolio_value, 'iloc'):
            portfolio_value = float(portfolio_value.iloc[-1])
        elif hasattr(portfolio_value, 'item'):
            portfolio_value = float(portfolio_value.item())
        else:
            portfolio_value = float(portfolio_value)
            
        self.peak_value = max(self.peak_value, portfolio_value)
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.equity_curve.append(portfolio_value)
    
    def check_risk_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any risk limits have been breached.
        
        Returns:
            is_safe: Whether position is within risk limits
            message: Description of risk breach if any
        """
        if self.current_drawdown > self.params.max_drawdown:
            return False, f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
            
        return True, None 