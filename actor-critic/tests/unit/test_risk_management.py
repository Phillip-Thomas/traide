import pytest
import numpy as np
from src.utils.risk_management import RiskManager, RiskParams

@pytest.fixture
def risk_params():
    return RiskParams(
        max_position=1.0,
        max_leverage=1.0,
        position_step=0.1,
        max_drawdown=0.15,
        vol_lookback=20,
        vol_target=0.15,
        transaction_cost=0.001
    )

@pytest.fixture
def risk_manager(risk_params):
    return RiskManager(risk_params)

def test_risk_params_initialization(risk_params):
    """Test risk parameters initialization."""
    assert risk_params.max_position == 1.0
    assert risk_params.max_leverage == 1.0
    assert risk_params.position_step == 0.1
    assert risk_params.max_drawdown == 0.15
    assert risk_params.vol_lookback == 20
    assert risk_params.vol_target == 0.15
    assert risk_params.transaction_cost == 0.001

def test_risk_manager_initialization(risk_manager):
    """Test risk manager initialization."""
    assert risk_manager.current_drawdown == 0.0
    assert risk_manager.peak_value == 1.0
    assert len(risk_manager.position_history) == 0
    assert len(risk_manager.equity_curve) == 1
    assert risk_manager.equity_curve[0] == 1.0

def test_position_limits(risk_manager):
    """Test position limit calculations."""
    current_vol = 0.2  # Higher than target
    min_pos, max_pos = risk_manager.get_position_limits(current_vol)
    
    # Check limits are symmetric
    assert abs(min_pos) == max_pos
    
    # Check scaling by volatility
    expected_pos = risk_manager.params.vol_target / current_vol
    assert np.isclose(max_pos, expected_pos)
    
    # Check max leverage constraint
    assert max_pos <= risk_manager.params.max_leverage

def test_position_adjustment(risk_manager):
    """Test position size adjustment."""
    current_position = 0.0
    target_position = 0.8
    current_vol = 0.15  # Equal to target
    
    adjusted_position = risk_manager.adjust_position_size(
        current_position, target_position, current_vol
    )
    
    # Check position is quantized
    assert adjusted_position % risk_manager.params.position_step == 0
    
    # Check position is within limits
    min_pos, max_pos = risk_manager.get_position_limits(current_vol)
    assert min_pos <= adjusted_position <= max_pos

def test_transaction_cost(risk_manager):
    """Test transaction cost calculation."""
    current_position = 0.0
    new_position = 0.5
    
    cost = risk_manager.calculate_transaction_cost(current_position, new_position)
    
    expected_cost = abs(new_position - current_position) * risk_manager.params.transaction_cost
    assert np.isclose(cost, expected_cost)

def test_drawdown_update(risk_manager):
    """Test drawdown calculation and updates."""
    # Simulate portfolio value changes
    values = [1.0, 1.1, 1.05, 0.95, 1.0]
    
    for value in values:
        risk_manager.update_drawdown(value)
    
    # Check peak value tracking
    assert risk_manager.peak_value == 1.1
    
    # Check current drawdown
    expected_drawdown = (1.1 - 1.0) / 1.1
    assert np.isclose(risk_manager.current_drawdown, expected_drawdown)
    
    # Check equity curve
    assert len(risk_manager.equity_curve) == len(values) + 1  # Including initial value
    assert np.allclose(risk_manager.equity_curve[1:], values)

def test_risk_limits(risk_manager):
    """Test risk limit checks."""
    # Test safe scenario
    risk_manager.current_drawdown = 0.1
    is_safe, message = risk_manager.check_risk_limits()
    assert is_safe
    assert message is None
    
    # Test drawdown breach
    risk_manager.current_drawdown = 0.2
    is_safe, message = risk_manager.check_risk_limits()
    assert not is_safe
    assert "Maximum drawdown exceeded" in message

def test_small_position_changes(risk_manager):
    """Test handling of small position changes."""
    current_position = 0.5
    target_position = 0.51  # Small change below step size
    current_vol = 0.15
    
    adjusted_position = risk_manager.adjust_position_size(
        current_position, target_position, current_vol
    )
    
    # Should maintain current position for small changes
    assert adjusted_position == current_position

def test_extreme_volatility(risk_manager):
    """Test position sizing with extreme volatility."""
    current_position = 0.0
    target_position = 1.0
    
    # Test very high volatility
    high_vol = 1.0
    adjusted_high_vol = risk_manager.adjust_position_size(
        current_position, target_position, high_vol
    )
    
    # Test very low volatility
    low_vol = 0.01
    adjusted_low_vol = risk_manager.adjust_position_size(
        current_position, target_position, low_vol
    )
    
    # High volatility should reduce position size
    assert abs(adjusted_high_vol) < abs(adjusted_low_vol)
    
    # Both should respect max leverage
    assert abs(adjusted_high_vol) <= risk_manager.params.max_leverage
    assert abs(adjusted_low_vol) <= risk_manager.params.max_leverage 