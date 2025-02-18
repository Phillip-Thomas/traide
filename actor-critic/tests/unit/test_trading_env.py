import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from src.env.trading_env import TradingEnvironment
from src.utils.risk_management import RiskParams

@pytest.fixture
def sample_data():
    """Create sample price and feature data for testing."""
    # Generate synthetic price data for 2 assets
    n_samples = 200  # Increased from 100 to ensure enough data for volatility calculation
    np.random.seed(42)
    
    # Generate more realistic price data with trends
    time = np.linspace(0, 4*np.pi, n_samples)
    trend1 = np.sin(time) * 10 + time * 2
    trend2 = np.cos(time) * 5 + time * 1.5
    
    noise1 = np.random.randn(n_samples)
    noise2 = np.random.randn(n_samples)
    
    price_data = pd.DataFrame({
        'close_ASSET_1': 100 + trend1 + noise1,
        'high_ASSET_1': 102 + trend1 + noise1 + np.abs(np.random.randn(n_samples)),
        'low_ASSET_1': 98 + trend1 + noise1 - np.abs(np.random.randn(n_samples)),
        'volume_ASSET_1': np.random.randint(1000, 10000, n_samples),
        'close_ASSET_2': 50 + trend2 + noise2,
        'high_ASSET_2': 52 + trend2 + noise2 + np.abs(np.random.randn(n_samples)),
        'low_ASSET_2': 48 + trend2 + noise2 - np.abs(np.random.randn(n_samples)),
        'volume_ASSET_2': np.random.randint(1000, 10000, n_samples)
    })
    
    # Generate synthetic features
    features = pd.DataFrame({
        'feature1_ASSET_1': np.random.randn(n_samples),
        'feature2_ASSET_1': np.random.randn(n_samples),
        'feature1_ASSET_2': np.random.randn(n_samples),
        'feature2_ASSET_2': np.random.randn(n_samples)
    })
    
    return price_data, features

@pytest.fixture
def trading_env(sample_data):
    """Create a trading environment for testing."""
    price_data, features = sample_data
    risk_params = RiskParams(
        max_position=1.0,
        max_leverage=1.0,
        position_step=0.5,
        max_drawdown=0.15,
        vol_lookback=20,
        vol_target=0.15,
        transaction_cost=0.001
    )
    return TradingEnvironment(
        price_data=price_data,
        features=features,
        risk_params=risk_params,
        window_size=10,
        commission=0.001
    )

def test_trading_env_initialization(trading_env, sample_data):
    """Test environment initialization."""
    price_data, features = sample_data
    
    # Check spaces
    assert isinstance(trading_env.action_space, gym.spaces.Box)
    assert isinstance(trading_env.observation_space, gym.spaces.Box)
    
    # Check dimensions
    assert trading_env.action_space.shape == (2,)  # 2 assets
    assert trading_env.observation_space.shape == (6,)  # 4 features + 2 positions
    
    # Check bounds
    assert np.all(trading_env.action_space.low == -1.0)
    assert np.all(trading_env.action_space.high == 1.0)
    
    # Check initial state
    assert trading_env.current_step == 10  # window_size
    assert np.all(trading_env.current_positions == 0)
    assert trading_env.portfolio_value == 1.0

def test_trading_env_reset(trading_env):
    """Test environment reset."""
    # Take some random actions
    for _ in range(5):
        action = trading_env.action_space.sample()
        trading_env.step(action)
    
    # Reset environment
    initial_state, info = trading_env.reset()
    
    # Check reset state
    assert trading_env.current_step == 10  # window_size
    assert np.all(trading_env.current_positions == 0)
    assert trading_env.portfolio_value == 1.0
    assert len(trading_env.returns_history) == 0
    
    # Check info dict
    assert 'portfolio_value' in info
    assert 'positions' in info
    assert info['portfolio_value'] == 1.0
    assert len(info['positions']) == 2  # 2 assets

def test_trading_env_step(trading_env):
    """Test environment step."""
    # Take a step with zero positions
    action = np.zeros(2)
    next_state, reward, done, truncated, info = trading_env.step(action)
    
    # Check state
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (6,)  # 4 features + 2 positions
    
    # Check reward
    assert isinstance(reward, float)
    
    # Check info
    assert 'portfolio_value' in info
    assert 'positions' in info
    assert 'transaction_cost' in info
    assert 'risk_message' in info
    
    # Take a step with non-zero positions
    action = np.array([0.5, -0.3])
    next_state, reward, done, truncated, info = trading_env.step(action)
    
    # Check position updates
    assert len(info['positions']) == 2
    assert abs(info['positions'][0]) <= 1.0  # Max position limit
    assert abs(info['positions'][1]) <= 1.0

def test_trading_env_done_conditions(trading_env):
    """Test environment termination conditions."""
    # Test episode length limit
    while True:
        action = trading_env.action_space.sample()
        _, _, done, _, _ = trading_env.step(action)
        if done:
            break
    
    assert trading_env.current_step >= len(trading_env.price_data) - 1 or \
           not trading_env.risk_manager.check_risk_limits()[0]

def test_trading_env_reward_calculation(trading_env):
    """Test reward calculation."""
    # Test reward with zero positions
    action = np.zeros(2)
    _, reward, _, _, _ = trading_env.step(action)
    assert reward == 0  # No position should give zero reward initially
    
    # Test reward with positions
    action = np.array([0.5, -0.3])
    _, reward, _, _, info = trading_env.step(action)
    assert isinstance(reward, float)
    assert np.isfinite(reward)

def test_trading_env_transaction_costs(trading_env):
    """Test transaction cost calculation."""
    # Reset to ensure we start from zero positions
    initial_state, info = trading_env.reset()
    print(f"\nInitial positions: {info['positions']}")
    
    # First action - large position change from zero
    action1 = np.array([1.0, 1.0])
    print(f"\nAction 1: {action1}")
    _, _, _, _, info1 = trading_env.step(action1)
    cost1 = info1['transaction_cost']
    positions1 = info1['positions']
    print(f"Positions after action 1: {positions1}")
    print(f"Cost after action 1: {cost1}")
    
    # Second action - complete reversal
    action2 = np.array([-1.0, -1.0])
    print(f"\nAction 2: {action2}")
    _, _, _, _, info2 = trading_env.step(action2)
    cost2 = info2['transaction_cost']
    positions2 = info2['positions']
    print(f"Positions after action 2: {positions2}")
    print(f"Cost after action 2: {cost2}")
    
    # Third action - back to neutral
    action3 = np.array([0.0, 0.0])
    print(f"\nAction 3: {action3}")
    _, _, _, _, info3 = trading_env.step(action3)
    cost3 = info3['transaction_cost']
    positions3 = info3['positions']
    print(f"Positions after action 3: {positions3}")
    print(f"Cost after action 3: {cost3}")
    
    # Get volatility for debugging
    current_vols = np.array([
        float(np.std(trading_env.price_data.iloc[trading_env.current_step-20:trading_env.current_step][f'close_ASSET_{i+1}']))
        for i in range(trading_env.n_assets)
    ])
    print(f"\nCurrent volatilities: {current_vols}")
    
    # Get position limits for debugging
    min_pos, max_pos = trading_env.risk_manager.get_position_limits(current_vols[0])
    print(f"Position limits: [{min_pos}, {max_pos}]")
    print(f"Position step size: {trading_env.risk_manager.params.position_step}")
    
    # Verify transaction costs
    assert cost1 > 0, f"Initial position change should incur costs, got {cost1}"
    assert cost2 > 0, f"Position reversal should incur costs, got {cost2}"
    assert cost3 > 0, f"Position change should incur costs, got {cost3}"
    
    # Verify position changes
    assert np.any(np.abs(positions1) > 0), f"Positions should change, got {positions1}"
    assert np.any(np.abs(positions2) > 0), f"Positions should change, got {positions2}"
    
    # Verify costs are proportional to position changes
    pos_change1 = np.sum(np.abs(positions1))
    pos_change2 = np.sum(np.abs(np.array(positions2) - np.array(positions1)))
    pos_change3 = np.sum(np.abs(np.array(positions3) - np.array(positions2)))
    
    print(f"\nPosition changes:")
    print(f"Change 1: {pos_change1}")
    print(f"Change 2: {pos_change2}")
    print(f"Change 3: {pos_change3}")
    
    assert cost1 == pos_change1 * trading_env.risk_manager.params.transaction_cost, \
        f"Cost should be proportional to position change: cost={cost1}, change={pos_change1}"
    assert cost2 == pos_change2 * trading_env.risk_manager.params.transaction_cost, \
        f"Cost should be proportional to position change: cost={cost2}, change={pos_change2}"
    assert cost3 == pos_change3 * trading_env.risk_manager.params.transaction_cost, \
        f"Cost should be proportional to position change: cost={cost3}, change={pos_change3}"

def test_trading_env_risk_limits(trading_env):
    """Test risk management limits."""
    # Reset to ensure we start from zero positions
    trading_env.reset()
    
    # Test max position limit
    action = np.array([2.0, 2.0])  # Exceeds max position
    _, _, _, _, info = trading_env.step(action)
    assert all(abs(pos) <= 1.0 for pos in info['positions'])
    
    # Test max leverage
    action = np.array([0.5, 0.5])  # Half max position for each asset
    _, _, _, _, info = trading_env.step(action)
    total_exposure = sum(abs(pos) for pos in info['positions'])
    assert total_exposure <= trading_env.risk_manager.params.max_leverage, \
        f"Total exposure {total_exposure} exceeds max leverage {trading_env.risk_manager.params.max_leverage}"
    
    # Test position step size
    action = np.array([0.1, 0.1])  # Small change below step size
    _, _, _, _, info = trading_env.step(action)
    position_changes = np.abs(np.array(info['positions']) - np.array([0.5, 0.5]))
    assert all(change == 0 or change >= trading_env.risk_manager.params.position_step 
              for change in position_changes), \
        f"Position changes {position_changes} smaller than step size {trading_env.risk_manager.params.position_step}"

def test_trading_env_state_calculation(trading_env):
    """Test state calculation."""
    state, _ = trading_env.reset()
    
    # Check state components
    assert len(state) == trading_env.observation_space.shape[0]
    assert np.all(np.isfinite(state))
    
    # Check position component
    positions_part = state[-2:]
    assert np.all(positions_part == 0)  # Initial positions should be zero

def test_trading_env_edge_cases(trading_env):
    """Test edge cases and error handling."""
    # Test with NaN action
    action = np.array([np.nan, np.nan])
    next_state, reward, done, _, info = trading_env.step(action)
    assert np.all(np.isfinite(next_state))
    assert np.isfinite(reward)
    assert all(np.isfinite(pos) for pos in info['positions'])
    
    # Test with extreme action values
    action = np.array([1e6, -1e6])
    next_state, reward, done, _, info = trading_env.step(action)
    assert np.all(np.isfinite(next_state))
    assert np.isfinite(reward)
    assert all(abs(pos) <= 1.0 for pos in info['positions']) 