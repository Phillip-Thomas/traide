import pytest
import numpy as np
from environments.trading_env import SimpleTradeEnv, get_state_size

def test_get_state_size():
    """Test the state size calculation function."""
    window_size = 48
    expected_size = window_size * 9 + 3  # 9 features per timestep + 3 additional features
    assert get_state_size(window_size) == expected_size

def test_env_initialization(sample_market_data):
    """Test environment initialization with sample data."""
    window_size = 48
    env = SimpleTradeEnv(sample_market_data, window_size=window_size)
    
    # Check initial values
    assert env.window_size == window_size
    assert env.position == 0
    assert env.cash == 1.0
    assert env.shares == 0
    assert env.entry_price is None
    assert env.entry_idx is None

def test_env_reset(sample_market_data):
    """Test environment reset functionality."""
    env = SimpleTradeEnv(sample_market_data, window_size=48)
    
    # Make some actions to change state
    state, reward, done = env.step(1)  # Buy
    state, reward, done = env.step(2)  # Sell
    
    # Reset and check values
    state = env.reset()
    assert env.position == 0
    assert env.cash == 1.0
    assert env.shares == 0
    assert env.entry_price is None
    assert env.entry_idx is None
    assert env.max_drawdown == 0
    assert env.max_profit == 0

def test_valid_actions(sample_market_data):
    """Test that valid actions are correctly returned based on position."""
    env = SimpleTradeEnv(sample_market_data)
    
    # Initial position (no position)
    assert env.get_valid_actions() == [0, 1]  # HOLD, BUY
    
    # After buying
    env.step(1)
    assert env.get_valid_actions() == [0, 2]  # HOLD, SELL
    
    # After selling
    env.step(2)
    assert env.get_valid_actions() == [0, 1]  # HOLD, BUY

def test_buy_action(sample_market_data):
    """Test the buy action mechanics."""
    env = SimpleTradeEnv(sample_market_data)
    initial_price = env.close_prices[env.idx]
    
    # Execute buy
    state, reward, done = env.step(1)
    
    assert env.position == 1
    assert env.cash == 0
    assert env.shares > 0
    assert env.entry_price == initial_price
    assert env.entry_idx == env.window_size
    # Check that shares * price ≈ 1 (initial cash, minus fees)
    assert abs(env.shares * initial_price - 1.0) < 0.01

def test_sell_action(sample_market_data):
    """Test the sell action mechanics."""
    env = SimpleTradeEnv(sample_market_data)
    
    # Buy first
    state, reward, done = env.step(1)
    initial_shares = env.shares
    
    # Then sell
    state, reward, done = env.step(2)
    
    assert env.position == 0
    assert env.shares == 0
    assert env.entry_price is None
    assert env.entry_idx is None
    assert env.cash > 0  # Should have some cash after selling
    # Check that we're not losing money just from buying and selling immediately
    # (beyond transaction costs)
    assert env.cash > 0.99  # Allow for small transaction costs

def test_hold_action(sample_market_data):
    """Test the hold action mechanics."""
    env = SimpleTradeEnv(sample_market_data)
    initial_state = env._get_state()
    
    # Execute hold
    state, reward, done = env.step(0)
    
    assert env.position == 0
    assert env.cash == 1.0
    assert env.shares == 0
    assert env.entry_price is None
    assert env.entry_idx is None
    
    # Position shouldn't change
    next_state = env._get_state()
    assert len(next_state) == len(initial_state)

def test_invalid_actions(sample_market_data):
    """Test that invalid actions are properly handled."""
    env = SimpleTradeEnv(sample_market_data)
    
    # Try to sell without position
    state, reward, done = env.step(2)
    assert reward < 0  # Should get negative reward for invalid action
    assert env.position == 0
    assert env.cash == 1.0
    
    # Buy to get into position
    state, reward, done = env.step(1)
    
    # Try to buy while already in position
    state, reward, done = env.step(1)
    assert reward < 0  # Should get negative reward for invalid action
    assert env.position == 1
    assert env.cash == 0 