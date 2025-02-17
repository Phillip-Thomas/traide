import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

class MockTrainingEnv:
    """Mock environment for testing training loop without real data dependencies."""
    def __init__(self, input_size=435):
        self._reset_state()
        self.input_size = input_size
        
        # Create mock market data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
        self.data = pd.DataFrame({
            'Open': np.linspace(100, 200, 1000),
            'High': np.linspace(102, 205, 1000),
            'Low': np.linspace(98, 195, 1000),
            'Close': np.linspace(101, 201, 1000),
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        self.close_prices = self.data['Close'].values
        self.window_size = 48

    def _reset_state(self):
        """Reset all internal state variables."""
        self.position = 0
        self.cash = 1.0
        self.reset_called = False
        self.step_called = False
        self.idx = 0

    def reset(self):
        """Reset the environment state."""
        self._reset_state()
        self.reset_called = True
        self.idx = self.window_size
        return np.zeros(self.input_size)

    def step(self, action):
        """Take a step in the environment."""
        self.step_called = True
        self.idx += 1
        reward = 0.1 if action == 1 else -0.1
        done = self.idx >= len(self.close_prices) - 1
        return np.zeros(self.input_size), reward, done

    def get_valid_actions(self):
        """Return valid actions based on current position."""
        return [0, 1] if self.position == 0 else [0, 2]

@pytest.fixture
def mock_training_env():
    """Create a mock training environment."""
    env = MockTrainingEnv()
    yield env
    # Reset the environment after each test
    env._reset_state()

@pytest.fixture
def mock_training_data():
    """Create mock training and validation data dictionaries."""
    env = MockTrainingEnv()
    train_data = {'MOCK': env.data}
    val_data = {'MOCK': env.data.copy()}
    return train_data, val_data

@pytest.fixture
def sample_market_data():
    """Generate a small, deterministic market dataset for testing."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='h')
    data = pd.DataFrame({
        'Open': np.linspace(100, 200, 200),
        'High': np.linspace(102, 205, 200),
        'Low': np.linspace(98, 195, 200),
        'Close': np.linspace(101, 201, 200),
        'Volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    return data

@pytest.fixture
def window_size():
    """Standard window size for testing."""
    return 48

@pytest.fixture
def device():
    """Fixture for torch device, preferring CPU for testing."""
    return torch.device('cpu')

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def mock_market_data_dict(sample_market_data):
    """Create a dictionary of market data for multiple symbols."""
    return {
        'AAPL': sample_market_data.copy(),
        'GOOGL': sample_market_data.copy() * 1.1  # Slightly different data
    }

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.append(str(src_dir)) 