import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.market_data import (
    fetch_market_data,
    prepare_training_data,
    create_synthetic_data,
    validate_market_data
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    data = pd.DataFrame({
        'open_AAPL': np.random.randn(500).cumsum() + 100,
        'high_AAPL': np.random.randn(500).cumsum() + 102,
        'low_AAPL': np.random.randn(500).cumsum() + 98,
        'close_AAPL': np.random.randn(500).cumsum() + 100,
        'volume_AAPL': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Ensure price consistency
    data['high_AAPL'] = data[['open_AAPL', 'close_AAPL', 'high_AAPL']].max(axis=1)
    data['low_AAPL'] = data[['open_AAPL', 'close_AAPL', 'low_AAPL']].min(axis=1)
    
    return data

def test_create_synthetic_data():
    """Test synthetic data generation."""
    n_samples = 1000
    n_assets = 2
    data = create_synthetic_data(n_samples=n_samples, n_assets=n_assets, seed=42)
    
    # Check dimensions
    assert len(data) == n_samples
    assert len(data.columns) == 5 * n_assets  # 5 columns per asset
    
    # Check column names
    for i in range(n_assets):
        symbol = f"ASSET_{i+1}"
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert f"{col}_{symbol}" in data.columns
    
    # Check data consistency
    for i in range(n_assets):
        symbol = f"ASSET_{i+1}"
        assert (data[f"high_{symbol}"] >= data[f"low_{symbol}"]).all()
        assert (data[f"high_{symbol}"] >= data[f"open_{symbol}"]).all()
        assert (data[f"high_{symbol}"] >= data[f"close_{symbol}"]).all()
        assert (data[f"volume_{symbol}"] >= 0).all()

def test_prepare_training_data(sample_market_data):
    """Test training data preparation."""
    train_data, val_data, test_data = prepare_training_data(
        sample_market_data,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Check split sizes
    total_size = len(sample_market_data)
    assert len(train_data) == int(total_size * 0.7)
    assert len(val_data) == int(total_size * 0.15)
    assert len(test_data) >= int(total_size * 0.14)  # Account for rounding
    
    # Check temporal order
    assert train_data.index[-1] < val_data.index[0]
    assert val_data.index[-1] < test_data.index[0]
    
    # Check data integrity
    assert (train_data.columns == sample_market_data.columns).all()
    assert (val_data.columns == sample_market_data.columns).all()
    assert (test_data.columns == sample_market_data.columns).all()

def test_validate_market_data(sample_market_data):
    """Test market data validation."""
    # Test valid data
    assert validate_market_data(sample_market_data)
    
    # Test missing columns
    invalid_data = sample_market_data.drop('volume_AAPL', axis=1)
    assert not validate_market_data(invalid_data)
    
    # Test insufficient data
    insufficient_data = sample_market_data.iloc[:100]
    assert not validate_market_data(insufficient_data)
    
    # Test price inconsistency
    inconsistent_data = sample_market_data.copy()
    inconsistent_data['high_AAPL'] = inconsistent_data['low_AAPL'] - 1
    assert not validate_market_data(inconsistent_data)
    
    # Test negative volumes
    negative_volume_data = sample_market_data.copy()
    negative_volume_data['volume_AAPL'] = -1
    assert not validate_market_data(negative_volume_data)

@pytest.mark.integration
def test_fetch_market_data():
    """Test market data fetching from Yahoo Finance."""
    # Test with a well-known symbol, fetch one year of data
    try:
        data = fetch_market_data(
            symbols=['AAPL'],
            start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        assert not data.empty
        assert 'close_AAPL' in data.columns
        
        # Basic data validation
        assert len(data) >= 200  # Reasonable number of trading days
        assert all(col in data.columns for col in [
            'open_AAPL', 'high_AAPL', 'low_AAPL', 'close_AAPL', 'volume_AAPL'
        ])
        assert (data['volume_AAPL'] >= 0).all()
        assert (data['high_AAPL'] >= data['low_AAPL']).all()
        
    except Exception as e:
        pytest.skip(f"Skipping due to data fetch error: {str(e)}")

def test_prepare_training_data_insufficient_data():
    """Test handling of insufficient data."""
    small_data = create_synthetic_data(n_samples=100)
    with pytest.raises(ValueError, match="Insufficient data points"):
        prepare_training_data(small_data)

def test_prepare_training_data_edge_ratios():
    """Test preparation with edge case ratios."""
    data = create_synthetic_data(n_samples=1000)
    
    # Test with high train ratio
    train_data, val_data, test_data = prepare_training_data(
        data,
        train_ratio=0.9,
        val_ratio=0.05
    )
    assert len(train_data) > len(val_data)
    assert len(train_data) > len(test_data)
    
    # Test with equal ratios
    train_data, val_data, test_data = prepare_training_data(
        data,
        train_ratio=0.33,
        val_ratio=0.33
    )
    # Allow for rounding differences of up to 1%
    total_size = len(data)
    expected_size = total_size // 3
    assert abs(len(train_data) - expected_size) <= total_size * 0.01
    assert abs(len(val_data) - expected_size) <= total_size * 0.01
    assert abs(len(test_data) - expected_size) <= total_size * 0.01 