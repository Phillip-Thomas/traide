import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import (
    calculate_technical_features,
    normalize_features,
    prepare_market_features
)

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

def test_technical_feature_calculation(sample_price_data):
    """Test calculation of technical indicators."""
    features = calculate_technical_features(
        data=sample_price_data,
        window_sizes=[14, 30]
    )
    
    # Check feature existence
    expected_features = [
        'sma_14', 'ema_14', 'bbands_upper_14', 'bbands_middle_14',
        'bbands_lower_14', 'atr_14', 'rsi_14', 'cci_14', 'vwap_14',
        'volume_sma_14', 'sma_30', 'ema_30', 'bbands_upper_30',
        'bbands_middle_30', 'bbands_lower_30', 'atr_30', 'rsi_30',
        'cci_30', 'vwap_30', 'volume_sma_30', 'macd', 'macd_signal',
        'macd_hist', 'stoch_k', 'stoch_d', 'adx'
    ]
    
    for feature in expected_features:
        assert feature in features.columns
    
    # Check data quality
    assert not features.isnull().all().any()  # No completely null columns
    assert features.shape[0] == len(sample_price_data)
    
    # Check specific feature properties
    assert (features['rsi_14'] >= 0).all() and (features['rsi_14'] <= 100).all()
    assert (features['stoch_k'] >= 0).all() and (features['stoch_k'] <= 100).all()
    assert (features['stoch_d'] >= 0).all() and (features['stoch_d'] <= 100).all()

def test_feature_normalization(sample_price_data):
    """Test feature normalization methods."""
    # Calculate raw features
    raw_features = calculate_technical_features(sample_price_data)
    
    # Test Z-score normalization
    zscore_features = normalize_features(raw_features, method='zscore', lookback=50)
    
    # Test min-max normalization
    minmax_features = normalize_features(raw_features, method='minmax', lookback=50)
    
    # Check normalization properties
    for column in zscore_features.columns:
        # Z-score properties (approximately)
        rolling_mean = zscore_features[column].rolling(50).mean()
        rolling_std = zscore_features[column].rolling(50).std()
        assert np.abs(rolling_mean.mean()) < 0.5  # Close to 0
        assert np.abs(rolling_std.mean() - 1) < 0.5  # Close to 1
    
    for column in minmax_features.columns:
        # Min-max properties
        assert minmax_features[column].min() >= -1e-6  # Approximately >= 0
        assert minmax_features[column].max() <= 1 + 1e-6  # Approximately <= 1

def test_complete_feature_pipeline(sample_price_data):
    """Test the complete feature engineering pipeline."""
    config = {
        'price_col': 'close',
        'volume_col': 'volume',
        'window_sizes': [14, 30, 50],
        'normalization': 'zscore',
        'normalization_lookback': 252
    }
    
    # Process features
    features = prepare_market_features(sample_price_data, config)
    
    # Check basic properties
    assert isinstance(features, pd.DataFrame)
    assert not features.isnull().any().any()  # No null values
    assert len(features) > 0
    
    # Check feature ranges after normalization
    for column in features.columns:
        values = features[column].values
        assert np.abs(values.mean()) < 5  # Reasonable mean
        assert np.abs(values.std()) < 5   # Reasonable std
        
def test_feature_engineering_edge_cases(sample_price_data):
    """Test feature engineering with edge cases."""
    # Test with very small window
    small_window_features = calculate_technical_features(
        sample_price_data,
        window_sizes=[2]
    )
    assert not small_window_features.isnull().all().any()
    
    # Test with window larger than data
    large_window_features = calculate_technical_features(
        sample_price_data,
        window_sizes=[len(sample_price_data) + 100]
    )
    assert large_window_features.isnull().any().any()  # Should have some null values
    
    # Test normalization with extreme values
    extreme_data = sample_price_data.copy()
    extreme_data['close'] = extreme_data['close'] * 1000000
    
    normalized = normalize_features(
        calculate_technical_features(extreme_data),
        method='zscore'
    )
    assert not normalized.isnull().any().any()
    assert normalized.std().mean() < 10  # Should still be reasonably scaled

def test_feature_stability(sample_price_data):
    """Test stability of feature calculations."""
    # Calculate features twice
    features1 = calculate_technical_features(sample_price_data)
    features2 = calculate_technical_features(sample_price_data)
    
    # Should be exactly the same
    assert (features1 == features2).all().all()
    
    # Test with different random seeds
    np.random.seed(42)
    normalized1 = normalize_features(features1, method='zscore')
    np.random.seed(100)
    normalized2 = normalize_features(features1, method='zscore')
    
    # Should be exactly the same (normalization shouldn't depend on random seed)
    assert (normalized1 == normalized2).all().all() 