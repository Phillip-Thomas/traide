import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from utils.get_market_data_multi import get_market_data_multi

@pytest.fixture
def mock_yf_ticker():
    """Create a mock yfinance Ticker object."""
    mock = MagicMock()
    
    def create_mock_history(period="2y", interval="1h"):
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
        data = pd.DataFrame({
            'Open': np.random.randn(1000) * 10 + 100,
            'High': np.random.randn(1000) * 10 + 102,
            'Low': np.random.randn(1000) * 10 + 98,
            'Close': np.random.randn(1000) * 10 + 101,
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        return data
    
    mock.history.side_effect = create_mock_history
    return mock

@pytest.fixture
def mock_failed_ticker():
    """Create a mock yfinance Ticker that fails."""
    mock = MagicMock()
    mock.history.side_effect = Exception("API Error")
    return mock

@pytest.fixture
def empty_ticker():
    """Create a mock yfinance Ticker that returns empty data."""
    mock = MagicMock()
    mock.history.return_value = pd.DataFrame()
    return mock

def test_market_data_fetch_success(mock_yf_ticker, test_data_dir):
    """Test successful market data fetching."""
    with patch('yfinance.Ticker', return_value=mock_yf_ticker):
        data = get_market_data_multi(
            tickers=['AAPL', 'GOOGL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        
        assert len(data) == 2
        assert 'AAPL' in data
        assert 'GOOGL' in data
        assert isinstance(data['AAPL'], pd.DataFrame)
        assert len(data['AAPL']) > 100
        assert all(col in data['AAPL'].columns 
                  for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_market_data_fetch_failure(mock_failed_ticker, test_data_dir):
    """Test handling of failed data fetching."""
    with patch('yfinance.Ticker', return_value=mock_failed_ticker):
        data = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        
        assert len(data) == 0

def test_market_data_empty_response(empty_ticker, test_data_dir):
    """Test handling of empty data response."""
    with patch('yfinance.Ticker', return_value=empty_ticker):
        data = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        
        assert len(data) == 0

def test_market_data_caching(mock_yf_ticker, test_data_dir):
    """Test that data is properly cached and reused."""
    cache_file = os.path.join(test_data_dir, "market_data_1y_1h.pkl")
    cache_meta_file = os.path.join(test_data_dir, "market_data_1y_1h_meta.pkl")
    
    # First call should fetch and cache
    with patch('yfinance.Ticker', return_value=mock_yf_ticker) as mock_ticker:
        data1 = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        assert mock_ticker.call_count == 1
        assert os.path.exists(cache_file)
        assert os.path.exists(cache_meta_file)
    
    # Second call within 4 hours should use cache
    with patch('yfinance.Ticker', return_value=mock_yf_ticker) as mock_ticker:
        data2 = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        assert mock_ticker.call_count == 0
        assert data1['AAPL'].equals(data2['AAPL'])

def test_market_data_cache_expiry(mock_yf_ticker, test_data_dir):
    """Test that expired cache is refreshed."""
    cache_file = os.path.join(test_data_dir, "market_data_1y_1h.pkl")
    cache_meta_file = os.path.join(test_data_dir, "market_data_1y_1h_meta.pkl")
    
    # First call to create cache
    with patch('yfinance.Ticker', return_value=mock_yf_ticker):
        data1 = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
    
    # Modify cache metadata to make it old
    import pickle
    with open(cache_meta_file, 'rb') as f:
        meta = pickle.load(f)
    meta['timestamp'] = datetime.now() - timedelta(hours=5)
    with open(cache_meta_file, 'wb') as f:
        pickle.dump(meta, f)
    
    # Second call should refresh cache
    with patch('yfinance.Ticker', return_value=mock_yf_ticker) as mock_ticker:
        data2 = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        assert mock_ticker.call_count == 1

def test_market_data_default_tickers(mock_yf_ticker, test_data_dir):
    """Test that default tickers list is used when none provided."""
    with patch('yfinance.Ticker', return_value=mock_yf_ticker) as mock_ticker:
        data = get_market_data_multi(
            tickers=None,
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        # Default list includes both ETFs and stocks
        assert mock_ticker.call_count > 30

def test_market_data_cache_corruption(mock_yf_ticker, test_data_dir):
    """Test handling of corrupted cache files."""
    cache_file = os.path.join(test_data_dir, "market_data_1y_1h.pkl")
    cache_meta_file = os.path.join(test_data_dir, "market_data_1y_1h_meta.pkl")
    
    # Create corrupted cache files
    with open(cache_file, 'w') as f:
        f.write("corrupted data")
    with open(cache_meta_file, 'w') as f:
        f.write("corrupted metadata")
    
    # Should handle corruption gracefully and fetch new data
    with patch('yfinance.Ticker', return_value=mock_yf_ticker) as mock_ticker:
        data = get_market_data_multi(
            tickers=['AAPL'],
            period='1y',
            interval='1h',
            cache_dir=str(test_data_dir)
        )
        assert mock_ticker.call_count == 1
        assert len(data) == 1
        assert 'AAPL' in data 