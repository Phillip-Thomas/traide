import os
import yfinance as yf
import pickle
import hashlib
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import pandas as pd

def get_yield_etf_configs():
    """
    Configuration for yield ETFs and their underlying assets.
    Format: {
        'ETF_TICKER': {
            'underlying': 'UNDERLYING_TICKER',
            'dividend_day': 0-6 (0=Monday),
            'dividend_hour': 0-23,
            'dividend_minute': 0-59
        }
    }
    """
    return {
        'YBTC': {
            'underlying': 'IBIT',
            'dividend_day': 2,    # Wednesday
            'dividend_hour': 15,   # 3 PM
            'dividend_minute': 59
        },
        'YETH': {
            'underlying': 'EETH',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'XDTE': {
            'underlying': 'SPY',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'QDTE': {
            'underlying': 'QQQ',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'RDTE': {
            'underlying': 'IWM',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'MSTY': {
            'underlying': 'MSTR',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'APLY': {
            'underlying': 'AAPL',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'TSLY': {
            'underlying': 'TSLA',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        },
        'NVDY': {
            'underlying': 'NVDA',
            'dividend_day': 2,
            'dividend_hour': 15,
            'dividend_minute': 59
        }
    }

def generate_dividend_calendar(start_date, end_date, config):
    """Generate dividend dates based on configuration"""
    dates = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq=f'W-{["MON","TUE","WED","THU","FRI","SAT","SUN"][config["dividend_day"]]}'
    )
    return dates.map(lambda x: x.replace(
        hour=config['dividend_hour'],
        minute=config['dividend_minute']
    ))

def fetch_data_in_chunks(symbol, total_period='60d', interval='15m'):
    """
    Fetch data with different granularity based on age:
    - Last 30 days: 15-minute data
    - Beyond 30 days: 1-hour data, resampled to 15-minute
    """
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(total_period)
    all_data = []
    
    # Split into recent and historical periods
    thirty_days_ago = end_date - pd.Timedelta(days=30)
    
    # Fetch recent data (15-minute granularity)
    if end_date > thirty_days_ago:
        try:
            recent_data = yf.Ticker(symbol).history(
                start=max(start_date, thirty_days_ago).strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='15m'
            )
            if not recent_data.empty:
                all_data.append(recent_data)
                print(f"Fetched {len(recent_data)} 15-minute records for {symbol} from {max(start_date, thirty_days_ago).date()} to {end_date.date()}")
        except Exception as e:
            print(f"Error fetching 15m data for {symbol}: {str(e)}")

    # Fetch historical data (1-hour granularity)
    if start_date < thirty_days_ago:
        try:
            historical_data = yf.Ticker(symbol).history(
                start=start_date.strftime('%Y-%m-%d'),
                end=thirty_days_ago.strftime('%Y-%m-%d'),
                interval='1h'
            )
            if not historical_data.empty:
                # Resample to 15-minute using linear interpolation for prices and forward fill for volume
                historical_resampled = pd.DataFrame(index=pd.date_range(
                    start=historical_data.index[0],
                    end=historical_data.index[-1],
                    freq='15min'
                ))
                
                # Interpolate OHLC prices
                for col in ['Open', 'High', 'Low', 'Close']:
                    historical_resampled[col] = historical_data[col].reindex(historical_resampled.index).interpolate(method='linear')
                
                # Forward fill volume
                historical_resampled['Volume'] = historical_data['Volume'].reindex(historical_resampled.index).ffill()
                
                all_data.append(historical_resampled)
                print(f"Fetched and resampled {len(historical_resampled)} records for {symbol} from {start_date.date()} to {thirty_days_ago.date()}")
        except Exception as e:
            print(f"Error fetching 1h data for {symbol}: {str(e)}")

    if not all_data:
        return None

    # Concatenate all data and sort by index
    combined_data = pd.concat(all_data)
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
    combined_data.sort_index(inplace=True)
    
    return combined_data

def get_market_data_multi(tickers=None, period="730d", interval="15m", use_cache=True):
    """
    Fetch data for multiple tickers with local caching
    
    Args:
        tickers: List of tickers to fetch, or None for default list
        period: Time period to fetch (e.g. "60d", "30d")
        interval: Data granularity (e.g. "1m", "15m", "1h")
        use_cache: Whether to use cached data
    """
    # If specifically requesting yield ETFs, use specialized function
    configs = get_yield_etf_configs()
    if tickers and all(ticker in configs for ticker in tickers):
        return prepare_yield_etf_data(tickers, period, interval, use_cache)

    # Create cache directory if it doesn't exist
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)

    # Create cache key from parameters
    param_str = f"{sorted(tickers) if tickers else 'default'}_{period}_{interval}"
    cache_key = hashlib.md5(param_str.encode()).hexdigest()
    cache_file = cache_dir / f"market_data_{cache_key}.pkl"

    # Try to load from cache first
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print("Loaded market data from cache")
            return cached_data
        except Exception as e:
            print(f"Error loading cache: {e}")

    # If not in cache or cache disabled, fetch from API
    if tickers is None:
        tickers = [
            # Core Tech/Growth Leaders (Strong consistent trends)
            "QQQ",   # Nasdaq 100
            "XLK",   # Technology Select SPDR
            "VGT",   # Vanguard Info Tech
            "SMH",   # Semiconductor ETF
            "SOXX",  # iShares Semiconductor
            "IGV",   # iShares Software ETF
            "FTEC",  # Fidelity MSCI Info Tech
            "IYW",   # iShares US Technology
            "QTEC",  # First Trust NASDAQ-100 Tech

            # AI/Future Tech (Strong momentum)
            "AIQ",   # Global X Artificial Intelligence
            "BOTZ",  # Global X Robotics & AI
            "ROBO",  # Robotics & Automation

            # Semiconductor Focus (Very strong trend)
            "SOXQ",  # First Trust NASDAQ Semi
            "PSI",   # Invesco Dynamic Semi
            "XSD",   # SPDR S&P Semi
            "USD",   # ProShares Ultra Semi

            # Cloud/Cybersecurity (Growing sectors)
            "WCLD",  # WisdomTree Cloud Computing
            "SKYY",  # First Trust Cloud Computing
            "CLOU",  # Global X Cloud Computing
            "CIBR",  # First Trust Cybersecurity
            "BUG",   # Global X Cybersecurity

            # Next-Gen Tech (Strong growth)
            "ARKW",  # ARK Next Gen Internet
            "KOMP",  # ProShares Genomics

            # Broad Tech/Growth (Stable uptrends)
            "IWF",   # iShares Russell 1000 Growth
            "VUG",   # Vanguard Growth ETF
            "SPYG",  # SPDR Portfolio S&P 500 Growth
            "VONG",  # Vanguard Russell 1000 Growth
            "SCHG"   # Schwab US Large-Cap Growth
        ]

    data_dict = {}
    for symbol in tickers:
        try:
            if interval == '1m':
                # Use chunked fetching for 1-minute data
                df = fetch_data_in_chunks(symbol, total_period=period, interval=interval)
            else:
                # Use regular fetching for other intervals
                df = yf.Ticker(symbol).history(period=period, interval=interval)

            if df is not None and not df.empty and len(df) > 100:  # Ensure sufficient data
                print(f"Successfully fetched total {len(df)} records for {symbol}")
                data_dict[symbol] = df
            else:
                print(f"Insufficient data for {symbol}")
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")

    # Save to cache if we got data
    if data_dict and use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data_dict, f)
            print("Saved market data to cache")
        except Exception as e:
            print(f"Error saving to cache: {e}")

    return data_dict

def clear_market_data_cache():
    """Clear the market data cache"""
    cache_dir = Path("data_cache")
    if cache_dir.exists():
        for cache_file in cache_dir.glob("market_data_*.pkl"):
            cache_file.unlink()
        print("Market data cache cleared")

def get_cache_info():
    """Get information about cached market data"""
    cache_dir = Path("data_cache")
    if not cache_dir.exists():
        return "No cache directory found"

    cache_files = list(cache_dir.glob("market_data_*.pkl"))
    if not cache_files:
        return "No cached data found"

    info = []
    for cache_file in cache_files:
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
        info.append(f"{cache_file.name}: {size_mb:.2f}MB, Last modified: {modified}")

    return "\n".join(info)

def save_stats(stats, filename):
    """Save training statistics to file"""
    # Convert numpy arrays to lists for JSON serialization
    stats_copy = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            stats_copy[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            stats_copy[key] = [v.tolist() for v in value]
        else:
            stats_copy[key] = value

    with open(os.path.join("stats", filename), 'w') as f:
        json.dump(stats_copy, f)

def prepare_yield_etf_data(tickers=None, period="60d", interval="15m", use_cache=True):
    """
    Prepare data for yield ETFs with their underlying correlations
    
    Args:
        tickers: List of yield ETF tickers to prepare, or None for all
        period: Time period to fetch (e.g. "60d", "30d")
        interval: Data granularity (e.g. "1m", "15m")
        use_cache: Whether to use cached data
    """
    configs = get_yield_etf_configs()
    
    # Filter configs if specific tickers requested
    if tickers:
        configs = {k: v for k, v in configs.items() if k in tickers}
    
    # Get all required tickers (both ETFs and underlying)
    all_tickers = list(configs.keys()) + [cfg['underlying'] for cfg in configs.values()]
    
    print(f"\nFetching {period} of {interval} data for tickers:", all_tickers)
    
    # Fetch all data at once
    data_dict = get_market_data_multi(
        tickers=all_tickers,
        period=period,
        interval=interval,
        use_cache=use_cache
    )
    
    if not data_dict:
        return None
    
    # Print data coverage summary
    print("\nData Coverage Summary:")
    for ticker in all_tickers:
        if ticker in data_dict:
            data = data_dict[ticker]
            print(f"{ticker}: {len(data)} records from {data.index[0]} to {data.index[-1]}")
        else:
            print(f"{ticker}: No data available")
    
    processed_data = {}
    
    for etf_ticker, config in configs.items():
        if etf_ticker not in data_dict:
            print(f"Warning: No data for {etf_ticker}")
            continue
            
        underlying_ticker = config['underlying']
        if underlying_ticker not in data_dict:
            print(f"Warning: No data for {underlying_ticker} (underlying of {etf_ticker})")
            continue
        
        etf_data = data_dict[etf_ticker].copy()
        underlying_data = data_dict[underlying_ticker]
        
        # Generate dividend calendar
        start_date = etf_data.index[0]
        end_date = etf_data.index[-1]
        dividend_dates = generate_dividend_calendar(start_date, end_date, config)
        
        # Initialize dividend-related columns with correct dtypes
        etf_data['next_dividend'] = pd.Series(dtype='datetime64[ns]', index=etf_data.index)
        etf_data['days_to_dividend'] = pd.Series(0, dtype='int64', index=etf_data.index)
        etf_data['hours_to_dividend'] = pd.Series(0.0, dtype='float64', index=etf_data.index)
        
        # For each timestamp, find the next dividend date
        for idx, timestamp in enumerate(etf_data.index):
            next_div = dividend_dates[dividend_dates > timestamp].min()
            if pd.notna(next_div):
                etf_data.at[timestamp, 'next_dividend'] = pd.Timestamp(next_div)
                time_to_div = next_div - timestamp
                etf_data.at[timestamp, 'days_to_dividend'] = time_to_div.days
                etf_data.at[timestamp, 'hours_to_dividend'] = time_to_div.total_seconds() / 3600
        
        # Add underlying correlation if available
        # Resample both to 1-minute if needed
        etf_1min = etf_data.asfreq('1min')
        underlying_1min = underlying_data.asfreq('1min')
        
        # Calculate correlation features
        etf_data['underlying_ratio'] = etf_data['Close'] / underlying_1min['Close']
        etf_data['underlying_spread'] = etf_data['Close'] - underlying_1min['Close']
        
        # Calculate rolling correlation with explicit fill method
        etf_returns = etf_data['Close'].pct_change(fill_method=None).fillna(0)
        underlying_returns = underlying_1min['Close'].pct_change(fill_method=None).fillna(0)
        etf_data['underlying_correlation'] = (
            pd.DataFrame({
                'etf': etf_returns,
                'underlying': underlying_returns
            })
            .rolling(window=60)  # 1-hour rolling correlation
            .corr()
            .unstack()
            .iloc[:, 1]
        )
        
        processed_data[etf_ticker] = etf_data
    
    return processed_data

def prepare_ybtc_data(period="60d", interval="15m", use_cache=True):
    """
    Prepare YBTC data with dividend dates and IBIT correlation
    
    Args:
        period: Time period to fetch (e.g. "60d", "30d")
        interval: Data granularity (e.g. "1m", "15m")
        use_cache: Whether to use cached data
    """
    # Get YBTC and IBIT data
    data_dict = get_market_data_multi(
        tickers=['YBTC', 'IBIT'],
        period=period,
        interval=interval,
        use_cache=use_cache
    )
    
    if not data_dict or 'YBTC' not in data_dict:
        return None
        
    ybtc_data = data_dict['YBTC']
    ibit_data = data_dict.get('IBIT')
    
    # Generate dividend calendar
    start_date = ybtc_data.index[0]
    end_date = ybtc_data.index[-1]
    dividend_dates = generate_dividend_calendar(start_date, end_date, get_yield_etf_configs()['YBTC'])
    
    # Add dividend-related columns
    ybtc_data['next_dividend'] = pd.NaT
    ybtc_data['days_to_dividend'] = 0
    ybtc_data['hours_to_dividend'] = 0
    
    # For each timestamp, find the next dividend date
    for idx, timestamp in enumerate(ybtc_data.index):
        next_div = dividend_dates[dividend_dates > timestamp].min()
        if pd.notna(next_div):
            ybtc_data.loc[timestamp, 'next_dividend'] = next_div
            time_to_div = next_div - timestamp
            ybtc_data.loc[timestamp, 'days_to_dividend'] = time_to_div.days
            ybtc_data.loc[timestamp, 'hours_to_dividend'] = time_to_div.total_seconds() / 3600
    
    # Add IBIT correlation if available
    if ibit_data is not None:
        # Resample both to 1-minute if needed
        ybtc_1min = ybtc_data.asfreq('1min')
        ibit_1min = ibit_data.asfreq('1min')
        
        # Calculate correlation features
        ybtc_data['ibit_ratio'] = ybtc_data['Close'] / ibit_1min['Close']
        ybtc_data['ibit_spread'] = ybtc_data['Close'] - ibit_1min['Close']
        
        # Calculate rolling correlation
        ybtc_returns = ybtc_data['Close'].pct_change()
        ibit_returns = ibit_1min['Close'].pct_change()
        ybtc_data['ibit_correlation'] = (
            pd.DataFrame({'ybtc': ybtc_returns, 'ibit': ibit_returns})
            .rolling(window=60)  # 1-hour rolling correlation
            .corr()
            .unstack()
            .iloc[:, 1]
        )
    
    return ybtc_data 