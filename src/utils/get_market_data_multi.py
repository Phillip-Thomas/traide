import pickle
import os
from datetime import datetime, timedelta
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_market_data_multi(tickers=None, period="730d", interval="1h", cache_dir="cache"):
    """
    Fetch data for multiple tickers with caching
    """
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # Create cache filename based on parameters
    cache_params = f"{period}_{interval}"
    cache_file = os.path.join(cache_dir, f"market_data_{cache_params}.pkl")
    cache_meta_file = os.path.join(cache_dir, f"market_data_{cache_params}_meta.pkl")
    
    # Check if cache exists and is recent (less than 4 hours old)
    use_cache = False
    if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
        try:
            with open(cache_meta_file, 'rb') as f:
                cache_meta = pickle.load(f)
            cache_age = datetime.now() - cache_meta['timestamp']
            
            # Use cache if it's less than 4 hours old
            if cache_age < timedelta(hours=4):
                print(f"Using cached data from {cache_meta['timestamp']} ({cache_age.total_seconds()/3600:.1f} hours old)")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"Cache is {cache_age.total_seconds()/3600:.1f} hours old. Refreshing...")
        except Exception as e:
            print(f"Error reading cache: {str(e)}")
    
    # If we get here, we need to fetch new data
    if tickers is None:
        etfs = [
            "QQQ",   "XLK",   "VGT",   "SMH",   "SOXX",  
            "IGV",   "FTEC",  "IYW",   "QTEC",  "AIQ",   
            "BOTZ",  "ROBO",  "SOXQ",  "PSI",   "XSD",   
            "USD",   "WCLD",  "SKYY",  "CLOU",  "CIBR",  
            "BUG",   "ARKW",  "KOMP",  "PTF",   "XITK",  
            "IWF",   "VUG",   "SPYG",  "VONG",  "SCHG"
        ]
        
        stocks = [
            # Big Tech / FAANG+
            "AAPL",  "MSFT",  "GOOGL", "AMZN",  "META",  "NVDA",  "TSLA",
            
            # Semiconductors
            "AMD",   "INTC",  "TSM",   "AVGO",  "QCOM",  "AMAT",  "MU",
            
            # Software & Cloud
            "CRM",   "ADBE",  "ORCL",  "VMW",   "SNOW",  "NET",   "PLTR",
            
            # Hardware & Infrastructure
            "CSCO",  "IBM",   "HPQ",   "DELL",  "STX",   "WDC",
            
            # Internet & Digital Services
            "PYPL",  "SQ",    "SHOP",  "ABNB",  "UBER",  "DASH",
            
            # Media & Gaming
            "NFLX",  "DIS",   "ATVI",  "EA",    "TTWO",  "RBLX",
            
            # Emerging Tech
            "COIN",  "U",     "PATH",  "AI",    "CRWD",  "ZS",
        ]
        
        tickers = etfs + stocks
    
    data_dict = {}
    failed_tickers = []
    
    print(f"\nFetching data for {len(tickers)} tickers:")
    for symbol in tickers:
        try:
            print(f"Fetching {symbol}...", end=" ")
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if not df.empty and len(df) > 100:
                print(f"Success ({len(df)} records)")
                data_dict[symbol] = df
            else:
                print("Insufficient data")
                failed_tickers.append((symbol, "Insufficient data"))
        except Exception as e:
            print(f"Error: {str(e)}")
            failed_tickers.append((symbol, str(e)))
    
    if failed_tickers:
        print("\nFailed to fetch data for the following tickers:")
        for ticker, reason in failed_tickers:
            print(f"  - {ticker}: {reason}")
    
    print(f"\nSuccessfully loaded {len(data_dict)} out of {len(tickers)} tickers")
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data_dict, f)
        with open(cache_meta_file, 'wb') as f:
            pickle.dump({
                'timestamp': datetime.now(),
                'period': period,
                'interval': interval,
                'num_tickers': len(data_dict)
            }, f)
        print(f"Saved data to cache: {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {str(e)}")
    
    return data_dict