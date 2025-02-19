#!/usr/bin/env python3
"""Hyperparameter optimization script for SAC trading agent using Optuna."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor
import logging
import psycopg2
from sqlalchemy.engine import create_engine
import time
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
from multiprocessing import Queue, Manager
import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity
import gc
from sqlalchemy import event
from sqlalchemy.exc import DBAPIError
from tenacity import retry, stop_after_attempt, wait_exponential
import warnings
from optuna.exceptions import ExperimentalWarning
import tempfile
import argparse

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.market_data import create_synthetic_data
from src.data.feature_engineering import prepare_market_features
from src.train.train import train_agent
from src.env.trading_env import TradingEnvironment
from src.models.sac_agent import SACAgent
from src.utils.risk_management import RiskParams

# Global progress bars
manager = Manager()
global_progress = None
active_trials = manager.dict()  # Shared dictionary for active trials
trial_progress_bars = manager.dict()  # Shared dictionary for trial progress bars

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_storage():
    """Get database storage with retry logic."""
    url = "postgresql://postgres:postgres@localhost:5432/optuna_db"
    storage = optuna.storages.RDBStorage(
        url=url,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )
    return storage

def create_study_name() -> str:
    """Create unique study name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sac_trading_optimization_{timestamp}"

def profile_trial(func, trial_dir, *args, **kwargs):
    """Profile a single trial and save results."""
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Save profiling stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats(str(trial_dir / 'profile_stats.txt'))
        
        return result
    finally:
        profiler.disable()

def save_trial_results(trial_dir, metrics, objective_value):
    """Save trial results and metrics."""
    results = {
        "objective_value": float(objective_value),
        "final_portfolio_value": float(metrics["portfolio_values"][-1]),
        "final_sharpe_ratio": float(metrics["sharpe_ratios"][-1]),
        "max_drawdown": float(max(metrics["max_drawdowns"])),
        "training_steps": len(metrics["portfolio_values"])
    }
    
    with open(trial_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

def generate_data_and_features():
    """Generate synthetic market data and calculate features."""
    print("Generating synthetic market data...")
    n_samples = 1000
    np.random.seed(42)
    
    # Generate more volatile price series
    returns = np.random.normal(0.0002, 0.02, n_samples)  # Increased volatility
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame with OHLCV data
    price_data = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_samples) * 0.002),
        'high': close_prices * (1 + np.abs(np.random.randn(n_samples) * 0.004)),
        'low': close_prices * (1 - np.abs(np.random.randn(n_samples) * 0.004)),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Print data statistics
    print("\nPrice Data Statistics:")
    print(f"Close Price Range: [{price_data['close'].min():.2f}, {price_data['close'].max():.2f}]")
    print(f"Daily Returns Mean: {price_data['close'].pct_change().mean():.4f}")
    print(f"Daily Returns Std: {price_data['close'].pct_change().std():.4f}")
    
    # Add datetime index
    price_data.index = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='D')
    
    print("Calculating technical features...")
    feature_config = {
        'window_sizes': [20, 50],
        'normalization': 'zscore',
        'normalization_lookback': 100,
        'features': {
            'price_based': ['sma', 'ema', 'bbands'],
            'momentum': ['rsi', 'macd'],
            'volatility': ['volatility']
        }
    }
    
    features = prepare_market_features(price_data, feature_config)
    
    # Ensure data alignment
    common_index = features.index.intersection(price_data.index)
    price_data = price_data.loc[common_index]
    features = features.loc[common_index]
    
    print(f"Generated data shapes - Price data: {price_data.shape}, Features: {features.shape}")
    return price_data, features

def objective(trial, price_data, features, device, gpu_queue):
    try:
        gpu_id = gpu_queue.get(timeout=1)
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    except:
        device = "cpu"
        gpu_id = None

    trial_dir = os.path.join("results", f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # Simple, direct config with hyperparameter ranges
    config = {
        "risk_params": {
            "max_position": trial.suggest_float("max_position", 0.3, 0.9),
            "max_leverage": trial.suggest_float("max_leverage", 0.6, 1.0),
            "position_step": trial.suggest_float("position_step", 0.05, 0.2),
            "vol_target": trial.suggest_float("vol_target", 0.1, 0.2)
        },
        "agent_params": {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_int("batch_size", 128, 1024, step=128),
            "gamma": trial.suggest_float("gamma", 0.95, 0.99),
            "tau": trial.suggest_float("tau", 0.001, 0.01),
            "alpha": trial.suggest_float("alpha", 0.1, 0.5)
        },
        "window_size": trial.suggest_int("window_size", 20, 100),
        "commission": 0.001,
        "num_episodes": 50,
        "start_training_after_steps": 1000,
        "save_interval": 10000,
        "eval_interval": 10,
        "eval_episodes": 5,
        "early_stopping_window": 10,
        "target_sharpe": 2.0
    }
    
    config_path = os.path.join(trial_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(trial_dir, "trial.log"),
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Train the agent
        results = train_agent(
            config_path=config_path,
            price_data=price_data,
            features=features,
            save_dir=trial_dir,
            device=device,
            disable_progress=False,
            log_level="INFO"
        )
        
        # Add debugging prints
        print(f"\nTrial {trial.number} Results:")
        print(f"Portfolio values: {results.get('portfolio_values', [])[-5:]}")  # Last 5 values
        print(f"Returns: {results.get('returns', [])[-5:]}")  # Last 5 values
        print(f"Sharpe ratios: {results.get('sharpe_ratios', [])[-5:]}")  # Last 5 values
        
        # Calculate objective value (mean of last 10 Sharpe ratios)
        if not results.get('sharpe_ratios'):
            print("Warning: No Sharpe ratios found in results")
            return 0.0
            
        objective_value = np.mean(results["sharpe_ratios"][-10:])
        
        # Save results
        with open(os.path.join(trial_dir, "results.json"), "w") as f:
            json.dump({
                "objective_value": float(objective_value),
                "sharpe_ratios": [float(x) for x in results["sharpe_ratios"]],
                "returns": [float(x) for x in results["returns"]],
                "config": config
            }, f, indent=4)
            
        return objective_value

    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        raise

    finally:
        # Clean up
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

def run_optimization(market_data=None, features=None, n_trials=100, jobs_per_gpu=12):
    """Run hyperparameter optimization."""
    # Suppress Optuna warnings
    warnings.filterwarnings('ignore', category=ExperimentalWarning)
    
    # Configure logging
    logging.basicConfig(
        filename="optimization.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Generate data if needed
    if market_data is None or features is None:
        market_data, features = generate_data_and_features()
    
    # Create study
    storage = get_storage()
    study_name = f"sac_trading_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )
    
    # Create GPU queue
    n_gpus = torch.cuda.device_count()
    gpu_queue = Queue()
    for i in range(n_gpus):
        for _ in range(jobs_per_gpu):
            gpu_queue.put(i)
    
    try:
        # Run optimization
        study.optimize(
            lambda trial: objective(
                trial, market_data, features, None, gpu_queue
            ),
            n_trials=n_trials,
            n_jobs=n_gpus * jobs_per_gpu if n_gpus > 0 else 1,
            gc_after_trial=True
        )
        
        print(f"\nOptimization completed!")
        print(f"Best trial value: {study.best_value:.3f}")
        print("\nBest trial parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        raise
    finally:
        # Clean up
        while not gpu_queue.empty():
            gpu_queue.get()
    
    return study

def main():
    """Main entry point for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for SAC trading agent")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--jobs-per-gpu", type=int, default=12, help="Number of jobs per GPU")
    args = parser.parse_args()
    
    # Run optimization
    study = run_optimization(
        market_data=None,
        features=None,
        n_trials=args.n_trials,
        jobs_per_gpu=args.jobs_per_gpu
    )
    
    # Print results
    print("\nOptimization Results:")
    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value: {best_trial.value:.3f}")
        print("\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError as e:
        print("No completed trials found.")
        print("\nTrials in progress:")
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                print(f"  Trial {trial.number}: Running")
            elif trial.state == optuna.trial.TrialState.COMPLETE:
                print(f"  Trial {trial.number}: Complete (value: {trial.value:.3f})")
            elif trial.state == optuna.trial.TrialState.FAIL:
                print(f"  Trial {trial.number}: Failed")
            elif trial.state == optuna.trial.TrialState.PRUNED:
                print(f"  Trial {trial.number}: Pruned")

if __name__ == "__main__":
    main() 