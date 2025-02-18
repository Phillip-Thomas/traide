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
from multiprocessing import Queue
import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.market_data import create_synthetic_data
from src.data.feature_engineering import prepare_market_features
from src.train.train import train_agent

# Global progress bar for trials
global_progress = None

def create_study_name():
    """Create unique study name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sac_trading_optimization_{timestamp}"

def profile_trial(func, trial_dir, *args, **kwargs):
    """Profile a single trial using cProfile."""
    profiler = cProfile.Profile()
    try:
        result = profiler.runcall(func, *args, **kwargs)
        # Save profiling stats
        stats_path = trial_dir / "profile_stats.prof"
        profiler.dump_stats(str(stats_path))
        
        # Create readable stats file
        stats = pstats.Stats(str(stats_path))
        with open(trial_dir / "profile_stats.txt", 'w') as f:
            stats.sort_stats('cumulative').stream = f
            stats.print_stats()
        return result
    finally:
        profiler.disable()

def objective(trial, base_config, price_data, features, n_episodes=50, jobs_per_gpu=12, gpu_queue=None, enable_profiling=False):
    """Optuna objective function for hyperparameter optimization."""
    global global_progress
    
    # Get GPU ID from queue if provided
    gpu_id = None
    if gpu_queue is not None:
        gpu_id = gpu_queue.get()
        
    try:
        # Clear CUDA cache at the start of each trial
        if gpu_id is not None:
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()
        
        # Create modified config for this trial
        config = base_config.copy()
        
        # Sample hyperparameters with importance-weighted ranges
        config["risk_params"].update({
            # High impact parameters (tighter ranges)
            "max_position": trial.suggest_float("max_position", 0.35, 0.40),  # Centered on 0.373
            
            # Medium impact parameters (moderate ranges)
            "max_leverage": trial.suggest_float("max_leverage", 0.33, 0.38),  # Centered on 0.349
            "vol_target": trial.suggest_float("vol_target", 0.11, 0.12),     # Centered on 0.117
            
            # Low impact parameters (wider ranges)
            "position_step": trial.suggest_float("position_step", 0.04, 0.06),  # Allow more exploration
        })
        
        config["agent_params"].update({
            # Highest impact parameters (very tight ranges)
            "batch_size": trial.suggest_int("batch_size", 145, 160),        # Centered on best performing range
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 8e-5, log=True),  # Tight around 6.58e-5
            
            # High impact parameter (tight range)
            "gamma": trial.suggest_float("gamma", 0.94, 0.97),              # Centered on 0.952
            
            # Medium impact parameters (moderate ranges)
            "alpha": trial.suggest_float("alpha", 0.21, 0.23),              # Centered on 0.220
            "tau": trial.suggest_float("tau", 0.002, 0.003),               # Centered on 0.0023
            
            # Low impact parameter (wider range)
            "hidden_dim": trial.suggest_int("hidden_dim", 90, 110),         # Centered on 98
            
            # Memory optimization
            "buffer_size": 100000,  # Reduced from 1M to 100K
        })

        # Training parameters tuned for better convergence
        config.update({
            "num_episodes": 30,                    # Increased from 20 for better convergence
            "start_training_after_steps": 2000,    # Increased for more initial exploration
            "save_interval": 5000,
            "eval_interval": 5,
            "eval_episodes": 3,                    # Increased for more stable evaluation
            "early_stopping_window": 10,
            "target_sharpe": 1.0,
            "window_size": 50,
            "commission": 0.001
        })

        # Create temporary directory for this trial
        trial_dir = Path(f"optimization_results/trial_{trial.number}")
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial config
        config_path = trial_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Set device based on GPU ID
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
        print(f"\nTrial {trial.number} running on device: {device}")
        
        if enable_profiling:
            # First run with cProfile
            profiler = cProfile.Profile()
            profiler.enable()
            
            metrics = train_agent(
                config_path=str(config_path),
                price_data=price_data,
                features=features,
                save_dir=trial_dir,
                device=device,
                disable_progress=True
            )
            
            profiler.disable()
            # Save cProfile results
            stats_path = trial_dir / "profile_stats.prof"
            profiler.dump_stats(str(stats_path))
            stats = pstats.Stats(str(stats_path))
            with open(trial_dir / "profile_stats.txt", 'w') as f:
                stats.sort_stats('cumulative').stream = f
                stats.print_stats()
            
            # Then run with PyTorch profiler for a shorter duration
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                profile_memory=True,
                record_shapes=True,
                with_modules=True,
                # Only profile for a few steps to avoid overhead
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                ),
            ) as prof:
                # Run a shorter training session for PyTorch profiling
                temp_config = config.copy()
                temp_config["num_episodes"] = 2  # Reduced episodes for profiling
                with open(trial_dir / "temp_config.yaml", "w") as f:
                    yaml.dump(temp_config, f)
                
                train_agent(
                    config_path=str(trial_dir / "temp_config.yaml"),
                    price_data=price_data,
                    features=features,
                    save_dir=trial_dir / "torch_profile",
                    device=device,
                    disable_progress=True
                )
            
            # Save PyTorch profiler results
            prof.export_chrome_trace(str(trial_dir / "pytorch_trace.json"))
            with open(trial_dir / "pytorch_profile.txt", "w") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
                f.write("\n\nDetailed GPU Memory Stats:\n")
                f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
        else:
            # Normal run without profiling
            metrics = train_agent(
                config_path=str(config_path),
                price_data=price_data,
                features=features,
                save_dir=trial_dir,
                device=device,
                disable_progress=True
            )
        
        # Calculate objective metrics
        final_sharpe = metrics["sharpe_ratios"][-1] if metrics["sharpe_ratios"] else -10.0
        final_portfolio_value = metrics["portfolio_values"][-1] if metrics["portfolio_values"] else 0.0
        max_drawdown = max(metrics["max_drawdowns"]) if metrics["max_drawdowns"] else 1.0
        
        # Calculate objective value with more forgiving terms
        portfolio_return = (final_portfolio_value - 1.0)
        drawdown_penalty = np.clip(max_drawdown - 0.15, 0, 1)
        scaled_sharpe = np.clip(final_sharpe, -5, 5)
        
        # Adjusted weights to prioritize portfolio value
        objective_value = (
            0.3 * scaled_sharpe +  # Reduced weight on Sharpe
            0.5 * portfolio_return +  # Increased weight on returns
            0.2 * (-drawdown_penalty)
        )
        
        # Log detailed metrics
        print(f"\nTrial {trial.number} metrics:")
        print(f"  Sharpe Ratio: {final_sharpe:.4f} (scaled: {scaled_sharpe:.4f})")
        print(f"  Portfolio Return: {portfolio_return:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.4f} (penalty: {drawdown_penalty:.4f})")
        print(f"  Final Objective: {objective_value:.4f}")
        
        # Log hyperparameters
        print("\nHyperparameters:")
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
        
        # Save trial results with detailed metrics
        results = {
            "final_sharpe": final_sharpe,
            "scaled_sharpe": scaled_sharpe,
            "portfolio_return": portfolio_return,
            "max_drawdown": max_drawdown,
            "drawdown_penalty": drawdown_penalty,
            "final_portfolio_value": final_portfolio_value,
            "objective_value": objective_value,
            "hyperparameters": config,
            "gpu_id": gpu_id,
            "training_time": time.time() - trial.datetime_start.timestamp()
        }
        with open(trial_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Update progress
        if global_progress is not None:
            global_progress.update(1)
            global_progress.refresh()
        
        return objective_value
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return float("-inf")
        
    finally:
        # Clean up CUDA memory
        if gpu_id is not None:
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()
        
        # Return GPU to queue
        if gpu_queue is not None and gpu_id is not None:
            gpu_queue.put(gpu_id)

def run_optimization(base_config, market_data, features, n_trials=100, jobs_per_gpu=12, enable_profiling=False):
    """Run hyperparameter optimization with Optuna."""
    global global_progress
    
    # Configure Optuna logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    # Set up study
    study_name = create_study_name()
    storage = optuna.storages.RDBStorage(
        url="postgresql://postgres:postgres@localhost:5432/optuna_db",
        engine_kwargs={"pool_size": jobs_per_gpu * 4}
    )
    
    # Create study with TPE sampler
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=5)
    )
    
    # Clear CUDA cache at the start
    torch.cuda.empty_cache()
    
    # Get available GPUs and set memory limits
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs available, using CPU")
        gpu_queue = None
    else:
        print(f"Found {n_gpus} GPUs")
        # Set memory limits for each GPU
        for gpu_id in range(n_gpus):
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()
                # Reserve some memory for PyTorch internal operations
                torch.cuda.set_per_process_memory_fraction(0.9 / jobs_per_gpu, gpu_id)
        
        gpu_queue = Queue()
        for gpu_id in range(n_gpus):
            for _ in range(jobs_per_gpu):
                gpu_queue.put(gpu_id)
    
    # Create progress bar
    global_progress = tqdm(total=n_trials, desc="Optimization Progress", position=0)
    
    try:
        if enable_profiling:
            # Run single trial with profiling
            print("Running single trial with profiling enabled...")
            study.optimize(
                lambda trial: objective(
                    trial, base_config, market_data, features,
                    jobs_per_gpu=jobs_per_gpu, gpu_queue=gpu_queue,
                    enable_profiling=True
                ),
                n_trials=1,  # Only run one trial when profiling
                n_jobs=1,    # Disable parallel execution for profiling
                gc_after_trial=True,
                show_progress_bar=False
            )
        else:
            # Run normal optimization with parallel execution
            study.optimize(
                lambda trial: objective(
                    trial, base_config, market_data, features,
                    jobs_per_gpu=jobs_per_gpu, gpu_queue=gpu_queue,
                    enable_profiling=False
                ),
                n_trials=n_trials,
                n_jobs=jobs_per_gpu * max(n_gpus, 1),
                gc_after_trial=True,
                show_progress_bar=False
            )
        
        # Calculate parameter importance
        importance = optuna.importance.get_param_importances(study)
        
        # Print results
        print("\nOptimization completed!")
        print(f"Best value: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        print("\nParameter importance:")
        for key, value in importance.items():
            print(f"  {key}: {value:.4f}")
        
        return study
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        return study
        
    finally:
        if global_progress is not None:
            global_progress.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for SAC trading agent")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--jobs-per-gpu", type=int, default=12, help="Number of jobs to run per GPU")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling (will run only one trial)")
    args = parser.parse_args()
    
    if args.enable_profiling:
        print("Profiling mode: Will run only one trial with detailed profiling enabled")
        args.n_trials = 1  # Override n_trials in profiling mode
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load base configuration
    with open("config/training_config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    
    # Create synthetic data with reduced size
    print("Generating synthetic market data...")
    market_data = create_synthetic_data(
        n_samples=500,
        n_assets=10,
        seed=42
    )
    
    # Calculate features
    print("Calculating technical features...")
    all_features = []
    for i in range(10):
        asset_data = market_data[[f'open_ASSET_{i+1}', f'high_ASSET_{i+1}', 
                                f'low_ASSET_{i+1}', f'close_ASSET_{i+1}', 
                                f'volume_ASSET_{i+1}']]
        asset_data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        feature_config = {
            'window_sizes': [20],
            'normalization': 'zscore',
            'normalization_lookback': 100,
            'features': {
                'price_based': ['sma', 'ema'],
                'momentum': ['rsi'],
                'volume': ['vwap'],
                'volatility': ['volatility']
            }
        }
        features = prepare_market_features(asset_data, feature_config)
        features.columns = [f'{col}_ASSET_{i+1}' for col in features.columns]
        all_features.append(features)
    
    features = pd.concat(all_features, axis=1)
    market_data = market_data.loc[features.index]
    
    print(f"Data shape: {market_data.shape}, Features shape: {features.shape}")
    
    study = run_optimization(base_config, market_data, features, 
                           n_trials=args.n_trials, 
                           jobs_per_gpu=args.jobs_per_gpu,
                           enable_profiling=args.enable_profiling) 