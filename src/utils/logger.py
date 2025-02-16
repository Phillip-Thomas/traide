# utils/logger.py
import os
import json
from datetime import datetime
import numpy as np
from collections import deque

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.jsonl")
        
        # Tracking metrics
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.trades_history = deque(maxlen=1000)
        self.validation_history = []
        
        # Initialize files
        self._write_header()
    
    def _write_header(self):
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - Started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_episode(self, episode_num, returns, length, std_dev, priority, 
                   epsilon, loss=None, trades_info=None):
        """Log episode metrics"""
        self.episode_returns.append(returns)
        self.episode_lengths.append(length)
        if trades_info:
            self.trades_history.extend(trades_info)
        
        # Calculate rolling statistics
        avg_return = np.mean(self.episode_returns)
        avg_length = np.mean(self.episode_lengths)
        
        msg = f"\nEpisode {episode_num:4d}"
        msg += f"\n  Returns: {returns:7.4f} (Avg: {avg_return:7.4f})"
        msg += f"\n  Length:  {length:5d}   (Avg: {avg_length:7.1f})"
        msg += f"\n  Std Dev: {std_dev:7.4f}"
        msg += f"\n  Priority: {priority:7.4f}"
        msg += f"\n  Epsilon:  {epsilon:7.4f}"
        if loss is not None:
            msg += f"\n  Loss:     {loss:7.4f}"
        
        if trades_info:
            profitable_trades = sum(1 for t in trades_info if t > 0)
            total_trades = len(trades_info)
            if total_trades > 0:
                win_rate = profitable_trades / total_trades * 100
                msg += f"\n  Trades:   {total_trades:4d} (Win Rate: {win_rate:5.1f}%)"
        
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
        print(msg)
        
        # Save metrics to JSON
        metrics = {
            'episode': episode_num,
            'returns': returns,
            'length': length,
            'std_dev': std_dev,
            'priority': priority,
            'epsilon': epsilon,
            'loss': loss,
            'avg_return': avg_return,
            'avg_length': avg_length,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def log_validation(self, episode_num, metrics):
        """Log validation results"""
        self.validation_history.append(metrics)
        
        msg = f"\nValidation Results (Episode {episode_num})"
        msg += f"\n  Excess Return: {metrics['excess_return']:7.2f}%"
        msg += f"\n  Profit:        {metrics['profit']*100:7.2f}%"
        msg += f"\n  Win Rate:      {metrics['win_rate']*100:7.2f}%"
        msg += f"\n  Trades:        {metrics['num_trades']:7d}"
        
        # Per-ticker breakdown
        if 'per_ticker_metrics' in metrics:
            msg += "\n\n  Per-Ticker Performance:"
            for ticker, ticker_metrics in metrics['per_ticker_metrics'].items():
                msg += f"\n    {ticker}:"
                msg += f"\n      Profit: {ticker_metrics['profit']*100:7.2f}%"
                msg += f"\n      Win Rate: {ticker_metrics['win_rate']*100:5.1f}%"
                msg += f"\n      Trades: {ticker_metrics['num_trades']:5d}"
        
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n" + "-"*80 + "\n")
        print(msg)
        
        # Save validation metrics
        metrics['episode'] = episode_num
        metrics['timestamp'] = datetime.now().isoformat()
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps({'validation': metrics}) + '\n')
    
    def log_model_save(self, episode_num, save_path, metrics):
        """Log model checkpoint saving"""
        msg = f"\nSaved Model Checkpoint (Episode {episode_num})"
        msg += f"\n  Path: {save_path}"
        msg += f"\n  Metrics:"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f"\n    {key}: {value:7.4f}"
            else:
                msg += f"\n    {key}: {value}"
        
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n" + "-"*80 + "\n")
        print(msg)
    
    def log_training_update(self, episode_num, learning_rate, grad_norm=None):
        """Log training progress update"""
        msg = f"\nTraining Update (Episode {episode_num})"
        msg += f"\n  Learning Rate: {learning_rate:.2e}"
        if grad_norm is not None:
            msg += f"\n  Gradient Norm: {grad_norm:.4f}"
        
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
        print(msg)
    
    def get_summary_stats(self):
        """Return summary statistics of training"""
        return {
            'avg_return': np.mean(self.episode_returns),
            'std_return': np.std(self.episode_returns),
            'max_return': np.max(self.episode_returns),
            'min_return': np.min(self.episode_returns),
            'avg_length': np.mean(self.episode_lengths),
            'total_trades': len(self.trades_history),
            'win_rate': np.mean([t > 0 for t in self.trades_history]) if self.trades_history else 0,
            'best_validation': max(self.validation_history, key=lambda x: x['excess_return']) if self.validation_history else None
        }