# utils/logger.py
import os
import json
from datetime import datetime
import numpy as np
from collections import deque
from pathlib import Path
from .fs import ensure_dir, safe_save, safe_load

class TrainingLogger:
    """Logger for training metrics and progress."""
    
    def __init__(self, log_dir="logs"):
        """Initialize logger with directory for storing metrics."""
        self.log_dir = ensure_dir(log_dir)
        self.metrics_file = Path(log_dir) / "metrics.json"
        self.metrics = {
            'episodes': [],
            'returns': [],
            'lengths': [],
            'losses': [],
            'priorities': [],
            'validations': [],
            'model_saves': [],
            'training_updates': []
        }
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file."""
        safe_save(self.metrics, self.metrics_file)
    
    def _load_metrics(self):
        """Load metrics from file."""
        loaded = safe_load(self.metrics_file)
        if loaded:
            self.metrics = loaded
    
    def log_episode(self, episode_num, returns, length, std_dev, priority, epsilon, loss, trades_info):
        """Log episode metrics."""
        self.metrics['episodes'].append(episode_num)
        self.metrics['returns'].append(float(returns))
        self.metrics['lengths'].append(int(length))
        self.metrics['losses'].append(float(loss))
        self.metrics['priorities'].append(float(priority))
        self._save_metrics()
    
    def log_validation(self, metrics, episode=None):
        """Log validation metrics."""
        if episode is not None:
            metrics['episode'] = episode
        self.metrics['validations'].append(metrics)
        self._save_metrics()
    
    def log_model_save(self, metrics, path, episode=None):
        """Log model save event."""
        save_info = {
            'path': str(path),
            'metrics': metrics
        }
        if episode is not None:
            save_info['episode'] = episode
        self.metrics['model_saves'].append(save_info)
        self._save_metrics()
    
    def log_training_update(self, episode, learning_rate=None, grad_norm=None, loss=None, epsilon=None, num_transitions=None, duration=None):
        """Log training update metrics."""
        update_info = {
            'episode': episode,
            'learning_rate': float(learning_rate) if learning_rate is not None else None,
            'grad_norm': float(grad_norm) if grad_norm is not None else None,
            'loss': float(loss) if loss is not None else None,
            'epsilon': float(epsilon) if epsilon is not None else None,
            'num_transitions': int(num_transitions) if num_transitions is not None else None,
            'duration': float(duration) if duration is not None else None
        }
        self.metrics['training_updates'].append(update_info)
        self._save_metrics()
    
    def get_summary_stats(self):
        """Get summary statistics of training."""
        if not self.metrics['returns']:
            return {}
            
        return {
            'total_episodes': len(self.metrics['episodes']),
            'avg_return': sum(self.metrics['returns']) / len(self.metrics['returns']),
            'max_return': max(self.metrics['returns']),
            'avg_episode_length': sum(self.metrics['lengths']) / len(self.metrics['lengths']),
            'final_loss': self.metrics['losses'][-1] if self.metrics['losses'] else None,
            'num_validations': len(self.metrics['validations']),
            'num_model_saves': len(self.metrics['model_saves'])
        }