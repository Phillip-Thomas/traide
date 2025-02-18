import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """Logger for training metrics and evaluation results."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
            config: Training configuration
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set experiment name with unique timestamp
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{datetime.now().strftime('%H%M%S_%f')}"
        
        # Initialize logging with unique name
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers and close them
        if self.logger.hasHandlers():
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        # Add file handler with unique name
        log_file = self.log_dir / f"{self.experiment_name}.log"
        if log_file.exists():
            log_file.unlink()  # Remove existing log file
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Initialize tensorboard with unique name
        self.writer = SummaryWriter(self.log_dir / f"tensorboard_{self.experiment_name}")
        
        # Save config if provided
        if config is not None:
            config_file = self.log_dir / f"config_{self.experiment_name}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
        
        # Initialize metrics storage
        self.metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
            "positions": [],
            "returns": []
        }
        
        self.logger.info(f"Initialized logger in {self.log_dir}")
    
    def log_episode(
        self,
        episode: int,
        metrics: Dict[str, float],
        step_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            metrics: Episode metrics
            step_metrics: Training step metrics
        """
        # Validate metrics
        required_metrics = {"reward", "portfolio_value", "sharpe_ratio", "max_drawdown"}
        if not all(key in metrics for key in required_metrics):
            raise ValueError(f"Missing required metrics. Expected {required_metrics}")
        
        # Store metrics
        self.metrics["episode_rewards"].append(metrics["reward"])
        self.metrics["portfolio_values"].append(metrics["portfolio_value"])
        self.metrics["sharpe_ratios"].append(metrics["sharpe_ratio"])
        self.metrics["max_drawdowns"].append(metrics["max_drawdown"])
        
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Episode {episode} - {metric_str}")
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"episode/{key}", value, episode)
        
        if step_metrics:
            for key, value in step_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, episode)
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log training step metrics.
        
        Args:
            step: Training step
            metrics: Training metrics
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, step)
    
    def log_evaluation(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics: Evaluation metrics
            step: Current training step
        """
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Evaluation Results - {metric_str}")
        
        if step is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, step)
    
    def save_metrics(self) -> None:
        """Save metrics to CSV and JSON files."""
        # Skip if no metrics have been logged
        if not any(self.metrics.values()):
            self.logger.warning("No metrics to save")
            return
            
        # Ensure all metric arrays have the same length
        max_len = max(len(v) for v in self.metrics.values() if v)
        if max_len == 0:
            self.logger.warning("No metrics to save")
            return
            
        metrics_dict = {}
        for key, values in self.metrics.items():
            # Skip empty arrays
            if not values:
                continue
            # Pad with None if necessary
            if len(values) < max_len:
                values.extend([None] * (max_len - len(values)))
            metrics_dict[key] = values
        
        # Save to CSV if we have metrics
        if metrics_dict:
            metrics_df = pd.DataFrame(metrics_dict)
            metrics_df.to_csv(self.log_dir / "training_metrics.csv", index=False)
        
        # Save summary statistics
        summary = {}
        if self.metrics["portfolio_values"]:
            summary["final_portfolio_value"] = self.metrics["portfolio_values"][-1]
        if self.metrics["episode_rewards"]:
            summary["mean_reward"] = sum(self.metrics["episode_rewards"]) / len(self.metrics["episode_rewards"])
        if self.metrics["sharpe_ratios"]:
            summary["final_sharpe"] = self.metrics["sharpe_ratios"][-1]
        if self.metrics["max_drawdowns"]:
            summary["max_drawdown"] = max(v for v in self.metrics["max_drawdowns"] if v is not None)
        if self.metrics["positions"]:
            summary["mean_position"] = sum(v for v in self.metrics["positions"] if v is not None) / sum(1 for v in self.metrics["positions"] if v is not None)
        
        # Save summary if we have any statistics
        if summary:
            with open(self.log_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=4)
    
    def close(self) -> None:
        """Clean up and save final metrics."""
        self.save_metrics()
        self.writer.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 