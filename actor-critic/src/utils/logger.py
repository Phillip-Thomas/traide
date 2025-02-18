import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """
    Logger for tracking training progress and performance metrics.
    Handles both file logging and tensorboard visualization.
    """
    def __init__(
        self,
        log_dir: Path,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Initialize file logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / f"{experiment_name}.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.log_dir / "tensorboard" / experiment_name)
        
        # Save config if provided
        if config is not None:
            with open(self.log_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=4)
        
        # Initialize metric storage
        self.metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
            "positions": [],
            "returns": []
        }
        
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
            metrics: Episode-level metrics
            step_metrics: Step-level metrics to average
        """
        # Update stored metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log to file
        self.logger.info(f"Episode {episode} - " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"episode/{key}", value, episode)
        
        if step_metrics:
            for key, value in step_metrics.items():
                self.writer.add_scalar(f"step/{key}", value, episode)
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log training step metrics.
        
        Args:
            step: Global step number
            metrics: Training metrics
        """
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"training/{key}", value, step)
    
    def log_evaluation(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics: Evaluation metrics
            step: Optional step number for tensorboard
        """
        # Log to file
        self.logger.info("Evaluation Results - " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        
        # Log to tensorboard if step provided
        if step is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"evaluation/{key}", value, step)
    
    def save_metrics(self) -> None:
        """Save accumulated metrics to CSV."""
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(self.log_dir / "metrics.csv", index=False)
        
        # Calculate and log summary statistics
        summary = {
            "final_portfolio_value": self.metrics["portfolio_values"][-1],
            "mean_reward": np.mean(self.metrics["episode_rewards"]),
            "final_sharpe": self.metrics["sharpe_ratios"][-1],
            "max_drawdown": max(self.metrics["max_drawdowns"]),
            "mean_position": np.mean(np.abs(self.metrics["positions"]))
        }
        
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
    
    def close(self) -> None:
        """Close tensorboard writer and save final metrics."""
        self.save_metrics()
        self.writer.close() 