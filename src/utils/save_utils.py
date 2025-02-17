# utils/save_utils.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import shutil
from datetime import datetime
from pathlib import Path
from .fs import ensure_dir, safe_save

def save_experiment(model, optimizer, metrics, checkpoint_dir="checkpoints", current_file_path=None, avg_loss=None):
    """
    Save experiment results including model checkpoint and metrics.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        metrics: Dictionary containing performance metrics
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        current_file_path: Path to the current running script
        avg_loss: The average training loss
    """
    checkpoint_dir = ensure_dir(checkpoint_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save metrics separately
    metrics_path = Path(checkpoint_dir) / "metrics.json"
    safe_save(metrics, metrics_path)
    
    # Copy current model file if provided
    if current_file_path and os.path.exists(current_file_path):
        try:
            shutil.copy2(current_file_path, os.path.join(checkpoint_dir, "model.py"))
        except Exception as e:
            print(f"Warning: Could not copy model.py file: {e}")
    
    # Create and save detailed log
    log_content = f"""Experiment Log - {timestamp}
===============================

Model Performance Metrics
-----------------------
Excess Return: {metrics.get('excess_return', 'N/A')}
Profit: {metrics.get('profit', 0)*100:.2f}%
Episode: {metrics.get('episode', 'N/A')}
Win Rate: {metrics.get('win_rate', 0)*100:.2f}%

Training Parameters
------------------
Learning Rate: {optimizer.param_groups[0]['lr']:.8f}
"""
    if avg_loss is not None:
        log_content += f"Average Loss: {avg_loss:.8f}\n"
    
    # Add per-ticker metrics if available
    if 'per_ticker_metrics' in metrics:
        log_content += "\nPer-Ticker Performance\n---------------------\n"
        for ticker, ticker_metrics in metrics['per_ticker_metrics'].items():
            log_content += f"\n{ticker}:\n"
            log_content += f"  Profit: {ticker_metrics['profit']*100:.2f}%\n"
            log_content += f"  Win Rate: {ticker_metrics['win_rate']*100:.1f}%\n"
            log_content += f"  Trades: {ticker_metrics['num_trades']}\n"
            if 'buy_and_hold_return' in ticker_metrics:
                excess = (ticker_metrics['profit'] - ticker_metrics['buy_and_hold_return'])*100
                log_content += f"  Excess Return: {excess:.2f}%\n"
    
    log_path = Path(checkpoint_dir) / f"experiment_log_{timestamp}.txt"
    with open(log_path, 'w') as f:
        f.write(log_content)
    
    print(f"\nExperiment saved to: {checkpoint_dir}")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Log: {log_path}")
    if current_file_path:
        print(f"  - Model Code: {os.path.join(checkpoint_dir, 'model.py')}")
    
    return str(checkpoint_path)

def save_last_checkpoint(model, optimizer, metrics=None, checkpoint_dir="checkpoints", filename="LAST_checkpoint.pt"):
    """
    Save the latest checkpoint during training.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        metrics: Dictionary containing metrics
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        filename: Name of the checkpoint file (default: "LAST_checkpoint.pt")
    """
    checkpoint_dir = ensure_dir(checkpoint_dir)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics if metrics is not None else {},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    checkpoint_path = Path(checkpoint_dir) / filename
    torch.save(checkpoint, checkpoint_path)
    
    print(f"[CHECKPOINT] Saved last checkpoint to {checkpoint_path}")
    return str(checkpoint_path)

def load_checkpoint(checkpoint_path, device=None):
    """Load checkpoint with proper device handling."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    
    # Load with proper device mapping and weights_only=True for security
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Ensure state dicts are on correct device
    if 'model_state_dict' in checkpoint:
        checkpoint['model_state_dict'] = {
            k: v.to(device) for k, v in checkpoint['model_state_dict'].items()
        }
    
    return checkpoint