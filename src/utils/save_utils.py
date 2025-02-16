# utils/save_utils.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import shutil
from datetime import datetime

def save_experiment(model, optimizer, metrics, current_file_path=None):
    """
    Save experiment results including model checkpoint and metrics.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        metrics: Dictionary containing performance metrics
        current_file_path: Path to the current running script
    """
    # Create timestamp for experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 1. Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': timestamp,
    }
    checkpoint_path = os.path.join(experiment_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # 2. Copy current model.py file if provided
    if current_file_path and os.path.exists(current_file_path):
        try:
            shutil.copy2(current_file_path, os.path.join(experiment_dir, "model.py"))
        except Exception as e:
            print(f"Warning: Could not copy model.py file: {e}")
    
    # 3. Create and save detailed log
    log_content = """Experiment Log - {}
    ===============================

    Model Performance Metrics
    -----------------------
    Excess Return: {}
    Profit: {}%
    Episode: {}
    Epsilon: {}

    Training Parameters
    ------------------
    Learning Rate: {}
    """.format(
        timestamp,
        "{:.2f}".format(metrics.get('excess_return', 0)) if metrics.get('excess_return') != 'N/A' else 'N/A',
        "{:.2f}".format(metrics.get('profit', 0) * 100) if metrics.get('profit') != 'N/A' else 'N/A',
        metrics.get('episode', 'N/A'),
        "{:.4f}".format(metrics.get('epsilon', 0)) if metrics.get('epsilon') != 'N/A' else 'N/A',
        "{:.8f}".format(optimizer.param_groups[0]['lr'])
    )
    
    # Save log file
    log_path = os.path.join(experiment_dir, "experiment_log.txt")
    with open(log_path, "w") as f:
        f.write(log_content)
    
    # 4. Save best model checkpoint in a consistent location
    best_checkpoint_dir = os.path.join("checkpoints")
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(best_checkpoint_dir, "best_model.pt")
    torch.save(checkpoint, best_checkpoint_path)
    
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Log: {log_path}")
    if current_file_path:
        print(f"  - Model Code: {os.path.join(experiment_dir, 'model.py')}")
    print(f"  - Best model saved to: {best_checkpoint_path}")
    
    return experiment_dir

def save_last_checkpoint(model, optimizer, episode, metrics=None, filename="last_checkpoint.pt"):
    """Save the latest checkpoint during training."""
    checkpoint = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics if metrics is not None else {},
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    checkpoint_dir = os.path.join("checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"[CHECKPOINT] Saved checkpoint at episode {episode} to {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load a saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
        
    try:
        checkpoint = torch.load(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Checkpoint timestamp: {checkpoint.get('timestamp', 'N/A')}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None