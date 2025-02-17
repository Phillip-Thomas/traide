# utils/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

class TrainingVisualizer:
    """Visualizer for training metrics and results."""
    
    def __init__(self, metrics_file):
        """Initialize visualizer with metrics file."""
        self.metrics_file = Path(metrics_file)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file."""
        if not self.metrics_file.exists():
            return {}
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def plot_training_progress(self, save_dir="plots"):
        """Generate training progress plots"""
        if not self.metrics or not self.metrics.get('episodes'):
            return None
            
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot returns
        episodes = self.metrics['episodes']
        returns = self.metrics['returns']
        ax1.plot(episodes, returns, label='Episode Return')
        ax1.set_title('Training Returns')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.grid(True)
        
        # Plot loss
        if 'losses' in self.metrics and self.metrics['losses']:
            ax2.plot(episodes, self.metrics['losses'], label='Loss')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # Plot validation metrics
        if 'validations' in self.metrics and self.metrics['validations']:
            # Extract validation metrics, handling missing episode keys
            val_data = []
            for i, v in enumerate(self.metrics['validations']):
                episode = v.get('episode', episodes[min(i, len(episodes)-1)])
                profit = v.get('profit', 0)
                val_data.append((episode, profit))
            
            if val_data:
                val_episodes, val_profits = zip(*val_data)
                ax3.plot(val_episodes, val_profits, label='Validation Profit')
                ax3.set_title('Validation Performance')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Profit')
                ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'training_progress_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
        return fig
    
    def plot_trade_distribution(self):
        """Plot distribution of trade returns"""
        if not self.metrics or 'validations' not in self.metrics:
            return None
            
        # Collect all trade returns from validations
        all_trades = []
        for validation in self.metrics['validations']:
            if isinstance(validation.get('trades'), list):
                all_trades.extend([trade['profit'] for trade in validation['trades']])
        
        if not all_trades:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_trades, bins=50, density=True, alpha=0.75)
        ax.axvline(x=0, color='r', linestyle='--', label='Break Even')
        
        mean_return = np.mean(all_trades)
        ax.axvline(x=mean_return, color='g', linestyle='--', 
                  label=f'Mean Return ({mean_return:.2%})')
        
        ax.set_title('Trade Returns Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def generate_summary_report(self, save_dir="reports", output_file=None):
        """Generate a summary report of training"""
        os.makedirs(save_dir, exist_ok=True)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(save_dir, f'training_summary_{timestamp}.txt')
        
        report = ["Training Summary", "=" * 50, ""]
        
        # Basic statistics
        report.append(f"Total Episodes: {len(self.metrics['episodes'])}")
        report.append(f"Average Return: {np.mean(self.metrics['returns']):.4f}")
        report.append(f"Max Return: {max(self.metrics['returns']):.4f}")
        
        if 'losses' in self.metrics and self.metrics['losses']:
            report.append(f"Final Loss: {self.metrics['losses'][-1]:.4f}")
        
        # Validation performance
        if 'validations' in self.metrics and self.metrics['validations']:
            best_validation = max(self.metrics['validations'], 
                                key=lambda x: x.get('profit', float('-inf')))
            report.extend([
                "",
                "Best Validation Performance",
                "-" * 30,
                f"Episode: {best_validation.get('episode', 'N/A')}",
                f"Profit: {best_validation.get('profit', 0):.2%}",
                f"Win Rate: {best_validation.get('win_rate', 0):.2%}",
                f"Number of Trades: {best_validation.get('num_trades', 0)}"
            ])
        
        report = "\n".join(report)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to: {output_file}")
        return output_file