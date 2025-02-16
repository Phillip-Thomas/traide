# utils/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class TrainingVisualizer:
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
        
    def _load_metrics(self):
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        return metrics
    
    def plot_training_progress(self, save_dir="plots"):
        """Generate training progress plots"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract episode metrics
        episodes = [m['episode'] for m in self.metrics if 'validation' not in m]
        returns = [m['returns'] for m in self.metrics if 'validation' not in m]
        priorities = [m['priority'] for m in self.metrics if 'validation' not in m]
        
        # Extract validation metrics
        val_episodes = [m['validation']['episode'] for m in self.metrics if 'validation' in m]
        excess_returns = [m['validation']['excess_return'] for m in self.metrics if 'validation' in m]
        profits = [m['validation']['profit']*100 for m in self.metrics if 'validation' in m]
        
        # Plot returns and priorities
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(episodes, returns, label='Episode Returns')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Returns')
        ax1.set_title('Training Returns Over Time')
        ax1.grid(True)
        
        ax2.plot(episodes, priorities, label='Priority', color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Priority')
        ax2.set_title('Sample Priorities Over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_metrics_{timestamp}.png'))
        plt.close()
        
        # Plot validation metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(val_episodes, excess_returns, label='Excess Return', color='green')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Excess Return (%)')
        ax1.set_title('Validation Excess Returns')
        ax1.grid(True)
        
        ax2.plot(val_episodes, profits, label='Profit', color='blue')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Profit (%)')
        ax2.set_title('Validation Profits')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'validation_metrics_{timestamp}.png'))
        plt.close()
    
    def plot_trade_distribution(self, save_dir="plots"):
        """Plot distribution of trade returns"""
        trade_returns = []
        for m in self.metrics:
            if 'validation' in m:
                for ticker_metrics in m['validation']['per_ticker_metrics'].values():
                    if 'trades' in ticker_metrics:
                        trade_returns.extend(ticker_metrics['trades'])
        
        if trade_returns:
            plt.figure(figsize=(10, 6))
            plt.hist(trade_returns, bins=50, density=True, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Trade Return (%)')
            plt.ylabel('Density')
            plt.title('Distribution of Trade Returns')
            plt.grid(True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(save_dir, f'trade_distribution_{timestamp}.png'))
            plt.close()
    
    def generate_summary_report(self, save_dir="reports"):
        """Generate a summary report of training"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(save_dir, f'training_summary_{timestamp}.txt')
        
        # Calculate summary statistics
        episode_metrics = [m for m in self.metrics if 'validation' not in m]
        validation_metrics = [m['validation'] for m in self.metrics if 'validation' in m]
        
        with open(report_file, 'w') as f:
            f.write(f"Training Summary Report - {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Training Statistics:\n")
            f.write("-"*50 + "\n")
            f.write(f"Total Episodes: {len(episode_metrics)}\n")
            f.write(f"Average Return: {np.mean([m['returns'] for m in episode_metrics]):.4f}\n")
            f.write(f"Return Std Dev: {np.std([m['returns'] for m in episode_metrics]):.4f}\n")
            f.write(f"Average Priority: {np.mean([m['priority'] for m in episode_metrics]):.4f}\n\n")
            
            f.write("Validation Statistics:\n")
            f.write("-"*50 + "\n")
            f.write(f"Number of Validations: {len(validation_metrics)}\n")
            f.write(f"Best Excess Return: {max([m['excess_return'] for m in validation_metrics]):.2f}%\n")
            f.write(f"Average Profit: {np.mean([m['profit'] for m in validation_metrics])*100:.2f}%\n")
            
            # Add per-ticker statistics
            if validation_metrics and 'per_ticker_metrics' in validation_metrics[-1]:
                f.write("\nPer-Ticker Performance (Last Validation):\n")
                f.write("-"*50 + "\n")
                for ticker, metrics in validation_metrics[-1]['per_ticker_metrics'].items():
                    f.write(f"\n{ticker}:\n")
                    f.write(f"  Profit: {metrics['profit']*100:.2f}%\n")
                    f.write(f"  Win Rate: {metrics['win_rate']*100:.1f}%\n")
                    f.write(f"  Trades: {metrics['num_trades']}\n")
        
        print(f"Summary report saved to: {report_file}")
        return report_file