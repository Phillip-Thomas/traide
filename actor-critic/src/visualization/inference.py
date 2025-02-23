import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from src.models.sac_agent import SACAgent
from src.env.trading_env import TradingEnvironment

class InferenceVisualizer:
    """
    Handles model inference and creates visualizations for trading performance.
    """
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        save_dir: str = "visualizations",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize environment
        self.env = TradingEnvironment(
            price_data=price_data,
            features=features,
            window_size=self.config["window_size"],
            commission=self.config["commission"],
            **self.config.get("env_params", {})
        )
        
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **self.config["agent_params"]
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.eval()
        
    def run_inference(self, num_episodes: int = 1) -> Dict:
        """Run inference and collect trading metrics and positions."""
        all_metrics = []
        all_positions = []
        all_portfolio_values = []
        all_returns = []
        all_prices = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_positions = []
            episode_portfolio_values = [1.0]
            episode_returns = []
            episode_prices = []
            
            while not done:
                # Select action
                with torch.no_grad():
                    action = self.agent.select_action(state, evaluate=True)
                
                # Take step
                next_state, reward, done, _, info = self.env.step(action)
                
                # Record data
                episode_positions.append(info['position'])
                episode_portfolio_values.append(info['portfolio_value'])
                episode_returns.append(info['step_return'])
                episode_prices.append(info['current_price'])
                
                state = next_state
            
            # Calculate episode metrics
            metrics = {
                'final_portfolio_value': episode_portfolio_values[-1],
                'total_return': (episode_portfolio_values[-1] - 1.0) * 100,
                'sharpe_ratio': np.mean(episode_returns) / (np.std(episode_returns) + 1e-6) * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(episode_portfolio_values),
                'avg_position': np.mean(np.abs(episode_positions)),
                'position_changes': np.sum(np.abs(np.diff(episode_positions)))
            }
            
            all_metrics.append(metrics)
            all_positions.append(episode_positions)
            all_portfolio_values.append(episode_portfolio_values)
            all_returns.append(episode_returns)
            all_prices.append(episode_prices)
        
        return {
            'metrics': all_metrics,
            'positions': all_positions,
            'portfolio_values': all_portfolio_values,
            'returns': all_returns,
            'prices': all_prices
        }
    
    def visualize_trading_performance(self, inference_data: Dict, episode_idx: int = 0):
        """Create comprehensive trading performance visualization."""
        positions = inference_data['positions'][episode_idx]
        portfolio_values = inference_data['portfolio_values'][episode_idx]
        prices = inference_data['prices'][episode_idx]
        metrics = inference_data['metrics'][episode_idx]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Price and Portfolio Value Plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_and_portfolio(ax1, prices, portfolio_values)
        
        # 2. Position Heatmap
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_position_heatmap(ax2, positions, prices)
        
        # 3. Position Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_position_distribution(ax3, positions)
        
        # 4. Metrics Table
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_metrics_table(ax4, metrics)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.save_dir / 'trading_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_price_and_portfolio(self, ax, prices, portfolio_values):
        """Plot price and portfolio value."""
        ax.plot(prices, label='Price', color='gray', alpha=0.6)
        ax2 = ax.twinx()
        ax2.plot(portfolio_values, label='Portfolio Value', color='blue')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price', color='gray')
        ax2.set_ylabel('Portfolio Value', color='blue')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Price and Portfolio Value Over Time')
    
    def _plot_position_heatmap(self, ax, positions, prices):
        """Create position heatmap overlay on price."""
        # Normalize positions to [-1, 1] for color mapping
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
        # Create colormap (red for short, green for long)
        cmap = mcolors.LinearSegmentedColormap.from_list('', ['red', 'white', 'green'])
        
        # Plot price line
        ax.plot(prices, color='black', alpha=0.6, zorder=1)
        
        # Create position heatmap
        for i in range(len(positions)-1):
            ax.fill_between([i, i+1], [min(prices), min(prices)], [max(prices), max(prices)],
                          color=cmap(norm(positions[i])), alpha=0.3)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.set_title('Position Heatmap (Red=Short, Green=Long)')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Position Size')
    
    def _plot_position_distribution(self, ax, positions):
        """Plot position size distribution."""
        sns.histplot(positions, bins=50, ax=ax)
        ax.set_xlabel('Position Size')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Size Distribution')
    
    def _plot_metrics_table(self, ax, metrics):
        """Create a table of performance metrics."""
        ax.axis('off')
        cell_text = [[f"{k}", f"{v:.2f}"] for k, v in metrics.items()]
        table = ax.table(cellText=cell_text, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Performance Metrics')
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return float(np.max(drawdown))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trading agent performance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--price_data", type=str, required=True, help="Path to price data CSV")
    parser.add_argument("--features", type=str, required=True, help="Path to features CSV")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()
    
    # Load data
    price_data = pd.read_csv(args.price_data)
    features = pd.read_csv(args.features)
    
    # Initialize visualizer
    visualizer = InferenceVisualizer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        price_data=price_data,
        features=features,
        save_dir=args.save_dir
    )
    
    # Run inference
    inference_data = visualizer.run_inference(num_episodes=args.episodes)
    
    # Create visualizations
    visualizer.visualize_trading_performance(inference_data)
    
    # Print metrics
    print("\nPerformance Metrics:")
    for i, metrics in enumerate(inference_data['metrics']):
        print(f"\nEpisode {i+1}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main() 