import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Import from your main script
from model import DQN, SimpleTradeEnv, get_state_size, get_market_data_multi

def visualize_trades(model, env, device):
    state = env.reset()
    done = False
    
    prices = []
    signals = []
    portfolio_values = []
    
    while not done:
        prices.append(env.close_prices[env.idx])
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor([state]).to(device)
        
        # 1) Get valid actions from the environment
        valid_actions = env.get_valid_actions()

        # 2) Get Q-values
        q_values = model(state_tensor)[0]

        # 3) Mask out invalid actions
        mask = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            mask[a] = q_values[a]
        
        # 4) Take the argmax from masked Q-values
        action = torch.argmax(mask).item()
        
        # Record the raw action for plotting
        signals.append(action)
        
        # Record portfolio value
        if env.position == 0:
            portfolio_values.append(env.cash)
        else:
            portfolio_values.append(env.shares * env.close_prices[env.idx])
        
        # Step in environment
        state, _, done = env.step(action)
    
    return np.array(prices), np.array(signals), np.array(portfolio_values)


def plot_trading_signals(data, prices, signals, portfolio_values):
    """
    Create a comprehensive visualization of trading activity
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    fig.suptitle('Trading Model Performance Visualization', fontsize=16)
    
    # Plot 1: Price and Trading Signals
    ax1.plot(data.index[-len(prices):], prices, label='Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_points = data.index[-len(signals):][signals == 1]
    buy_prices = prices[signals == 1]
    ax1.scatter(buy_points, buy_prices, color='green', marker='^', 
                label='Buy Signal', s=100)
    
    # Plot sell signals
    sell_points = data.index[-len(signals):][signals == 2]
    sell_prices = prices[signals == 2]
    ax1.scatter(sell_points, sell_prices, color='red', marker='v',
                label='Sell Signal', s=100)
    
    ax1.set_title('Price Action and Trading Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Portfolio Value
    initial_value = portfolio_values[0]
    portfolio_returns = (portfolio_values - initial_value) / initial_value * 100
    
    # Calculate buy & hold returns for comparison
    buy_hold_shares = initial_value / prices[0]
    buy_hold_values = buy_hold_shares * prices
    buy_hold_returns = (buy_hold_values - initial_value) / initial_value * 100
    
    ax2.plot(data.index[-len(portfolio_values):], portfolio_returns, 
             label='Strategy Returns (%)', color='blue')
    ax2.plot(data.index[-len(buy_hold_values):], buy_hold_returns,
             label='Buy & Hold Returns (%)', color='gray', linestyle='--')
    
    ax2.set_title('Performance Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Returns (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'visualizations/trading_visualization_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load the best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BEST_CHECKPOINT_FILE = os.path.join("checkpoints", "best_checkpoint.pt")
    
    if not os.path.exists(BEST_CHECKPOINT_FILE):
        print("No checkpoint found. Please train the model first.")
        return
        
    checkpoint = torch.load(BEST_CHECKPOINT_FILE)
    
    full_data_dict = get_market_data_multi()

    if not full_data_dict:
        print("No data available for training. Exiting.")
        return

    # Split data for each ticker
    train_data_dict = {}
    val_data_dict = {}
    for ticker, data in full_data_dict.items():
        split_idx = int(len(data) * 0.7)
        train_data_dict[ticker] = data[:split_idx].copy()
        val_data_dict[ticker] = data[split_idx:].copy()
    
    # Initialize model
    window_size = 48
    input_size = get_state_size(window_size)
    model = DQN(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"data: {val_data_dict['SPY']}")
    # Create environment and get trading signals
    env = SimpleTradeEnv(val_data_dict['SPY'], window_size=window_size)
    prices, signals, portfolio_values = visualize_trades(model, env, device)
    
    # Create visualization
    plot_trading_signals(val_data_dict['SPY'], prices, signals, portfolio_values)
    
    # Print performance metrics
    final_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100
    n_trades = np.sum(signals != 0)
    
    print("\nPerformance Summary:")
    print(f"Strategy Return: {final_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Number of Trades: {n_trades}")
    print(f"Average Trade Duration: {len(prices)/n_trades:.1f} periods")

if __name__ == "__main__":
    main()