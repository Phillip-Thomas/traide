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
        
        # Convert state to tensor and match training format
        state_tensor = (
            torch.from_numpy(np.array(state))
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        
        valid_actions = env.get_valid_actions()

        # Get Q-values with new format
        q_values, _ = model(state_tensor)  # shape [1,3]
        q_values = q_values.squeeze(0)     # shape [3]

        # Mask invalid actions
        mask = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            mask[a] = q_values[a]
        
        action = torch.argmax(mask).item()
        
        signals.append(action)
        
        # Calculate portfolio value
        if env.position == 0:
            portfolio_values.append(env.cash)
        else:
            portfolio_values.append(env.shares * env.close_prices[env.idx])
        
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
    
    # Get market data
    full_data_dict = get_market_data_multi()
    if not full_data_dict:
        print("No data available. Exiting.")
        return

    # Initialize model with same parameters as training
    window_size = 48
    input_size = get_state_size(window_size)
    model = DQN(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    while True:
        # Print available tickers
        tickers = list(full_data_dict.keys())
        print("\nAvailable tickers:")
        for i, ticker in enumerate(tickers):
            print(f"{i+1}. {ticker}")
        print("0. Exit")
            
        # Get user selection
        while True:
            try:
                selection = int(input("\nSelect a ticker (enter number, 0 to exit): ")) - 1
                if selection == -1:  # Exit
                    print("Exiting...")
                    return
                if 0 <= selection < len(tickers):
                    selected_ticker = tickers[selection]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Split data for selected ticker
        data = full_data_dict[selected_ticker]
        split_idx = int(len(data) * 0.7)
        val_data = data[split_idx:].copy()
        
        # Create environment and get trading signals
        env = SimpleTradeEnv(val_data, window_size=window_size)
        prices, signals, portfolio_values = visualize_trades(model, env, device)
        
        # Create visualization
        plot_trading_signals(val_data, prices, signals, portfolio_values)
        
        # Print performance metrics
        final_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
        buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100
        n_trades = np.sum(signals != 0)
        win_rate = np.mean([t > 0 for t in np.diff(portfolio_values)[signals[:-1] == 2]]) * 100 if n_trades > 0 else 0
        
        print(f"\nPerformance Summary for {selected_ticker}:")
        print(f"Strategy Return: {final_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Excess Return: {final_return - buy_hold_return:.2f}%")
        print(f"Number of Trades: {n_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Trade Duration: {len(prices)/max(1, n_trades):.1f} periods")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    main()