# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from models.model import DQN
from train.train import train_dqn
from utils.get_market_data_multi import get_market_data_multi
from environments.trading_env import get_state_size

# Configuration
window_size = 48

def main():
    # Load and preprocess data
    full_data_dict = get_market_data_multi()
    if not full_data_dict:
        print("No data available for training. Exiting.")
        return

    # Split data into train and validation sets
    train_data_dict = {}
    val_data_dict = {}
    for ticker, data in full_data_dict.items():
        split_idx = int(len(data) * 0.7)
        train_data_dict[ticker] = data[:split_idx].copy()
        val_data_dict[ticker] = data[split_idx:].copy()

    # Setup training parameters
    input_size = get_state_size(window_size)
    print(f"Input size: {input_size}")
    
    # Initialize model and load best checkpoint if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size).to(device)
    
    best_checkpoint_file = os.path.join("checkpoints", "LAST_checkpoint.pt")
    if os.path.exists(best_checkpoint_file):
        checkpoint = torch.load(best_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        initial_best_profit = checkpoint['metrics']['profit']
        initial_best_excess = checkpoint['metrics'].get('excess_return', float('-inf'))
        print(f"Loaded checkpoint: profit={initial_best_profit*100:.2f}%, excess={initial_best_excess:.2f}%")
    else:
        initial_best_profit = float('-inf')
        initial_best_excess = float('-inf')

    # Train model
    results = train_dqn(
        train_data_dict=train_data_dict,
        val_data_dict=val_data_dict,
        input_size=input_size,
        n_episodes=1000,
        batch_size=4,
        gamma=0.99,
        initial_best_profit=initial_best_profit,
        initial_best_excess=initial_best_excess
    )

    print("Training completed!")
    return results

if __name__ == "__main__":
    main()