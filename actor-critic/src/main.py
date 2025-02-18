import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from src.train.train import train_agent, evaluate_agent
from src.utils.risk_management import RiskParams

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SAC trading agent")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--price_data", type=str, required=True, help="Path to price data")
    parser.add_argument("--features", type=str, required=True, help="Path to feature data")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save models and logs")
    parser.add_argument("--model_path", type=str, help="Path to model for evaluation")
    args = parser.parse_args()
    
    # Load data
    price_data = pd.read_csv(args.price_data)
    features = pd.read_csv(args.features)
    
    if args.mode == "train":
        # Train agent
        metrics = train_agent(
            config_path=args.config,
            price_data=price_data,
            features=features,
            save_dir=Path(args.save_dir)
        )
        
        # Ensure all arrays have the same length
        min_length = min(len(value) for value in metrics.values() if isinstance(value, (list, np.ndarray)))
        processed_metrics = {
            key: value[:min_length] if isinstance(value, (list, np.ndarray)) else value
            for key, value in metrics.items()
        }
        
        # Save metrics
        metrics_df = pd.DataFrame(processed_metrics)
        metrics_df.to_csv(Path(args.save_dir) / "training_metrics.csv", index=False)
        
    else:  # Evaluate
        if not args.model_path:
            raise ValueError("Model path must be provided for evaluation")
            
        # Load config for risk parameters
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Evaluate agent
        metrics = evaluate_agent(
            model_path=args.model_path,
            price_data=price_data,
            features=features,
            risk_params=RiskParams(**config["risk_params"])
        )
        
        # Print evaluation metrics
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main() 