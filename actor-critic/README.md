# SAC Trading Agent

A Soft Actor-Critic (SAC) reinforcement learning implementation for continuous trading decisions, focusing on robust position sizing and risk management.

## Features

- Continuous action space for fine-grained position sizing
- Risk-aware trading with position limits and drawdown control
- Efficient prioritized experience replay
- Advanced feature engineering pipeline
- Comprehensive testing suite
- Tensorboard integration for monitoring

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd actor-critic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
actor-critic/
├── config/
│   ├── feature_config.yaml     # Feature engineering configuration
│   └── training_config.yaml    # Training hyperparameters
├── src/
│   ├── models/
│   │   ├── sac_actor.py       # Actor network implementation
│   │   ├── sac_critic.py      # Critic networks implementation
│   │   └── sac_agent.py       # SAC agent implementation
│   ├── data/
│   │   └── feature_engineering.py  # Feature calculation
│   ├── env/
│   │   └── trading_env.py     # Trading environment
│   ├── utils/
│   │   ├── replay_buffer.py   # Experience replay implementation
│   │   ├── risk_management.py # Risk management utilities
│   │   └── logger.py          # Logging utilities
│   ├── train/
│   │   └── train.py           # Training loop implementation
│   └── main.py                # Main entry point
└── tests/
    ├── unit/                  # Unit tests
    └── integration/           # Integration tests
```

## Usage

### Training

To train the agent:

```bash
python src/main.py --mode train \
    --config config/training_config.yaml \
    --price_data path/to/price_data.csv \
    --features path/to/features.csv \
    --save_dir path/to/save/directory
```

### Evaluation

To evaluate a trained model:

```bash
python src/main.py --mode evaluate \
    --config config/training_config.yaml \
    --price_data path/to/price_data.csv \
    --features path/to/features.csv \
    --model_path path/to/model.pt
```

## Configuration

### Training Configuration

Key parameters in `config/training_config.yaml`:

```yaml
num_episodes: 1000          # Number of training episodes
window_size: 50            # Observation window size
commission: 0.001          # Trading commission
start_training_after_steps: 10000  # Initial exploration steps
save_interval: 50000       # Model checkpoint interval
target_sharpe: 1.5         # Target Sharpe ratio for early stopping

risk_params:
  max_position: 1.0        # Maximum allowed position size
  max_leverage: 1.0        # Maximum allowed leverage
  position_step: 0.1       # Minimum position change increment
  max_drawdown: 0.15       # Maximum allowed drawdown
  vol_target: 0.15         # Target annualized volatility

agent_params:
  hidden_dim: 256          # Hidden layer dimensions
  buffer_size: 1000000     # Replay buffer size
  batch_size: 256          # Training batch size
  learning_rate: 0.0003    # Learning rate
```

### Feature Configuration

Key parameters in `config/feature_config.yaml`:

```yaml
window_sizes: [14, 30, 50, 100, 200]  # Technical indicator windows
normalization: zscore                  # Feature normalization method
normalization_lookback: 252            # Normalization window
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Monitoring

Training progress can be monitored using Tensorboard:

```bash
tensorboard --logdir path/to/save/directory/logs
```

Key metrics tracked:
- Episode rewards
- Portfolio value
- Sharpe ratio
- Maximum drawdown
- Position distribution
- Training losses

## Performance

The agent is designed to achieve:
- Sharpe Ratio > 1.5
- Maximum Drawdown < 15%
- Transaction costs < 0.1% per trade
