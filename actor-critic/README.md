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

# Technical Analysis Visualization System

A comprehensive visualization system for technical analysis indicators, built with Python and Cairo graphics.

## Features

### Core Components
- Base chart system with configurable dimensions and scales
- Candlestick chart for price visualization
- Volume bars with color coding
- Multiple overlay types for technical indicators

### Overlay Types
1. **Line Overlay**
   - Simple moving averages
   - Exponential moving averages
   - Custom line-based indicators

2. **Band Overlay**
   - Bollinger Bands
   - Keltner Channels
   - Custom band-based indicators

3. **Histogram Overlay**
   - MACD histogram
   - Volume profile
   - Custom histogram-based indicators

4. **Oscillator Overlay**
   - Relative Strength Index (RSI)
   - Stochastic Oscillator
   - Custom oscillator-based indicators

### Styling Options
- Configurable colors, line styles, and widths
- Fill opacity for bands and oscillators
- Smooth or straight line rendering
- Point markers
- Custom level lines and labels

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/technical-analysis-visualization.git
cd technical-analysis-visualization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from marketdata.visualization.components import BaseChart, ChartDimensions
from marketdata.visualization.overlays import LineOverlay, LineStyle
from marketdata.indicators import SimpleMovingAverage

# Create chart
dimensions = ChartDimensions(width=800, height=600)
chart = BaseChart(dimensions)

# Add moving average overlay
sma = SimpleMovingAverage(period=20)
style = LineStyle(color=(0.0, 0.0, 0.8))  # Blue
overlay = LineOverlay(sma, style)
chart.overlays.add_overlay(overlay)

# Update with data
overlay.update(data)

# Render chart
chart.render(renderer)
```

### AAPL Example
See `examples/aapl_visualization.py` for a comprehensive example that demonstrates:
- Fetching AAPL data from Yahoo Finance
- Creating a candlestick chart
- Adding multiple technical indicators:
  - Bollinger Bands
  - RSI with overbought/oversold levels
  - MACD with histogram
  - Volume bars

To run the example:
```bash
python examples/aapl_visualization.py
```

## Development

### Project Structure
```
src/
  marketdata/
    visualization/
      components/     # Chart components
      overlays/      # Technical indicator overlays
      utils/         # Utility functions and classes
    indicators/      # Technical indicators
tests/              # Unit tests
examples/           # Example scripts
```

### Running Tests
```bash
pytest tests/
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License
MIT License - see LICENSE file for details
