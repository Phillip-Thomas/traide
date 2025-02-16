# Traide

A deep reinforcement learning trading system optimized for GPU acceleration.

## Features

- GPU-accelerated trading environment
- Distributed training support
- Advanced technical indicators
- Efficient memory management
- Automatic device selection (CUDA > MPS > CPU)
- Caching system for market data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traide.git
cd traide
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

To run the training:

```bash
python -m src.main
```

The system will automatically:
- Use the best available device (CUDA GPU, Apple Silicon, or CPU)
- Load or download market data
- Create the necessary directories
- Start training with optimal batch sizes

## Directory Structure

- `src/`: Source code
  - `data_management.py`: Market data handling
  - `technical_indicators.py`: Technical analysis
  - `environment.py`: Trading environment
  - `models.py`: Neural network models
  - `training.py`: Training loop
  - `utils.py`: Utility functions
  - `main.py`: Main execution
- `checkpoints/`: Model checkpoints
- `experiments/`: Experiment logs
- `results/`: Training results
- `data_cache/`: Cached market data

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA toolkit (optional, for GPU support)
- Other dependencies are listed in setup.py