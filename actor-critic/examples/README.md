## Hyperparameter Optimization with Optuna

This project uses Optuna for systematic hyperparameter optimization of the SAC trading agent. The optimization process is distributed across multiple GPUs and uses PostgreSQL as a backend for trial storage and management.

### Setup

1. Install PostgreSQL and create the database:
```bash
# Install PostgreSQL (if not already installed)
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Create the database
sudo -u postgres createdb optuna_db
```

2. Install required packages:
```bash
# Using conda
conda install optuna psycopg2 sqlalchemy
conda install -c conda-forge optuna-dashboard
```

### Running Hyperparameter Optimization

The hyperparameter sweep script (`hyperparameter_sweep.py`) optimizes key parameters including:
- Risk management parameters (position sizes, leverage, volatility targets)
- Network architecture (hidden dimensions)
- Training parameters (learning rates, batch sizes)
- SAC-specific parameters (temperature, discount factor)

Run the optimization:
```bash
python examples/hyperparameter_sweep.py --n-trials 100 --jobs-per-gpu 12
```

Parameters:
- `--n-trials`: Total number of trials to run
- `--jobs-per-gpu`: Number of parallel jobs per GPU (default: 12)

### Monitoring Progress

#### Optuna Dashboard

Start the web-based dashboard:
```bash
# Local access only
optuna-dashboard postgresql://postgres:postgres@localhost:5432/optuna_db

# External access (accessible from other machines)
optuna-dashboard --host 0.0.0.0 --port 8080 postgresql://postgres:postgres@localhost:5432/optuna_db
```

Access the dashboard at:
- Local: http://localhost:8080
- External: http://[server-ip]:8080

The dashboard provides:
- Real-time trial monitoring
- Parameter importance analysis
- Interactive visualization of parameter relationships
- History plots of optimization progress

#### Database Queries

You can directly query the PostgreSQL database to examine the optimization process:

```sql
-- List all studies and their trial counts
SELECT s.study_name, t.state, COUNT(*) as count 
FROM trials t 
JOIN studies s ON t.study_id = s.study_id 
GROUP BY s.study_name, t.state 
ORDER BY s.study_name, t.state;

-- Examine trial parameters and their values
SELECT t.trial_id, t.state, p.param_name, p.param_value, v.value 
FROM trials t 
LEFT JOIN trial_params p ON t.trial_id = p.trial_id 
LEFT JOIN trial_values v ON t.trial_id = v.trial_id 
ORDER BY t.trial_id LIMIT 10;

-- Execute queries using psql:
sudo -u postgres psql -d optuna_db -c "YOUR_QUERY_HERE" | cat
```

### Results

The optimization results are saved in the `optimization_results` directory:
- Individual trial results in `trial_[N]/results.json`
- Best parameters in `[study_name]_results.json`
- Training metrics and logs for each trial

The best hyperparameters can be loaded into the training configuration for future runs:
```python
with open("optimization_results/best_params.json", "r") as f:
    best_params = json.load(f)
```

### Tips for Efficient Optimization

1. **GPU Utilization**: The script automatically distributes trials across available GPUs. Adjust `jobs_per_gpu` based on your GPU memory.

2. **Early Stopping**: Trials are stopped early if they show poor performance, saving computational resources.

3. **Database Management**: Regularly backup the PostgreSQL database to prevent loss of optimization history:
```bash
pg_dump -U postgres optuna_db > optuna_backup.sql
```

4. **Resource Monitoring**: Use `nvidia-smi` to monitor GPU utilization during optimization:
```bash
watch -n 1 nvidia-smi
``` 