import os
import json
from pathlib import Path

def ensure_dir(path):
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path

def safe_save(data, file_path):
    """Safely save data to file with error handling."""
    try:
        file_path = Path(file_path)
        ensure_dir(file_path.parent)
        
        if file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(file_path, 'wb') as f:
                f.write(data)
        return True
    except Exception as e:
        print(f"Error saving file {file_path}: {str(e)}")
        return False

def safe_load(file_path):
    """Safely load data from file with error handling."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
            
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            with open(file_path, 'rb') as f:
                return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None 