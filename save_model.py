import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"# save_model.py
import torch
from model import DQN  # Import the model class from model.py

def save_current_model(model, file_path='current_model.pt'):
    # Save only the state_dict (recommended)
    torch.save(model.state_dict(), file_path)
    print(f"Model state_dict saved to {file_path}")

if __name__ == '__main__':
    # Define your input size based on your model's requirements.
    input_size = 241  # Adjust this as needed.
    
    # Instantiate your model
    model = DQN(input_size)
    
    # Optionally, load a checkpoint if needed:
    # checkpoint = torch.load('path/to/checkpoint.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save the current model state.
    save_current_model(model)
