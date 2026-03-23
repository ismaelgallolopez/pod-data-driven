import os
# Disable torch.compile / dynamo if it causes import errors in this environment
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.train_pinn import train_pinn

def main():
    # 1. Load the tensorized data
    data_path = 'data/processed/noisy_spp.pt'
    if not os.path.exists(data_path):
        print("Data not found. Run: python src/utils/sp3_parser.py first.")
        return

    data = torch.load(data_path)
    
    # data format: [t, x, y, z]
    t = data[:, 0]
    r = data[:, 1:4] # Position in km

    # 2. Chronological Split
    num_samples = len(t)
    train_idx = int(num_samples * 0.8)
    
    t_train, r_train = t[:train_idx], r[:train_idx]
    t_test, r_test = t[train_idx:], r[train_idx:]

    print(f"Total Samples: {num_samples}")
    print(f"Training on: {len(t_train)} points (~{24} days)")
    
    # 3. Initialize and Train the PINN (Stage 1: Kinematic Smoothing)
    # Note: We pass the training data to our trainer
    print("\n>>> Starting Stage 1: PINN Kinematic Smoothing...")
    pinn_model = train_pinn(t_train, r_train, epochs=2000)

    # 4. Save the smoothed model
    torch.save(pinn_model.state_dict(), 'data/processed/pinn_smoother.pth')
    print("\n>>> PINN Training Complete. Model saved.")

if __name__ == "__main__":
    main()