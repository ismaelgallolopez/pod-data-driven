import torch
import warnings
from torch.utils.data import DataLoader, TensorDataset
from src.training.train_pinn import train_pinn
import os

def main():
    # 1. Load the tensorized data
    data_path = 'data/processed/noisy_spp.pt'
    if not os.path.exists(data_path):
        print("Data not found. Run: python src/utils/sp3_parser.py first.")
        return

    # Try using weights_only to avoid FutureWarning in newer torch versions;
    # fall back for older torch that doesn't support the kwarg.
    try:
        data = torch.load(data_path, weights_only=True)
    except TypeError:
        # Suppress the FutureWarning about weights_only for backwards compatibility
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
            data = torch.load(data_path)
    
    # data format: [t, x, y, z]
    t = data[:, 0]
    r = data[:, 1:4] # Position in km

    # Optionally load truth (if present) — not required for filtering
    truth_path = 'data/processed/truth_odcp.pt'
    truth = None
    if os.path.exists(truth_path):
        try:
            truth = torch.load(truth_path)
        except Exception:
            truth = None

    # ── Filter SPP outliers before training
    # Use a fixed physical threshold on |r| (km) to remove bad GNSS fixes.
    r_mag = torch.norm(r, dim=1)
    r_med = r_mag.median()
    mask = (r_mag - r_med).abs() < 30.0   # keep points within ±30 km of median
    n_removed = (~mask).sum().item()
    if n_removed > 0:
        pct = 100.0 * n_removed / len(t)
        print(f"Removed {n_removed} outlier epochs ({pct:.1f}%)")
        t = t[mask]
        r = r[mask]
    else:
        print("No outlier SPP epochs found")

    # 2. Chronological Split
    num_samples = len(t)
    train_idx = int(num_samples * 0.8)
    
    t_train, r_train = t[:train_idx], r[:train_idx]
    t_test, r_test = t[train_idx:], r[train_idx:]

    print(f"Total samples: {num_samples}")
    print(f"Training samples: {len(t_train)}")

    # 3. Initialize and Train the PINN (Stage 1)
    print("Stage 1: PINN kinematic smoothing")
    pinn_model = train_pinn(t_train, r_train, epochs=2000)

    # 4. Save the smoothed model
    torch.save(pinn_model.state_dict(), 'data/processed/pinn_smoother.pth')
    print("\n>>> PINN Training Complete. Model saved.")

if __name__ == "__main__":
    main()