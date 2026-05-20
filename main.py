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

    # Optionally load truth (if present) — use safe loader to avoid FutureWarning
    def _safe_torch_load(path):
        try:
            return torch.load(path, weights_only=True)
        except TypeError:
            # older torch doesn't accept weights_only kwarg
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
                try:
                    return torch.load(path)
                except Exception:
                    return None
        except Exception:
            return None

    truth_path = 'data/processed/truth_odcp.pt'
    truth = _safe_torch_load(truth_path) if os.path.exists(truth_path) else None

    # ── Filter SPP outliers before training (IQR-based)
    # Use interquartile range to catch genuine outliers while ignoring orbital
    # oscillation. We use a loose multiplier (10× IQR) to only remove gross bad fixes.
    r_mag = torch.norm(r, dim=1)
    q25 = r_mag.quantile(0.25)
    q75 = r_mag.quantile(0.75)
    iqr = q75 - q25
    lower = q25 - 10.0 * iqr
    upper = q75 + 10.0 * iqr

    mask = (r_mag >= lower) & (r_mag <= upper)
    n_removed = (~mask).sum().item()
    if n_removed > 0:
        pct = 100.0 * n_removed / len(t)
        print(f"Removed {n_removed} outlier epochs ({pct:.1f}%)")
        t = t[mask]
        r = r[mask]
    else:
        print("No outlier SPP epochs found")

    # 2. Train on full dataset (no held-out test split for a smoother)
    num_samples = len(t)

    print(f"Total samples: {num_samples}")
    print(f"Training on all samples (full dataset for smoother)")

    # 3. Initialize and Train the PINN (Stage 1)
    print("Stage 1: PINN kinematic smoothing (full dataset)")
    pinn_model = train_pinn(t, r, epochs=2000)

    # 4. Save the smoothed model
    torch.save(pinn_model.state_dict(), 'data/processed/pinn_smoother.pth')
    print("\n>>> PINN Training Complete. Model saved.")

if __name__ == "__main__":
    main()