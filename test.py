import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import torch
from src.training.train_pinn import train_pinn

def run_quick_check(epochs=10, subset=1000):
    data_path = 'data/processed/noisy_spp.pt'
    if not os.path.exists(data_path):
        print("Data not found. Run: python src/utils/sp3_parser.py first.")
        return

    data = torch.load(data_path)
    t = data[:, 0]
    r = data[:, 1:4]

    num_samples = len(t)
    print(f"Total Samples: {num_samples}")

    # Subsample for quick test
    subset = int(os.environ.get('TEST_DATASET_SIZE', str(subset)))
    if num_samples > subset:
        torch.manual_seed(0)
        perm = torch.randperm(num_samples)[:subset]
        t_small = t[perm]
        r_small = r[perm]
        print(f"Using reduced training set: {subset} / {num_samples} samples")
    else:
        t_small = t
        r_small = r

    # Run a short training (no resume by default to be deterministic for quick tests)
    model = train_pinn(t_small, r_small, epochs=epochs, batch_size=512,
                       resume=False, checkpoint_dir='data/processed/test_ckpt', save_freq=5)
    print("Quick-check finished. Checkpoints and model saved to data/processed/test_ckpt")

if __name__ == '__main__':
    run_quick_check()
