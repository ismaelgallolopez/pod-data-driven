import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
from src.training.train_pinn import train_pinn

spp = torch.load('data/processed/noisy_spp.pt', map_location='cpu')
# Use first 20k samples for a quick smoke test
n = len(spp)
use_n = min(20000, n)
train_idx = int(use_n * 0.8)

# Training inputs: first 80% of the selected window
t_train = spp[:train_idx, 0]
r_train = spp[:train_idx, 1:4]

model = train_pinn(t_train, r_train, epochs=50, batch_size=2048, resume=False, save_freq=1000)
print('Smoke training finished')
torch.save(model.state_dict(), 'logs/smoke_model_state.pth')
