import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.optim as optim
from src.models.pinn import KinematicPINN

spp = torch.load('data/processed/noisy_spp.pt', map_location='cpu')
ckpt = torch.load('data/processed/pinn_smoother.pth', map_location='cpu', weights_only=False)

model = KinematicPINN()
if isinstance(ckpt, dict) and 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    model.load_state_dict(ckpt)
    t = spp[:,0]
    n = len(t)
    t_train = t[:int(n*0.8)]
    t_min = t_train.min().item()
    t_scale = (t_train.max() - t_train.min()).item()
    L_star = 6378137.0

if isinstance(ckpt, dict) and 't_min' in ckpt:
    t_min = ckpt['t_min']
    t_scale = ckpt['t_scale']
    L_star = ckpt.get('L_star', 6378137.0)

# prepare training data (first 80% of full SPP)
n = len(spp)
t_train = spp[:int(n*0.8), 0]
r_train = spp[:int(n*0.8), 1:4]

model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# compute initial loss
with torch.no_grad():
    t_norm = (t_train - t_min) / t_scale
    r_pred = model(t_norm.float())
    r_nd = (r_train * 1000.0 / 6378137.0).float()
    loss0 = torch.mean((r_pred - r_nd)**2).item()
print('Initial training MSE (non-dim):', loss0)

# run a few optimization steps
batch_size = 4096
dataset = torch.utils.data.TensorDataset(((t_train - t_min)/t_scale).float(), r_nd)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(5):
    tot = 0.0
    for bt, br in loader:
        optimizer.zero_grad()
        pred = model(bt)
        loss = torch.mean((pred - br)**2)
        loss.backward()
        optimizer.step()
        tot += loss.item()
    print(f'Epoch {epoch}: avg loss {tot/len(loader):.6e}')

# final loss
with torch.no_grad():
    t_norm = (t_train - t_min) / t_scale
    lossf = torch.mean((model(t_norm.float()) - r_nd)**2).item()
print('Final training MSE (non-dim):', lossf)

# save fine-tuned state
torch.save(model.state_dict(), 'logs/ckpt_finetuned.pth')
print('Saved finetuned state to logs/ckpt_finetuned.pth')
