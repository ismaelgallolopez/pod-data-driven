import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
from src.models.pinn import KinematicPINN

ckpt = torch.load('data/processed/pinn_smoother.pth', map_location='cpu', weights_only=False)
print('checkpoint type', type(ckpt))
if isinstance(ckpt, dict) and 'model_state' in ckpt:
    sd = ckpt['model_state']
else:
    sd = ckpt
print('ckpt keys count', len(sd))

model = KinematicPINN()
msd = model.state_dict()
print('model state keys count', len(msd))

mismatch = False
for k in sd:
    if k in msd:
        a = getattr(sd[k], 'shape', None)
        b = getattr(msd[k], 'shape', None)
        print(f"{k}: ckpt {a}  model {b}")
        if a != b:
            mismatch = True
    else:
        print(f"{k}: (in ckpt) NOT IN model")
        mismatch = True

for k in msd:
    if k not in sd:
        print(f"{k}: (in model) NOT IN ckpt")
        mismatch = True

print('mismatch?', mismatch)
