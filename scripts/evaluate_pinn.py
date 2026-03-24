import torch
import numpy as np
from scipy.interpolate import interp1d
import sys

from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics

# Safe load
def _safe_load(path):
    try:
        return torch.load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(1)

ckpt = _safe_load('data/processed/pinn_smoother.pth')
model = KinematicPINN()
model.load_state_dict(ckpt['model_state'])
model.eval()

physics = OrbitPhysics()

# Load data
spp = _safe_load('data/processed/noisy_spp.pt')
try:
    odcp = _safe_load('data/processed/truth_odcp.pt')
except SystemExit:
    odcp = None

# Test split: last 20%
n = len(spp)
t_test = spp[int(n*0.8):, 0]
r_test_km = spp[int(n*0.8):, 1:4]

# Normalize with stored params
t_min = ckpt['t_min']
t_scale = ckpt['t_scale']
L_star = ckpt['L_star']

# Evaluate
with torch.no_grad():
    t_norm = (t_test - t_min) / t_scale
    r_pred_nd = model(t_norm.float())

r_pred_km = r_pred_nd * L_star / 1000.0

# vs SPP
diff_spp = (r_pred_km - r_test_km) * 1000.0  # m
rms_spp = np.sqrt(np.mean(np.sum(diff_spp.numpy()**2, axis=1)))
print(f"PINN vs SPP  RMS: {rms_spp:.1f} m")

# vs ODCP truth
if odcp is not None:
    t_np = t_test.numpy()
    t_tru = odcp[:, 0].numpy()
    r_tru = odcp[:, 1:4].numpy()

    mask = (t_np >= t_tru[0]) & (t_np <= t_tru[-1])
    if mask.sum() == 0:
        print("No overlapping times with ODCP truth to evaluate.")
        sys.exit(0)

    for i, coord in enumerate(['x','y','z']):
        interp = interp1d(t_tru, r_tru[:, i], bounds_error=False, fill_value='extrapolate')
        r_tru_i = interp(t_np[mask])
        diff_i = (r_pred_km[mask, i].numpy() - r_tru_i) * 1000.0
        print(f"  {coord}: {np.sqrt(np.mean(diff_i**2)):.1f} m RMS vs ODCP")

    rms_3d = np.sqrt(np.mean(np.sum(((r_pred_km[mask].numpy() - np.column_stack([
        interp1d(t_tru, r_tru[:,i])(t_np[mask]) for i in range(3)
    ])) * 1000.0)**2, axis=1)))
    print(f"PINN vs ODCP 3D RMS: {rms_3d:.1f} m")
else:
    print("No ODCP truth file found; only SPP comparison printed.")
