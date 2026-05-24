import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from scipy.interpolate import interp1d
import warnings

from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics

def _safe_load(path):
    try:
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
                return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

def compute_lvlh_components(r_pred, r_truth, t_indices=None):
    r_pred_np = r_pred.numpy() if torch.is_tensor(r_pred) else r_pred
    r_truth_np = r_truth.numpy() if torch.is_tensor(r_truth) else r_truth

    v_truth = np.zeros_like(r_truth_np)
    dt = 1.0

    for i in range(len(r_truth_np)):
        if i == 0:
            v_truth[i] = (r_truth_np[1] - r_truth_np[0]) / dt
        elif i == len(r_truth_np) - 1:
            v_truth[i] = (r_truth_np[i] - r_truth_np[i-1]) / dt
        else:
            v_truth[i] = (r_truth_np[i+1] - r_truth_np[i-1]) / (2 * dt)

    r_mag = np.linalg.norm(r_truth_np, axis=1, keepdims=True)
    r_hat = r_truth_np / r_mag
    h = np.cross(r_truth_np, v_truth)
    h_mag = np.linalg.norm(h, axis=1, keepdims=True)
    n_hat = h / h_mag
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat, axis=1, keepdims=True)

    delta_r = (r_pred_np - r_truth_np) * 1000.0

    radial = np.sum(delta_r * r_hat, axis=1)
    along_track = np.sum(delta_r * t_hat, axis=1)
    cross_track = np.sum(delta_r * n_hat, axis=1)

    return {
        'radial': radial,
        'along_track': along_track,
        'cross_track': cross_track,
    }

checkpoint_path = 'logs/ckpt_finetuned.pth'
if not Path(checkpoint_path).exists():
    print(f"Finetuned checkpoint not found at {checkpoint_path}")
    sys.exit(1)

print(f"Testing finetuned checkpoint: {checkpoint_path}")
spp = _safe_load('data/processed/noisy_spp.pt')
print(f"Loaded SPP: {spp.shape}")

ckpt = _safe_load(checkpoint_path)
if ckpt is None:
    sys.exit(1)

model = KinematicPINN()

if isinstance(ckpt, dict) and 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
    t_min = ckpt['t_min']
    t_scale = ckpt['t_scale']
    L_star = ckpt.get('L_star', 6378137.0)
elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    model.load_state_dict(ckpt['state_dict'])
    t_min = ckpt.get('t_min', spp[:, 0].min().item())
    t_scale = ckpt.get('t_scale', (spp[:, 0].max() - spp[:, 0].min()).item())
    L_star = ckpt.get('L_star', 6378137.0)
elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    model.load_state_dict(ckpt)
    t = spp[:, 0]
    n = len(t)
    t_train = t[:int(n*0.8)]
    t_min = t_train.min().item()
    t_scale = (t_train.max() - t_train.min()).item()
    L_star = 6378137.0
else:
    print("Unknown checkpoint format")
    sys.exit(1)

print(f"Loaded model successfully")

t = spp[:, 0]
r = spp[:, 1:4]

r_mag = torch.norm(r, dim=1)
q25 = r_mag.quantile(0.25)
q75 = r_mag.quantile(0.75)
iqr = q75 - q25
lower = q25 - 10.0 * iqr
upper = q75 + 10.0 * iqr

mask = (r_mag >= lower) & (r_mag <= upper)
n_removed = (~mask).sum().item()
if n_removed > 0:
    t = t[mask]
    r = r[mask]

num_samples = len(t)
train_idx = int(num_samples * 0.8)

t_train = t[:train_idx]
r_train = r[:train_idx]
t_test = t[train_idx:]
r_test_km = r[train_idx:]

model.eval()

try:
    odcp = _safe_load('data/processed/truth_odcp.pt')
    print(f"Loaded ODCP truth: {odcp.shape}")
except:
    odcp = None
    print("ODCP truth not found")

with torch.no_grad():
    t_norm = (t_test - t_min) / t_scale
    r_pred_nd = model(t_norm.float())

r_pred_km = r_pred_nd * L_star / 1000.0

with torch.no_grad():
    t_train_norm = (t_train - t_min) / t_scale
    r_pred_train_nd = model(t_train_norm.float())

r_pred_train_km = r_pred_train_nd * L_star / 1000.0

diff_train_km = r_pred_train_km - r_train
diff_train_m = diff_train_km * 1000.0
rms_train = np.sqrt(np.mean(np.sum(diff_train_m.numpy()**2, axis=1)))

diff_spp_km = r_pred_km - r_test_km
diff_spp_m = diff_spp_km * 1000.0
rms_test = np.sqrt(np.mean(np.sum(diff_spp_m.numpy()**2, axis=1)))

print(f"\n{'-'*70}")
print(f"Finetuned Model - Reconstruction Error")
print(f"{'-'*70}")
print(f"Training split: 3D RMS error = {rms_train:.1f} m")
print(f"Test split:     3D RMS error = {rms_test:.1f} m")
print(f"Degradation:    {rms_test/rms_train:.2f}x")

if odcp is not None:
    t_np = t_test.numpy()
    t_tru = odcp[:, 0].numpy()
    r_tru_km = odcp[:, 1:4].numpy()

    mask = (t_np >= t_tru[0]) & (t_np <= t_tru[-1])
    if mask.sum() > 0:
        print(f"\n{'-'*70}")
        print(f"Finetuned Model vs ODCP Truth")
        print(f"{'-'*70}")

        interp_x = interp1d(t_tru, r_tru_km[:, 0], kind='linear')
        interp_y = interp1d(t_tru, r_tru_km[:, 1], kind='linear')
        interp_z = interp1d(t_tru, r_tru_km[:, 2], kind='linear')

        r_tru_interp_km = np.column_stack([
            interp_x(t_np[mask]),
            interp_y(t_np[mask]),
            interp_z(t_np[mask]),
        ])

        r_pred_test_km = r_pred_km[mask].numpy()

        diff_odcp_km = r_pred_test_km - r_tru_interp_km
        diff_odcp_m = diff_odcp_km * 1000.0
        rms_3d = np.sqrt(np.mean(np.sum(diff_odcp_m**2, axis=1)))

        print(f"3D RMS error: {rms_3d:.1f} m")

        lvlh_components = compute_lvlh_components(r_pred_test_km, r_tru_interp_km)

        print(f"\n{'-'*70}")
        print(f"LVLH Decomposition")
        print(f"{'-'*70}")
        for comp_name in ['radial', 'along_track', 'cross_track']:
            err_comp = lvlh_components[comp_name]
            rms_comp = np.sqrt(np.mean(err_comp**2))
            print(f"{comp_name:12s}: RMS={rms_comp:7.1f} m")

        print(f"\n{'-'*70}")
        print(f"Summary")
        print(f"{'-'*70}")
        print(f"{'3D RMS Error':<25} {rms_3d:>15.1f} m")
        print(f"{'Radial RMS':<25} {np.sqrt(np.mean(lvlh_components['radial']**2)):>15.1f} m")
        print(f"{'Along-track RMS':<25} {np.sqrt(np.mean(lvlh_components['along_track']**2)):>15.1f} m")
        print(f"{'Cross-track RMS':<25} {np.sqrt(np.mean(lvlh_components['cross_track']**2)):>15.1f} m")

