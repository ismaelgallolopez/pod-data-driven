"""Train a smoother variant and report full-arc + interior metrics (cubic truth).

Usage: python scripts/tune_edge.py <prior_alpha> <phys_lam> [recover 0|1] [edge_phys_lam]
Writes to data/processed/_tune/ so it never clobbers the production checkpoint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scipy.interpolate import CubicSpline
from src.training.train_pinn import train_pinn
from src.models.pinn import KinematicPINN

def load(p):
    try: return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(p, map_location='cpu')

prior_alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
phys_lam    = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
recover     = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
edge_phys   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

spp = load('data/processed/noisy_spp.pt').numpy().astype(np.float64)
t, r = spp[:, 0], spp[:, 1:4]
out = 'data/processed/_tune'
kw = dict(checkpoint_dir=out, prior_alpha=prior_alpha, phys_lam=phys_lam,
          recover_timetags=recover, verbose=True)
if edge_phys:
    kw['stage2_phys_lam'] = edge_phys
train_pinn(t, r, **kw)

odcp = load('data/processed/truth_odcp.pt').numpy().astype(np.float64)
ck = load(out + '/pinn_best.pth'); model = KinematicPINN.from_checkpoint(ck)
t_o, r_o = odcp[:, 0], odcp[:, 1:4]
o = np.argsort(t_o); t_o, r_o = t_o[o], r_o[o]
uq = np.concatenate([[True], np.diff(t_o) > 0]); t_o, r_o = t_o[uq], r_o[uq]
cs = CubicSpline(t_o, r_o, axis=0)
m = (t >= t_o[0] + 5) & (t <= t_o[-1] - 5); ts = t[m]
with torch.no_grad():
    tn = (torch.from_numpy(ts).double() - float(ck['t_min'])) / float(ck['t_scale'])
    rp = model(tn).numpy() * 6378137.0 / 1000.0
err = np.linalg.norm((rp - cs(ts)) * 1000.0, axis=1)
tmax = ts.max()
intr = (ts >= 3 * 3600) & (ts <= tmax - 3 * 3600)
print(f"\n=== prior_alpha={prior_alpha} phys_lam={phys_lam} recover={recover} edge_phys={edge_phys} ===")
print(f"FULL ARC : RMS={np.sqrt(np.mean(err**2)):7.1f}m  median={np.median(err):5.1f}  "
      f"p99={np.percentile(err,99):6.1f}  max={err.max():7.0f}")
print(f"EXCL 3h  : RMS={np.sqrt(np.mean(err[intr]**2)):7.1f}m  median={np.median(err[intr]):5.1f}  "
      f"p99={np.percentile(err[intr],99):6.1f}  max={err[intr].max():7.0f}")
