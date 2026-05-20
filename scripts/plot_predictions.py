import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
        raise


def main():
    spp = _safe_load('data/processed/noisy_spp.pt')
    ckpt = _safe_load('data/processed/pinn_smoother.pth')

    model = KinematicPINN()

    # load model params (handle raw state dicts)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        t_min = ckpt.get('t_min', None)
        t_scale = ckpt.get('t_scale', None)
        L_star = ckpt.get('L_star', 6378137.0)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
        t_min = ckpt.get('t_min', None)
        t_scale = ckpt.get('t_scale', None)
        L_star = ckpt.get('L_star', 6378137.0)
    elif isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
        except:
            # Try loading only tensor values
            tensor_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            model.load_state_dict(tensor_dict)
        t = spp[:, 0]
        n = len(t)
        t_train = t[:int(n * 0.8)]
        t_min = t_train.min().item()
        t_scale = (t_train.max() - t_train.min()).item()
        L_star = 6378137.0
        print(f"Recomputed normalization from training split: t_min={t_min}, t_scale={t_scale}")
    else:
        raise RuntimeError('Unknown checkpoint format')

    model.eval()

    # test split: last 20%
    n = len(spp)
    t_test = spp[int(n*0.8):, 0]
    r_test_km = spp[int(n*0.8):, 1:4]

    with torch.no_grad():
        t_norm = (t_test - t_min) / t_scale
        r_pred_nd = model(t_norm.float())
    r_pred_km = r_pred_nd.numpy() * L_star / 1000.0

    # Plot per-axis time series for first N samples of test
    N = min(1000, len(t_test))
    times = t_test[:N].numpy()

    out_dir = Path('logs')
    out_dir.mkdir(exist_ok=True)

    for i, coord in enumerate(['x','y','z']):
        plt.figure(figsize=(10,3))
        plt.plot(times, r_test_km[:N, i], label='SPP', lw=1)
        plt.plot(times, r_pred_km[:N, i], label='PINN', lw=1)
        plt.xlabel('time (s)')
        plt.ylabel(f'{coord} (km)')
        plt.title(f'PINN vs SPP — {coord}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'pred_vs_spp_{coord}.png')
        plt.close()

    # 3D scatter for the same window
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(r_test_km[:N,0], r_test_km[:N,1], r_test_km[:N,2], s=4, label='SPP')
        ax.scatter(r_pred_km[:N,0], r_pred_km[:N,1], r_pred_km[:N,2], s=4, label='PINN')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        ax.legend()
        plt.savefig(out_dir / 'pred_vs_spp_3d.png')
        plt.close()
    except Exception as e:
        print('3D plot failed:', e)

    print(f'Plots saved to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
