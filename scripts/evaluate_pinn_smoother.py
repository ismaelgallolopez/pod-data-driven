"""
PINN Smoother Evaluation: Smoothing Quality and Gap-Bridging Capability

For a PINN used as a smoother:
1. Evaluate smoothing quality: RMS of predictions vs noisy input on full dataset
2. Test gap-bridging: mask a 30-minute window during training, evaluate prediction
   through the gap against ODCP ground truth
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics
from src.training.train_pinn import train_pinn


def _safe_load(path):
    """Load torch file, handling various checkpoint formats and torch versions."""
    try:
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
                return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(1)


def main():
    print("=" * 70)
    print("PINN Smoother Evaluation: Full Dataset Analysis")
    print("=" * 70)

    # Load data
    spp = _safe_load('data/processed/noisy_spp.pt')
    print(f"Loaded SPP: {spp.shape}")

    try:
        odcp = _safe_load('data/processed/truth_odcp.pt')
        print(f"Loaded ODCP truth: {odcp.shape}")
    except SystemExit:
        odcp = None
        print("WARNING: ODCP truth not found")

    t = spp[:, 0]
    r = spp[:, 1:4]  # km

    # IQR filtering (same as before)
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

    # Load trained model
    checkpoint_path = 'data/processed/pinn_best.pth'
    if not Path(checkpoint_path).exists():
        checkpoint_path = 'data/processed/pinn_smoother.pth'

    ckpt = _safe_load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    model = KinematicPINN()

    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        t_min = ckpt['t_min']
        t_scale = ckpt['t_scale']
        L_star = ckpt.get('L_star', 6378137.0)
    else:
        print("Unknown checkpoint format")
        sys.exit(1)

    print(f"Normalization: t_min={t_min:.1f}, t_scale={t_scale:.1f}, L_star={L_star:.0f}")
    model.eval()

    # ── Evaluate smoothing quality on full dataset ──────────────────────────
    print(f"\n{'-'*70}")
    print(f"Smoothing Quality: PINN vs SPP (full {len(t)} epochs)")
    print(f"{'-'*70}")

    with torch.no_grad():
        t_norm = (t - t_min) / t_scale
        r_pred_nd = model(t_norm.float())

    r_pred_km = r_pred_nd * L_star / 1000.0

    diff_km = r_pred_km - r
    diff_m = diff_km * 1000.0
    rms_full = np.sqrt(np.mean(np.sum(diff_m.numpy()**2, axis=1)))

    print(f"Full dataset RMS error: {rms_full:.1f} m")
    for i, coord in enumerate(['X', 'Y', 'Z']):
        err_i = diff_m[:, i].numpy()
        print(f"  {coord}: RMS={np.sqrt(np.mean(err_i**2)):.1f} m, "
              f"mean={np.mean(err_i):.1f} m, std={np.std(err_i):.1f} m")

    # Magnitude statistics
    r_pred_mag = np.linalg.norm(r_pred_km.numpy(), axis=1)
    r_spp_mag = np.linalg.norm(r.numpy(), axis=1)
    print(f"\nPredicted |r|: mean={r_pred_mag.mean():.1f} km, std={r_pred_mag.std():.1f} km")
    print(f"SPP |r|:       mean={r_spp_mag.mean():.1f} km, std={r_spp_mag.std():.1f} km")

    # ── Gap-bridging test: mask 30-minute window from middle of dataset ─────
    if odcp is not None:
        print(f"\n{'-'*70}")
        print(f"Gap-Bridging Test: Mask 30-minute window and evaluate prediction")
        print(f"{'-'*70}")

        # Select 30-minute window (1800 seconds) from middle of dataset
        t_mid = (t.min() + t.max()) / 2.0
        gap_start = t_mid - 900.0  # 15 min before
        gap_end = t_mid + 900.0    # 15 min after
        gap_mask = (t >= gap_start) & (t <= gap_end)
        keep_mask = ~gap_mask

        print(f"Gap window: {gap_start:.0f} to {gap_end:.0f} s ({gap_mask.sum()} epochs)")
        print(f"Training without gap: {keep_mask.sum()} epochs")

        # Train model without gap data
        t_train_gap = t[keep_mask]
        r_train_gap = r[keep_mask]

        print("\nTraining PINN on data with masked gap (this will take ~10 min)...")
        model_gap = train_pinn(t_train_gap, r_train_gap, epochs=2000)

        # Evaluate on the masked gap
        t_gap = t[gap_mask]
        with torch.no_grad():
            t_gap_norm = (t_gap - t_min) / t_scale
            r_gap_pred_nd = model_gap(t_gap_norm.float())

        r_gap_pred_km = r_gap_pred_nd * L_star / 1000.0

        # Compare against ODCP truth
        t_tru = odcp[:, 0].numpy()
        r_tru_km = odcp[:, 1:4].numpy()

        mask_gap = (t_gap.numpy() >= t_tru[0]) & (t_gap.numpy() <= t_tru[-1])

        if mask_gap.sum() > 0:
            # Interpolate truth onto gap timestamps
            interp_x = interp1d(t_tru, r_tru_km[:, 0], kind='linear')
            interp_y = interp1d(t_tru, r_tru_km[:, 1], kind='linear')
            interp_z = interp1d(t_tru, r_tru_km[:, 2], kind='linear')

            r_tru_interp = np.column_stack([
                interp_x(t_gap[mask_gap].numpy()),
                interp_y(t_gap[mask_gap].numpy()),
                interp_z(t_gap[mask_gap].numpy()),
            ])

            diff_gap = (r_gap_pred_km[mask_gap].numpy() - r_tru_interp) * 1000.0
            rms_gap = np.sqrt(np.mean(np.sum(diff_gap**2, axis=1)))

            print(f"\nGap-bridging RMS error vs ODCP truth: {rms_gap:.1f} m")
            for i, coord in enumerate(['X', 'Y', 'Z']):
                err_i = diff_gap[:, i]
                print(f"  {coord}: RMS={np.sqrt(np.mean(err_i**2)):.1f} m")

            # Plot the gap-bridging result
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            t_gap_h = t_gap[mask_gap].numpy() / 3600.0

            for i, (ax, coord) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
                ax.plot(t_gap_h, r_gap_pred_km[mask_gap, i].numpy(), 'b-', lw=2, label='PINN (trained without gap)')
                ax.plot(t_gap_h, r_tru_interp[:, i], 'r--', lw=1.5, label='ODCP truth')
                ax.set_ylabel(f'{coord} [km]')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()

            axes[-1].set_xlabel('Time [hours]')
            fig.suptitle('Gap-Bridging: PINN Prediction Through 30-minute Data Gap', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('data/processed/eval_pinn_gap_bridging.png', dpi=150, bbox_inches='tight')
            print(f"Saved gap-bridging plot: data/processed/eval_pinn_gap_bridging.png")
        else:
            print("Gap window does not overlap with ODCP truth")
    else:
        print("\nNo ODCP truth available for gap-bridging test")

    print(f"\n{'-'*70}")
    print(f"Summary: Smoother is acceptable if full RMS < 1 km")
    print(f"Current RMS: {rms_full:.1f} m ({'PASS' if rms_full < 1000 else 'FAIL'})")
    print(f"{'-'*70}\n")


if __name__ == "__main__":
    main()
