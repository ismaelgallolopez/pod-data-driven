"""
PINN Evaluation Script: 3D RMS Error with LVLH Decomposition

Loads a trained KinematicPINN checkpoint and evaluates it on the test split.
- Applies the same 80/20 chronological split and IQR filtering as main.py
- Runs inference on test data (normalised time -> denormalised positions in km)
- Interpolates ODCP ground truth onto SPP test timestamps
- Computes 3D RMS error and decomposes into LVLH frame components:
  * Radial: along position vector (outward)
  * Along-track: tangent to velocity
  * Cross-track: normal to orbital plane
- Outputs summary table and time-series error plot
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when running scripts
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics


def _safe_load(path):
    """Load torch file, handling various checkpoint formats and torch versions."""
    try:
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # older torch doesn't accept weights_only kwarg
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
                return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(1)


def compute_lvlh_components(r_pred, r_truth, t_indices=None):
    """
    Compute LVLH (Local Vertical Local Horizontal) frame decomposition of position errors.

    Args:
        r_pred: (N, 3) predicted positions in km (ECI)
        r_truth: (N, 3) truth positions in km (ECI)
        t_indices: optional time indices for computing velocities numerically

    Returns:
        Dictionary with 'radial', 'along_track', 'cross_track' error components in metres
    """
    r_pred_np = r_pred.numpy() if torch.is_tensor(r_pred) else r_pred
    r_truth_np = r_truth.numpy() if torch.is_tensor(r_truth) else r_truth

    # Compute velocities by numerical differentiation (finite differences)
    # Use centered differences for interior points, forward/backward for edges
    v_truth = np.zeros_like(r_truth_np)

    # Assume uniform time spacing (can refine if needed)
    dt = 1.0  # nominal, used for relative ratios only

    for i in range(len(r_truth_np)):
        if i == 0:
            v_truth[i] = (r_truth_np[1] - r_truth_np[0]) / dt
        elif i == len(r_truth_np) - 1:
            v_truth[i] = (r_truth_np[i] - r_truth_np[i-1]) / dt
        else:
            v_truth[i] = (r_truth_np[i+1] - r_truth_np[i-1]) / (2 * dt)

    # Use truth for LVLH frame definition (more accurate)
    r_mag = np.linalg.norm(r_truth_np, axis=1, keepdims=True)

    # Radial unit vector (outward)
    r_hat = r_truth_np / r_mag

    # Specific angular momentum
    h = np.cross(r_truth_np, v_truth)  # (N, 3)
    h_mag = np.linalg.norm(h, axis=1, keepdims=True)

    # Cross-track (normal) unit vector
    n_hat = h / h_mag

    # Along-track (tangential) unit vector: completes right-handed system
    # Use: t_hat = n_hat × r_hat (cross product order for right-handed)
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat, axis=1, keepdims=True)

    # Position error in ECI (convert km to m)
    delta_r = (r_pred_np - r_truth_np) * 1000.0  # km → m

    # Project onto LVLH axes (unit vectors are dimensionless)
    radial = np.sum(delta_r * r_hat, axis=1)
    along_track = np.sum(delta_r * t_hat, axis=1)
    cross_track = np.sum(delta_r * n_hat, axis=1)

    return {
        'radial': radial,
        'along_track': along_track,
        'cross_track': cross_track,
    }


def main():
    print("=" * 70)
    print("PINN Evaluation: 3D RMS Error with LVLH Decomposition")
    print("=" * 70)

    # Load SPP early — needed if checkpoint is a raw state dict
    spp = _safe_load('data/processed/noisy_spp.pt')
    print(f"Loaded SPP: {spp.shape}")

    # Try loading pinn_best.pth first (has metadata), fall back to pinn_smoother.pth
    checkpoint_path = 'data/processed/pinn_best.pth'
    if not Path(checkpoint_path).exists():
        checkpoint_path = 'data/processed/pinn_smoother.pth'
        print(f"Note: using {checkpoint_path} (best model with metadata)")
    else:
        print(f"Note: using {checkpoint_path} (best model)")

    ckpt = _safe_load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    model = KinematicPINN()

    # Handle multiple checkpoint formats
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        t_min = ckpt['t_min']
        t_scale = ckpt['t_scale']
        L_star = ckpt.get('L_star', 6378137.0)  # Default to R_earth
        T_star = ckpt.get('T_star', None)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
        t_min = ckpt.get('t_min', spp[:, 0].min().item())
        t_scale = ckpt.get('t_scale', (spp[:, 0].max() - spp[:, 0].min()).item())
        L_star = ckpt.get('L_star', 6378137.0)
        T_star = ckpt.get('T_star', None)
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # Raw state dict — recompute from training split (same as train_pinn.py)
        model.load_state_dict(ckpt)
        t = spp[:, 0]
        n = len(t)
        t_train = t[:int(n*0.8)]
        t_min = t_train.min().item()
        t_scale = (t_train.max() - t_train.min()).item()
        L_star = 6378137.0
        T_star = None
    else:
        print("Unknown checkpoint format. Keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
        sys.exit(1)

    print(f"Normalization: t_min={t_min:.1f}, t_scale={t_scale:.1f}, L_star={L_star:.0f}")

    model.eval()
    physics = OrbitPhysics()

    # Load truth
    try:
        odcp = _safe_load('data/processed/truth_odcp.pt')
        print(f"Loaded ODCP truth: {odcp.shape}")
    except SystemExit:
        odcp = None
        print("WARNING: ODCP truth not found")

    # ── IQR-based outlier filtering (same as main.py) ──────────────────────────
    t = spp[:, 0]
    r = spp[:, 1:4]  # km

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

    # ── Chronological 80/20 split ────────────────────────────────────────────
    num_samples = len(t)
    train_idx = int(num_samples * 0.8)

    t_train = t[:train_idx]
    r_train = r[:train_idx]
    t_test = t[train_idx:]
    r_test_km = r[train_idx:]

    print(f"\nChronological split (after filtering):")
    print(f"  Total samples:     {num_samples}")
    print(f"  Training samples:  {len(t_train)} ({100*len(t_train)/num_samples:.1f}%)")
    print(f"  Test samples:      {len(t_test)} ({100*len(t_test)/num_samples:.1f}%)")
    print(f"  Test time span:    {t_test[0].item():.1f} to {t_test[-1].item():.1f} s ({(t_test[-1]-t_test[0]).item()/86400:.2f} days)")
    print(f"\nModel training range (from checkpoint):")
    print(f"  Time range:        {t_min:.1f} to {t_min+t_scale:.1f} s ({t_scale/3600:.2f} hours)")
    if t_test[0].item() > t_min + t_scale:
        print(f"  WARNING: Test set starts {(t_test[0].item()-(t_min+t_scale))/86400:.1f} days AFTER training range")
        print(f"  This evaluation is EXTRAPOLATION far beyond training domain.")

    # ── Inference on test set ────────────────────────────────────────────────
    with torch.no_grad():
        t_norm = (t_test - t_min) / t_scale
        r_pred_nd = model(t_norm.float())

    r_pred_km = r_pred_nd * L_star / 1000.0  # non-dim → m → km

    # ── Inference on training split for fit quality analysis ────────────────
    with torch.no_grad():
        t_train_norm = (t_train - t_min) / t_scale
        r_pred_train_nd = model(t_train_norm.float())

    r_pred_train_km = r_pred_train_nd * L_star / 1000.0

    # ── Sanity check: raw predictions vs inputs ──────────────────────────────
    print(f"\n{'-'*70}")
    print(f"Sanity Check: Raw Predictions vs Data")
    print(f"{'-'*70}")

    # Magnitude statistics
    r_pred_mag = np.linalg.norm(r_pred_km.numpy(), axis=1)
    r_spp_mag = np.linalg.norm(r_test_km.numpy(), axis=1)

    print(f"Predicted |r| (km): mean={r_pred_mag.mean():.1f}, std={r_pred_mag.std():.1f}")
    print(f"  range: {r_pred_mag.min():.1f} to {r_pred_mag.max():.1f}")
    print(f"SPP |r| (km):       mean={r_spp_mag.mean():.1f}, std={r_spp_mag.std():.1f}")
    print(f"  range: {r_spp_mag.min():.1f} to {r_spp_mag.max():.1f}")

    # Plot: X, Y, Z overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    t_h = t_test.numpy() / 3600.0  # hours

    for i, (ax, coord) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
        ax.plot(t_h, r_pred_km[:, i].numpy(), '-', lw=1, alpha=0.7, label='PINN prediction', color='blue')
        ax.plot(t_h, r_test_km[:, i].numpy(), '.', ms=1, alpha=0.3, label='SPP input', color='green')

        # Overlay ODCP truth if available
        if odcp is not None:
            t_np = t_test.numpy()
            t_tru = odcp[:, 0].numpy()
            r_tru_km = odcp[:, 1:4].numpy()
            mask = (t_np >= t_tru[0]) & (t_np <= t_tru[-1])

            if mask.sum() > 0:
                interp_func = interp1d(t_tru, r_tru_km[:, i], kind='linear')
                r_tru_interp = interp_func(t_np[mask])
                ax.plot(t_h[mask], r_tru_interp, '-', lw=0.8, alpha=0.6, label='ODCP truth', color='red')

        ax.set_ylabel(f'{coord} [km]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time [hours]')
    fig.suptitle('PINN Prediction Sanity Check: ECI Position Components', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/processed/eval_pinn_sanity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved sanity check plot: data/processed/eval_pinn_sanity.png")

    # ── Zoomed 3-hour window from middle of training set ──────────────────────
    # Find indices around hour 300-303 of the training set
    t_train_h = t_train.numpy() / 3600.0
    window_start_h = 300.0
    window_end_h = 303.0
    zoom_mask = (t_train_h >= window_start_h) & (t_train_h <= window_end_h)

    if zoom_mask.sum() > 0:
        print(f"\nZoomed 3-hour window ({window_start_h:.0f}-{window_end_h:.0f} h): {zoom_mask.sum()} epochs")

        t_zoom = t_train[zoom_mask]
        r_pred_zoom = r_pred_train_km[zoom_mask].numpy()
        r_train_zoom = r_train[zoom_mask].numpy()
        t_zoom_h = t_zoom.numpy() / 3600.0

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        for i, (ax, coord) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
            # PINN prediction
            ax.plot(t_zoom_h, r_pred_zoom[:, i], 'b-', lw=1.5, alpha=0.7, label='PINN (smooth)')
            # SPP data points
            ax.plot(t_zoom_h, r_train_zoom[:, i], 'g.', ms=4, alpha=0.6, label='SPP (noisy input)')

            # ODCP truth if available
            if odcp is not None:
                t_tru = odcp[:, 0].numpy()
                r_tru_km = odcp[:, 1:4].numpy()
                mask_zoom = (t_zoom >= t_tru[0]) & (t_zoom <= t_tru[-1])

                if mask_zoom.sum() > 0:
                    interp_func = interp1d(t_tru, r_tru_km[:, i], kind='linear')
                    r_tru_zoom = interp_func(t_zoom[mask_zoom].numpy())
                    ax.plot(t_zoom[mask_zoom].numpy() / 3600.0, r_tru_zoom, 'r-', lw=1, alpha=0.7, label='ODCP truth')

            ax.set_ylabel(f'{coord} [km]')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right')

        axes[-1].set_xlabel('Time [hours]')
        fig.suptitle(f'PINN Fit Quality: 3-hour window (hours {window_start_h:.0f}–{window_end_h:.0f})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/processed/eval_pinn_zoom.png', dpi=150, bbox_inches='tight')
        print(f"Saved zoomed window plot: data/processed/eval_pinn_zoom.png")
    else:
        print(f"Warning: No data in requested zoom window ({window_start_h}–{window_end_h} hours)")

    # ── Error on training vs test splits ────────────────────────────────────
    # Training split error (using pre-computed r_pred_train_km)
    diff_train_km = r_pred_train_km - r_train
    diff_train_m = diff_train_km * 1000.0
    rms_train = np.sqrt(np.mean(np.sum(diff_train_m.numpy()**2, axis=1)))

    # Test split error
    diff_spp_km = r_pred_km - r_test_km
    diff_spp_m = diff_spp_km * 1000.0
    rms_test = np.sqrt(np.mean(np.sum(diff_spp_m.numpy()**2, axis=1)))

    print(f"\n{'-'*70}")
    print(f"PINN Reconstruction Error: Training vs Test Split")
    print(f"{'-'*70}")
    print(f"Training split (first 80%, {len(t_train)} epochs):")
    print(f"  3D RMS error: {rms_train:.1f} m")
    print(f"Test split (last 20%, {len(t_test)} epochs):")
    print(f"  3D RMS error: {rms_test:.1f} m")
    print(f"Degradation (test/train ratio): {rms_test/rms_train:.2f}x")

    for i, coord in enumerate(['X', 'Y', 'Z']):
        err_i = diff_spp_m[:, i].numpy()
        print(f"  {coord}: RMS={np.sqrt(np.mean(err_i**2)):.1f} m, "
              f"mean={np.mean(err_i):.1f} m, std={np.std(err_i):.1f} m")

    # ── Comparison vs ODCP truth (if available) ──────────────────────────────
    if odcp is not None:
        t_np = t_test.numpy()
        t_tru = odcp[:, 0].numpy()
        r_tru_km = odcp[:, 1:4].numpy()

        # Mask: only evaluate where times overlap
        mask = (t_np >= t_tru[0]) & (t_np <= t_tru[-1])
        if mask.sum() == 0:
            print("\nNo overlapping times with ODCP truth.")
        else:
            print(f"\n{'-'*70}")
            print(f"PINN vs ODCP (ground truth)")
            print(f"{'-'*70}")

            # Interpolate truth onto test timestamps
            interp_x = interp1d(t_tru, r_tru_km[:, 0], kind='linear')
            interp_y = interp1d(t_tru, r_tru_km[:, 1], kind='linear')
            interp_z = interp1d(t_tru, r_tru_km[:, 2], kind='linear')

            r_tru_interp_km = np.column_stack([
                interp_x(t_np[mask]),
                interp_y(t_np[mask]),
                interp_z(t_np[mask]),
            ])

            r_pred_test_km = r_pred_km[mask].numpy()

            # 3D RMS error
            diff_odcp_km = r_pred_test_km - r_tru_interp_km
            diff_odcp_m = diff_odcp_km * 1000.0
            rms_3d = np.sqrt(np.mean(np.sum(diff_odcp_m**2, axis=1)))

            print(f"3D RMS error: {rms_3d:.1f} m")
            print(f"Number of overlapping test epochs: {mask.sum()}")

            # Component-wise RMS
            for i, coord in enumerate(['X', 'Y', 'Z']):
                err_i = diff_odcp_m[:, i]
                print(f"  {coord}: RMS={np.sqrt(np.mean(err_i**2)):.1f} m, "
                      f"mean={np.mean(err_i):.1f} m, std={np.std(err_i):.1f} m")

            # ── LVLH decomposition ───────────────────────────────────────────
            lvlh_components = compute_lvlh_components(
                r_pred_test_km,
                r_tru_interp_km
            )

            print(f"\n{'-'*70}")
            print(f"LVLH Frame Decomposition (w.r.t. truth)")
            print(f"{'-'*70}")

            for comp_name in ['radial', 'along_track', 'cross_track']:
                err_comp = lvlh_components[comp_name]
                rms_comp = np.sqrt(np.mean(err_comp**2))
                print(f"{comp_name:12s}: RMS={rms_comp:7.1f} m, "
                      f"mean={np.mean(err_comp):7.1f} m, std={np.std(err_comp):7.1f} m")

            # Compute magnitude (for plotting)
            err_mag = np.sqrt(np.sum(diff_odcp_m**2, axis=1))

            # ── Plot time-series error ───────────────────────────────────────
            fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

            t_h = t_np[mask] / 3600.0  # hours

            # Plot each LVLH component
            for ax, comp_name in zip(axes[:3], ['radial', 'along_track', 'cross_track']):
                err = lvlh_components[comp_name]
                ax.plot(t_h, err, '.', ms=2, alpha=0.6, label=comp_name.replace('_', ' ').title())
                ax.axhline(0, color='k', lw=0.5, alpha=0.3)
                ax.set_ylabel(f'{comp_name} Error [m]')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')

            axes[-1].set_xlabel('Time [hours]')
            fig.suptitle('PINN Position Error: LVLH Decomposition', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('data/processed/eval_pinn.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved plot: data/processed/eval_pinn.png")

            # ── Summary table ────────────────────────────────────────────────
            print(f"\n{'-'*70}")
            print(f"Summary Table")
            print(f"{'-'*70}")
            print(f"{'Metric':<25} {'Value':>15} {'Unit':>20}")
            print(f"{'-'*70}")
            print(f"{'3D RMS Error':<25} {rms_3d:>15.1f} {'m':>20}")
            print(f"{'Radial RMS':<25} {np.sqrt(np.mean(lvlh_components['radial']**2)):>15.1f} {'m':>20}")
            print(f"{'Along-track RMS':<25} {np.sqrt(np.mean(lvlh_components['along_track']**2)):>15.1f} {'m':>20}")
            print(f"{'Cross-track RMS':<25} {np.sqrt(np.mean(lvlh_components['cross_track']**2)):>15.1f} {'m':>20}")
            print(f"{'Test set size':<25} {mask.sum():>15} {'epochs':>20}")
            print(f"{'-'*70}\n")

    else:
        print("\nNo ODCP truth file found; only SPP comparison available.")


if __name__ == "__main__":
    main()
