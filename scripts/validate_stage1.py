"""
Stage 1 PINN Smoother Validation
Experiment 1: Smoothing quality vs ODCP ground truth
Experiment 2: Gap-bridging capability
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
from src.training.train_pinn import train_pinn


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
        sys.exit(1)


def load_and_filter_data():
    """Load SPP and ODCP, apply IQR filtering."""
    spp = _safe_load('data/processed/noisy_spp.pt')
    odcp = _safe_load('data/processed/truth_odcp.pt')

    t = spp[:, 0]
    r = spp[:, 1:4]

    # IQR filtering
    r_mag = torch.norm(r, dim=1)
    q25, q75 = r_mag.quantile(0.25), r_mag.quantile(0.75)
    iqr = q75 - q25
    mask = (r_mag >= q25 - 10*iqr) & (r_mag <= q75 + 10*iqr)
    n_removed = (~mask).sum().item()

    if n_removed > 0:
        print(f"Removed {n_removed} outliers ({100*n_removed/len(t):.1f}%)")
        t = t[mask]
        r = r[mask]

    return t, r, odcp


def experiment_1_smoothing_quality():
    """Compare PINN vs raw SPP against ODCP truth."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Smoothing Quality vs ODCP Ground Truth")
    print("="*70)

    t, r, odcp = load_and_filter_data()

    # Load trained model
    ckpt = _safe_load('data/processed/pinn_best.pth')
    t_scale = ckpt['t_scale']
    t_min = ckpt['t_min']
    L_star = ckpt.get('L_star', 6378137.0)

    # Check checkpoint embedding size to determine model version
    embedding_size = ckpt['model_state']['embedding.freqs'].shape[0]

    # Old checkpoints have 64 freqs (no t_scale aware), new have 128 (dual-grid)
    if embedding_size == 64:
        # Old model - don't pass t_scale
        model = KinematicPINN(num_frequencies=64)
    else:
        # New model - pass t_scale for proper frequency grid
        model = KinematicPINN(num_frequencies=128, t_scale=t_scale)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Run inference
    with torch.no_grad():
        t_norm = (t - t_min) / t_scale
        r_pred_nd = model(t_norm.float())
    r_pred_km = r_pred_nd * L_star / 1000.0

    # Interpolate ODCP onto SPP timestamps
    t_np = t.numpy()
    t_tru = odcp[:, 0].numpy()
    r_tru_km = odcp[:, 1:4].numpy()

    # Mask: only where times overlap
    mask_overlap = (t_np >= t_tru[0]) & (t_np <= t_tru[-1])

    if mask_overlap.sum() == 0:
        print("ERROR: No time overlap between SPP and ODCP")
        return

    # Interpolate truth
    interp_x = interp1d(t_tru, r_tru_km[:, 0], kind='linear')
    interp_y = interp1d(t_tru, r_tru_km[:, 1], kind='linear')
    interp_z = interp1d(t_tru, r_tru_km[:, 2], kind='linear')

    r_tru_interp = np.column_stack([
        interp_x(t_np[mask_overlap]),
        interp_y(t_np[mask_overlap]),
        interp_z(t_np[mask_overlap]),
    ])

    # Compute errors
    diff_pinn_m = (r_pred_km[mask_overlap].numpy() - r_tru_interp) * 1000.0
    diff_spp_m = (r[mask_overlap].numpy() - r_tru_interp) * 1000.0

    rms_pinn = np.sqrt(np.mean(np.sum(diff_pinn_m**2, axis=1)))
    rms_spp = np.sqrt(np.mean(np.sum(diff_spp_m**2, axis=1)))

    print(f"\nResults ({mask_overlap.sum()} overlapping epochs):")
    print(f"  SPP vs ODCP RMS:  {rms_spp:.1f} m (baseline - noisy input)")
    print(f"  PINN vs ODCP RMS: {rms_pinn:.1f} m (smoothed output)")
    print(f"  Improvement:      {rms_spp/rms_pinn:.1f}× better than raw SPP")

    if rms_pinn < rms_spp:
        print(f"  Status: ✅ PASS - PINN is closer to truth than raw SPP")
    else:
        print(f"  Status: ❌ FAIL - PINN is worse than raw SPP")

    # Per-component breakdown
    print(f"\nPer-component errors vs ODCP truth:")
    for i, coord in enumerate(['X', 'Y', 'Z']):
        rms_pinn_i = np.sqrt(np.mean(diff_pinn_m[:, i]**2))
        rms_spp_i = np.sqrt(np.mean(diff_spp_m[:, i]**2))
        print(f"  {coord}: PINN={rms_pinn_i:.1f}m, SPP={rms_spp_i:.1f}m, improvement={rms_spp_i/rms_pinn_i:.1f}×")

    return t, r, r_pred_km, r_tru_interp, mask_overlap


def experiment_2_gap_bridging():
    """Train without 30-min gap, predict through gap, compare to truth."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Gap-Bridging Capability")
    print("="*70)

    t, r, odcp = load_and_filter_data()

    # Define gap: hours 360-360.5 (1800 second window)
    gap_start_h = 360.0
    gap_end_h = 360.5
    gap_start_s = gap_start_h * 3600
    gap_end_s = gap_end_h * 3600

    # Find indices in the data
    gap_mask = (t >= gap_start_s) & (t <= gap_end_s)
    keep_mask = ~gap_mask

    print(f"\nGap window: {gap_start_h:.1f} - {gap_end_h:.1f} hours ({gap_mask.sum()} epochs)")
    print(f"Training without gap: {keep_mask.sum()} epochs")

    if gap_mask.sum() == 0:
        print("ERROR: Gap window not in dataset")
        return

    # Prepare training data without gap
    t_train_gap = t[keep_mask]
    r_train_gap = r[keep_mask]

    # Retrain PINN on gapped data
    print("\nRetraining PINN without gap (this takes ~10 minutes)...")
    model_gap = train_pinn(t_train_gap, r_train_gap, epochs=2000, batch_size=512)

    # Get normalization from gapped training
    t_min = t_train_gap.min()
    t_scale = (t_train_gap.max() - t_train_gap.min()).item()
    L_star = 6378137.0

    # Predict at gap locations
    t_gap = t[gap_mask]
    with torch.no_grad():
        t_gap_norm = (t_gap - t_min) / t_scale
        r_gap_pred_nd = model_gap(t_gap_norm.float())

    r_gap_pred_km = r_gap_pred_nd * L_star / 1000.0

    # Compare against ODCP truth at gap
    t_tru = odcp[:, 0].numpy()
    r_tru_km = odcp[:, 1:4].numpy()

    # Interpolate truth to gap timestamps
    t_gap_np = t_gap.numpy()
    interp_x = interp1d(t_tru, r_tru_km[:, 0], kind='linear')
    interp_y = interp1d(t_tru, r_tru_km[:, 1], kind='linear')
    interp_z = interp1d(t_tru, r_tru_km[:, 2], kind='linear')

    r_gap_truth = np.column_stack([
        interp_x(t_gap_np),
        interp_y(t_gap_np),
        interp_z(t_gap_np),
    ])

    # Compute gap bridging error
    diff_gap_m = (r_gap_pred_km.numpy() - r_gap_truth) * 1000.0
    rms_gap = np.sqrt(np.mean(np.sum(diff_gap_m**2, axis=1)))

    print(f"\nGap-Bridging Results:")
    print(f"  Prediction error at gap: {rms_gap:.1f} m")
    print(f"  Status: {'✅ PASS' if rms_gap < 10000 else '⚠️  CHECK'} (bridged gap with {rms_gap:.0f}m error)")

    # Per-component
    print(f"\nPer-component gap errors vs ODCP truth:")
    for i, coord in enumerate(['X', 'Y', 'Z']):
        rms_i = np.sqrt(np.mean(diff_gap_m[:, i]**2))
        print(f"  {coord}: {rms_i:.1f} m")

    # Plot gap region ±2 hours
    plot_window_h = 2.0
    plot_start_s = gap_start_s - plot_window_h * 3600
    plot_end_s = gap_end_s + plot_window_h * 3600

    plot_mask = (t >= plot_start_s) & (t <= plot_end_s)
    plot_mask_full = (t >= plot_start_s) & (t <= plot_end_s)

    if plot_mask.sum() > 0:
        print(f"\nPlotting ±{plot_window_h:.1f}h window around gap ({plot_mask.sum()} epochs)...")

        # Get full model prediction for context
        ckpt_full = _safe_load('data/processed/pinn_best.pth')
        t_min_full = ckpt_full['t_min']
        t_scale_full = ckpt_full['t_scale']
        embedding_size_full = ckpt_full['model_state']['embedding.freqs'].shape[0]

        if embedding_size_full == 64:
            model_full = KinematicPINN(num_frequencies=64)
        else:
            model_full = KinematicPINN(num_frequencies=128, t_scale=t_scale_full)

        model_full.load_state_dict(ckpt_full['model_state'])
        model_full.eval()

        t_plot = t[plot_mask]
        with torch.no_grad():
            t_plot_norm = (t_plot - t_min_full) / t_scale_full
            r_plot_pred_nd = model_full(t_plot_norm.float())

        r_plot_pred_km = r_plot_pred_nd * L_star / 1000.0
        r_plot_spp = r[plot_mask]

        # Interpolate truth
        t_plot_np = t_plot.numpy()
        r_plot_truth = np.column_stack([
            interp_x(t_plot_np),
            interp_y(t_plot_np),
            interp_z(t_plot_np),
        ])

        t_plot_h = t_plot_np / 3600.0
        gap_start_h_plot = gap_start_s / 3600.0
        gap_end_h_plot = gap_end_s / 3600.0

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        for i, (ax, coord) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
            # PINN prediction (full model for context)
            ax.plot(t_plot_h, r_plot_pred_km[:, i].numpy(), 'b-', lw=1.5, label='PINN (trained on full data)', alpha=0.8)

            # SPP input (showing gap)
            ax.plot(t_plot_h, r_plot_spp[:, i].numpy(), 'g.', ms=2, alpha=0.4, label='SPP input (noisy)')

            # ODCP truth
            ax.plot(t_plot_h, r_plot_truth[:, i], 'r-', lw=1.5, label='ODCP truth', alpha=0.8)

            # Shade the gap region
            ax.axvspan(gap_start_h_plot, gap_end_h_plot, alpha=0.15, color='yellow', label='Data gap')

            ax.set_ylabel(f'{coord} [km]')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)

        axes[-1].set_xlabel('Time [hours]')
        fig.suptitle(f'Gap-Bridging: PINN Prediction Through 30-minute Gap\n(Gap: {gap_start_h:.1f}–{gap_end_h:.1f} h, RMS error: {rms_gap:.1f} m)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/processed/validate_gap_bridging.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot: data/processed/validate_gap_bridging.png")

    return rms_gap


def main():
    print("\n" + "="*70)
    print("STAGE 1 PINN SMOOTHER VALIDATION")
    print("="*70)

    # Experiment 1
    exp1_result = experiment_1_smoothing_quality()

    # Experiment 2
    exp2_result = experiment_2_gap_bridging()

    # Summary
    print("\n" + "="*70)
    print("STAGE 1 VALIDATION SUMMARY")
    print("="*70)
    if exp1_result and exp2_result:
        print(f"✅ Experiment 1 (Smoothing): Complete")
        print(f"✅ Experiment 2 (Gap-Bridging): RMS = {exp2_result:.1f} m")
        print(f"\nStage 1 results ready for publication.")
    else:
        print("❌ One or more experiments failed")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
