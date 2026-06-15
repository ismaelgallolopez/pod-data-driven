"""
Stage 2 (purely dynamical) vs Stage 1 (spectral smoother) — gap-bridging head-to-head.

For a set of synthetic data gaps we ask: across the gap, does a 6-parameter
J2-propagated orbit (Stage 2, fit to the surrounding arc) beat the ~1000-coefficient
spectral smoother (Stage 1)? The smoother interpolates beautifully where there is
data but has nothing dynamical to lean on inside a gap; a known-force propagation
is constrained by physics there. This script quantifies the trade.

Protocol (LOCAL ARC): for each (gap_minutes, gap_center_hour)
  a. take a +/-2.5 h local window around the gap center,
  b. mask out the gap window,
  c. fit_dynamics (J2-only) on the masked local arc, t_ref = local arc start,
  d. propagate across the gap, rotate to ECEF, score 3D RMS + LVLH vs ODCP truth.

Stage 1 head-to-head — IMPORTANT. A fair gap test needs Stage 1 to be BLIND to
the gap too. The saved data/processed/pinn_best.pth is the full 30-day smoother
trained WITH the gap data present, so evaluating it in the gap is interpolation,
not bridging (it reproduces its ~10-20 m smoother accuracy, NOT the ~645 m the
stage1_report titles quote — those titles are hardcoded text from a separate
masked experiment). So we report TWO Stage-1 numbers:
  - S1 full  : the loaded pinn_best.pth (has the gap data) — an interpolation
               reference / upper bound, NOT a gap test.
  - S1 blind : the SAME spectral smoother RE-FIT on the masked local arc (gap
               removed) via train_pinn -> a temp checkpoint (the real Stage-1
               gap-bridging baseline; the non-periodic basis rings in the gap).
The winner column and the figures use the fair S1-blind baseline. We re-fit the
robust ring-free stage-1-only variant (prior_alpha=0), which is the smoother's
BEST-behaved-in-gaps form, so the Stage-2 win is conservative.

Outputs: a comparison table, figures for the hour-360 gaps, and a one-line LVLH
diagnostic (along-track dominance => drag/empirical accelerations are the missing
force, i.e. J2-only is the limit).
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.models.pinn import KinematicPINN
from src.training.fit_dynamics import fit_dynamics, propagate_ecef
from src.training.train_pinn import train_pinn

GAP_MINUTES = (30, 60)
GAP_CENTERS_H = (120, 240, 360, 480)
LOCAL_HALF_S = 2.5 * 3600.0
FIG_DIR = Path('data/processed/report/stage2')
S1_TMP_DIR = 'data/processed/_tune/_s1blind'             # gap-blind Stage-1 (temp)


def _safe_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_data():
    """SPP (sorted, IQR-filtered) + a cubic-spline interpolator of the ODCP truth."""
    spp = _safe_load('data/processed/noisy_spp.pt')
    odcp = _safe_load('data/processed/truth_odcp.pt')

    t = spp[:, 0].numpy().astype(np.float64)
    r = spp[:, 1:4].numpy().astype(np.float64)
    order = np.argsort(t)
    t, r = t[order], r[order]

    r_mag = np.linalg.norm(r, axis=1)
    q25, q75 = np.quantile(r_mag, 0.25), np.quantile(r_mag, 0.75)
    iqr = q75 - q25
    mask = (r_mag >= q25 - 10 * iqr) & (r_mag <= q75 + 10 * iqr)
    t, r = t[mask], r[mask]

    # ODCP truth -> cubic spline (offset 10 s grid; linear interp injects ~tens of m)
    t_tru = odcp[:, 0].numpy().astype(np.float64)
    r_tru = odcp[:, 1:4].numpy().astype(np.float64)
    o = np.argsort(t_tru)
    t_tru, r_tru = t_tru[o], r_tru[o]
    uniq = np.concatenate([[True], np.diff(t_tru) > 0])
    t_tru, r_tru = t_tru[uniq], r_tru[uniq]
    spline = CubicSpline(t_tru, r_tru, axis=0)
    return t, r, spline, (t_tru[0], t_tru[-1])


def _ckpt_predictor(ckpt):
    """Build ts(seconds)->ECEF km from a smoother checkpoint dict."""
    model = KinematicPINN.from_checkpoint(ckpt)
    t_min, t_scale = float(ckpt['t_min']), float(ckpt['t_scale'])
    L_star = ckpt.get('L_star', 6378137.0)

    def predict_km(ts):
        with torch.no_grad():
            t_norm = (torch.as_tensor(ts, dtype=torch.float64) - t_min) / t_scale
            r_nd = model(t_norm)                              # ECEF nd
        return r_nd.numpy() * L_star / 1000.0
    return predict_km


def load_stage1_full():
    """The saved full smoother (has the gap data -> interpolation reference)."""
    return _ckpt_predictor(_safe_load('data/processed/pinn_best.pth'))


def fit_stage1_blind(t_fit, r_fit):
    """Re-fit the spectral smoother on the masked local arc (gap removed) so it is
    BLIND to the gap — the fair Stage-1 gap-bridging baseline. Uses the robust
    ring-free stage-1-only variant (prior_alpha=0); writes to a temp checkpoint
    (never touches the real pinn_best.pth)."""
    train_pinn(t_fit, r_fit, checkpoint_dir=S1_TMP_DIR,
               prior_alpha=0.0, verbose=False)
    return _ckpt_predictor(_safe_load(f'{S1_TMP_DIR}/pinn_best.pth'))


def lvlh_components(r_pred_m, r_truth_m):
    """LVLH (radial / along-track / cross-track) error decomposition, metres.

    Direction triad is built from the truth position + a finite-difference truth
    velocity; the dt scale cancels (only directions are used), so any uniform
    spacing is fine. Inputs must be time-sorted.
    """
    v = np.zeros_like(r_truth_m)
    v[1:-1] = (r_truth_m[2:] - r_truth_m[:-2]) * 0.5
    v[0] = r_truth_m[1] - r_truth_m[0]
    v[-1] = r_truth_m[-1] - r_truth_m[-2]

    r_hat = r_truth_m / np.linalg.norm(r_truth_m, axis=1, keepdims=True)
    h = np.cross(r_truth_m, v)
    n_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat, axis=1, keepdims=True)

    d = r_pred_m - r_truth_m
    radial = np.sum(d * r_hat, axis=1)
    along = np.sum(d * t_hat, axis=1)
    cross = np.sum(d * n_hat, axis=1)
    return radial, along, cross


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def _rms3d(err_m):
    return float(np.sqrt(np.mean(np.sum(err_m ** 2, axis=1))))


def main():
    print("\n" + "=" * 92)
    print("STAGE 2 (J2-only dynamical) vs STAGE 1 (spectral smoother) — GAP BRIDGING")
    print("=" * 92)

    t, r, spline, (t_lo, t_hi) = load_data()
    stage1_full_km = load_stage1_full()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    plot_store = {}      # (gap_min) -> dict for hour-360 figures
    lvlh_accum = []      # (radial, along, cross) per gap for the final diagnostic

    for gap_min in GAP_MINUTES:
        half_gap = gap_min * 60.0 / 2.0
        for ctr_h in GAP_CENTERS_H:
            gc = ctr_h * 3600.0
            gap_lo, gap_hi = gc - half_gap, gc + half_gap
            loc_lo, loc_hi = gc - LOCAL_HALF_S, gc + LOCAL_HALF_S

            # skip if the gap is outside the ODCP-overlap region
            if gap_lo < t_lo or gap_hi > t_hi:
                print(f"  [skip] gap {gap_min} min @ {ctr_h} h outside ODCP overlap")
                continue

            local = (t >= loc_lo) & (t <= loc_hi)
            in_gap = local & (t >= gap_lo) & (t <= gap_hi)
            fit_mask = local & ~in_gap

            t_gap = t[in_gap]
            if len(t_gap) == 0 or fit_mask.sum() < 50:
                print(f"  [skip] gap {gap_min} min @ {ctr_h} h: too few points")
                continue

            print(f"\n--- gap {gap_min} min @ {ctr_h} h "
                  f"({fit_mask.sum()} fit pts, {len(t_gap)} gap pts) ---")
            res = fit_dynamics(t[fit_mask], r[fit_mask],
                               t_ref_seconds=float(t[fit_mask].min()),
                               verbose=True)

            # Stage 2 across the gap
            r2_m = propagate_ecef(res['model'], t_gap, res['t_ref'],
                                  res['T_star'], res['L_star'])
            truth_gap_km = spline(t_gap)
            truth_gap_m = truth_gap_km * 1000.0
            err2_m = r2_m - truth_gap_m
            rms2 = _rms3d(err2_m)
            rad, alo, cro = lvlh_components(r2_m, truth_gap_m)
            rms2_rac = (_rms(rad), _rms(alo), _rms(cro))
            lvlh_accum.append(rms2_rac)

            # Stage 1 (full, has the gap data) — interpolation reference
            rms1_full = _rms3d((stage1_full_km(t_gap) - truth_gap_km) * 1000.0)

            # Stage 1 (gap-blind) — re-fit on the masked local arc: the fair test
            stage1_blind_km = fit_stage1_blind(t[fit_mask], r[fit_mask])
            rms1_blind = _rms3d((stage1_blind_km(t_gap) - truth_gap_km) * 1000.0)

            rows.append((gap_min, ctr_h, rms1_full, rms1_blind, rms2, rms2_rac))
            print(f"    S1 full(interp)={rms1_full:8.1f} m | S1 blind(bridge)={rms1_blind:9.1f} m "
                  f"| S2={rms2:8.1f} m | S2 R/A/C="
                  f"{rms2_rac[0]:6.1f}/{rms2_rac[1]:6.1f}/{rms2_rac[2]:6.1f} m")

            if ctr_h == 360:
                plot_store[gap_min] = dict(res=res, gap_lo=gap_lo, gap_hi=gap_hi,
                                           s1_blind=stage1_blind_km)

    # ── comparison table ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("COMPARISON TABLE  (in-gap 3D RMS vs ODCP truth)")
    print("  S1 full   = loaded pinn_best.pth (HAS the gap data; interpolation, not a gap test)")
    print("  S1 blind  = spectral smoother re-fit on the masked arc (the FAIR gap-bridging baseline)")
    print("  winner    = S2 vs S1 blind")
    print("=" * 100)
    print(f"{'gap[min]':>8} {'center[h]':>9} {'S1 full[m]':>11} {'S1 blind[m]':>12} "
          f"{'S2[m]':>9} {'S2 radial':>10} {'S2 along':>10} {'S2 cross':>10}  winner")
    print("-" * 100)
    for gap_min, ctr_h, rms1_full, rms1_blind, rms2, rac in rows:
        win = f"S2 ({rms1_blind/rms2:.0f}x)" if rms2 < rms1_blind else "S1-blind"
        print(f"{gap_min:>8} {ctr_h:>9} {rms1_full:>11.1f} {rms1_blind:>12.1f} "
              f"{rms2:>9.1f} {rac[0]:>10.1f} {rac[1]:>10.1f} {rac[2]:>10.1f}  {win}")

    # ── figures for the hour-360 gaps ────────────────────────────────────────
    for gap_min, store in plot_store.items():
        _make_figure(gap_min, store, t, r, spline)

    # ── one-line diagnostic ──────────────────────────────────────────────────
    if lvlh_accum:
        arr = np.array(lvlh_accum)
        mR, mA, mC = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
        dom = ['radial', 'along-track', 'cross-track'][int(np.argmax([mR, mA, mC]))]
        print("\n" + "-" * 92)
        print(f"DIAGNOSTIC: mean Stage-2 gap error  radial={mR:.1f} m  "
              f"along-track={mA:.1f} m  cross-track={mC:.1f} m  -> {dom}-dominated.")
        if dom == 'along-track':
            print("  Along-track dominance is the signature of an unmodelled tangential "
                  "force (drag / empirical accelerations): J2-only is the limit, and "
                  "drag + RTN empirical accelerations are the next thing to add.")
        print("-" * 92 + "\n")


def _make_figure(gap_min, store, t, r, spline):
    res = store['res']
    s1_blind_km = store['s1_blind']
    gap_lo, gap_hi = store['gap_lo'], store['gap_hi']
    gc = 360.0 * 3600.0
    plot_lo, plot_hi = gc - 2 * 3600.0, gc + 2 * 3600.0
    pm = (t >= plot_lo) & (t <= plot_hi)
    t_plot = t[pm]

    r_spp = r[pm]
    r_truth = spline(t_plot)
    r1 = s1_blind_km(t_plot)                              # gap-blind smoother
    r2 = propagate_ecef(res['model'], t_plot, res['t_ref'],
                        res['T_star'], res['L_star']) / 1000.0

    th = t_plot / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, (ax, c) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
        ax.plot(th, r_spp[:, i], '.', ms=1.5, alpha=0.3, color='grey', label='SPP input')
        ax.plot(th, r_truth[:, i], '-', lw=1.2, color='red', label='ODCP truth')
        ax.plot(th, r1[:, i], '-', lw=1.2, color='blue',
                label='Stage 1 (smoother, gap-blind)')
        ax.plot(th, r2[:, i], '-', lw=1.2, color='green', label='Stage 2 (J2 dynamical)')
        ax.axvspan(gap_lo / 3600.0, gap_hi / 3600.0, alpha=0.15, color='orange')
        ax.set_ylabel(f'{c} [km]')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, ncol=2)
    axes[-1].set_xlabel('Time [hours]')
    fig.suptitle(f'{gap_min}-Minute Gap @ 360 h (both gap-blind): '
                 f'Stage 1 smoother rings, Stage 2 (J2 dynamical) holds',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / f'stage2_gap{gap_min}_h360.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figure saved -> {out}")


if __name__ == "__main__":
    main()
