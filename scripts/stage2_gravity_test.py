"""
Stage 2 gravity ablation: J2-only vs J2+J3-J6 on IDENTICAL gap arcs.

How much of the ~150 m in-gap floor (see DYNAMICAL.md) is missing STATIC ZONAL
gravity vs something else (drag / tesserals)? For each gap we fit the same masked
local arc twice — n_zonal=2 (J2) and n_zonal=6 (J2+J3-J6) — propagate each across
the gap, and compare 3D RMS and the along-track component vs ODCP truth.

CAVEAT: J3-J6 is ZONAL (axisymmetric) only. The ODCP truth uses full GGM02S to
degree/order 100 INCLUDING tesserals (plus tides/3-body/drag/SRP), so this bounds
the *zonal* gravity contribution, not all missing gravity.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Reuse the Stage-2 data/eval helpers verbatim (do not re-implement).
from scripts.stage2_gap_compare import (
    load_data, lvlh_components, _rms3d, _rms,
    GAP_MINUTES, GAP_CENTERS_H, LOCAL_HALF_S,
)
from src.training.fit_dynamics import fit_dynamics, propagate_ecef

FIG_DIR = Path('data/processed/report/stage2/gravity')


def _fit_and_score(t_fit, r_fit, t_gap, truth_gap_m, n_zonal):
    res = fit_dynamics(t_fit, r_fit, t_ref_seconds=float(t_fit.min()),
                       n_zonal=n_zonal, verbose=False)
    r_pred_m = propagate_ecef(res['model'], t_gap, res['t_ref'],
                              res['T_star'], res['L_star'])
    rms3d = _rms3d(r_pred_m - truth_gap_m)
    rad, alo, cro = lvlh_components(r_pred_m, truth_gap_m)
    return res, r_pred_m, rms3d, (_rms(rad), _rms(alo), _rms(cro))


def main():
    print("\n" + "=" * 104)
    print("STAGE 2 GRAVITY ABLATION  —  J2-only  vs  J2+J3-J6  (same arcs)")
    print("  CAVEAT: J3-J6 is ZONAL only; ODCP truth has full GGM02S d/o 100 incl. tesserals.")
    print("=" * 104)

    t, r, spline, (t_lo, t_hi) = load_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    plot_store = {}
    dalong_accum = []

    for gap_min in GAP_MINUTES:
        half_gap = gap_min * 60.0 / 2.0
        for ctr_h in GAP_CENTERS_H:
            gc = ctr_h * 3600.0
            gap_lo, gap_hi = gc - half_gap, gc + half_gap
            loc_lo, loc_hi = gc - LOCAL_HALF_S, gc + LOCAL_HALF_S
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

            t_fit, r_fit = t[fit_mask], r[fit_mask]
            truth_gap_m = spline(t_gap) * 1000.0

            print(f"\n--- gap {gap_min} min @ {ctr_h} h "
                  f"({fit_mask.sum()} fit pts, {len(t_gap)} gap pts) ---")
            res2, _, rms2_3d, rac2 = _fit_and_score(t_fit, r_fit, t_gap, truth_gap_m, 2)
            res6, _, rms6_3d, rac6 = _fit_and_score(t_fit, r_fit, t_gap, truth_gap_m, 6)

            d_rms = 100.0 * (rms2_3d - rms6_3d) / rms2_3d         # reduction %
            d_along = 100.0 * (rac2[1] - rac6[1]) / rac2[1]
            dalong_accum.append(d_along)
            rows.append((gap_min, ctr_h, rms2_3d, rms6_3d, d_rms,
                         rac2[1], rac6[1], d_along))
            print(f"    J2:    3D={rms2_3d:7.1f} m  along={rac2[1]:7.1f} m")
            print(f"    J2-J6: 3D={rms6_3d:7.1f} m  along={rac6[1]:7.1f} m  "
                  f"| dRMS={d_rms:+5.1f}%  dAlong={d_along:+5.1f}%")

            if ctr_h == 360:
                plot_store[gap_min] = dict(res2=res2, res6=res6,
                                           gap_lo=gap_lo, gap_hi=gap_hi)

    # ── table ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("COMPARISON TABLE  (in-gap RMS vs ODCP truth; dRMS%/dAlong% = reduction from adding J3-J6)")
    print("=" * 104)
    print(f"{'gap[min]':>8} {'center[h]':>9} {'S2(J2) 3D':>11} {'S2(J2-J6) 3D':>13} "
          f"{'dRMS%':>7} {'along(J2)':>10} {'along(J6)':>10} {'dAlong%':>8}")
    print("-" * 104)
    for gap_min, ctr_h, r2, r6, dr, a2, a6, da in rows:
        print(f"{gap_min:>8} {ctr_h:>9} {r2:>11.1f} {r6:>13.1f} {dr:>+7.1f} "
              f"{a2:>10.1f} {a6:>10.1f} {da:>+8.1f}")

    # ── figures (hour-360) ─────────────────────────────────────────────────────
    for gap_min, store in plot_store.items():
        _make_figure(gap_min, store, t, r, spline)

    # ── decision line ──────────────────────────────────────────────────────────
    if dalong_accum:
        mean_da = float(np.mean(dalong_accum))
        print("\n" + "-" * 104)
        print(f"DIAGNOSTIC: mean along-track reduction from J3-J6 = {mean_da:+.1f}%")
        if mean_da > 40.0:
            print("  -> static zonal gravity is a MAJOR part of the floor; next step is a "
                  "fuller gravity field (higher degree + tesserals), not drag.")
        else:
            print("  -> adding low zonals barely moves the floor; the residual is drag and/or "
                  "higher/tesseral gravity — drag + RTN empirical accelerations are the next test.")
        print("  CAVEAT: J3-J6 is ZONAL (axisymmetric) only; ODCP truth uses full GGM02S to d/o "
              "100 incl. tesserals, so this bounds the *zonal* contribution, not all missing gravity.")
        print("-" * 104 + "\n")


def _make_figure(gap_min, store, t, r, spline):
    # Plot the ERROR vs ODCP truth in LVLH (radial/along/cross), metres — at the
    # km position scale a ~150 m gap error is invisible against the ~thousands-of-km
    # orbit. The error view is where the floor and the small zonal improvement show.
    res2, res6 = store['res2'], store['res6']
    gap_lo, gap_hi = store['gap_lo'], store['gap_hi']
    gc = 360.0 * 3600.0
    plot_lo, plot_hi = gc - 2 * 3600.0, gc + 2 * 3600.0
    pm = (t >= plot_lo) & (t <= plot_hi)
    t_plot = t[pm]

    r_truth_m = spline(t_plot) * 1000.0
    r2_m = propagate_ecef(res2['model'], t_plot, res2['t_ref'],
                          res2['T_star'], res2['L_star'])
    r6_m = propagate_ecef(res6['model'], t_plot, res6['t_ref'],
                          res6['T_star'], res6['L_star'])
    lvlh2 = lvlh_components(r2_m, r_truth_m)                # (radial, along, cross)
    lvlh6 = lvlh_components(r6_m, r_truth_m)

    th = t_plot / 3600.0
    names = ['Radial', 'Along-track', 'Cross-track']
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, (ax, nm) in enumerate(zip(axes, names)):
        ax.axhline(0, color='red', lw=0.8, alpha=0.6, label='ODCP truth (0)')
        ax.plot(th, lvlh2[i], '-', lw=1.2, color='green', label='S2 (J2)')
        ax.plot(th, lvlh6[i], '-', lw=1.2, color='purple', label='S2 (J2+J3-J6)')
        ax.axvspan(gap_lo / 3600.0, gap_hi / 3600.0, alpha=0.15, color='orange',
                   label='gap')
        ax.set_ylabel(f'{nm} error [m]')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, ncol=2)
    axes[-1].set_xlabel('Time [hours]')
    fig.suptitle(f'{gap_min}-Minute Gap @ 360 h: in-gap LVLH error vs ODCP '
                 f'(J2 vs J2+J3-J6 zonal gravity)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / f'gravity_gap{gap_min}_h360.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figure saved -> {out}")


if __name__ == "__main__":
    main()
