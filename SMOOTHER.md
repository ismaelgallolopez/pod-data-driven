# Stage 1 PINN Smoother — Method & Results

Smooths a 30-day arc of noisy single-point-positioning (SPP) fixes into a
continuous orbit, evaluated against ODCP ground truth.

## Result

| Metric (3D, vs ODCP truth) | Raw SPP | Old SGD PINN | Stage-1 only | **This smoother** |
|---|---|---|---|---|
| RMS            | 4248 m | 5794 m | 60.6 m | **~33 m** |
| Median         |   — | — | 44.7 m | **~18 m** |
| p99            |   — | — | 141 m | **~68 m** |

~128× better than the SPP input. The SPP error is almost entirely along-track
(from a ~25% population of ~7.7 km outliers); the robust fit cuts it by ~100×.

## Method (`src/training/train_pinn.py`)

The smoother is `r(t) = W·φ(t)` with a fixed spectral basis φ (float64 Fourier
features + degree-5 polynomial). It is a PINN — the loss is the usual
data + PDE residual — but because r is linear in W, each Gauss-Newton step is a
direct linear solve, not SGD. Two stages plus three cross-cutting fixes:

**Stage 1 — stable fit.** A stiff 22-band basis, fit by **Least-Trimmed-Squares**
(SPP is ~75% good at ~2.6 m + ~25% gross outliers at a discrete ~7.7 km offset;
OLS is dragged ~1.9 km off the good cluster — LTS locks onto the good 75%) plus
a **J2 physics penalty** that stops the non-periodic basis ringing at the arc
edges (first-5h RMS 3446 → 218 m). Robust and ring-free, but band-limited to
~55 m in the interior.

**Stage 2 — rich refinement.** A 47-band basis (orbital line + all its
J2/eccentricity sidebands + 2nd/3rd harmonics) follows the good 2.6 m data to
~18 m, but on its own rings catastrophically in the gaps left by outlier
removal. We tame it by **regularizing toward the stage-1 fit** on a dense grid:
good data dominates where present (→ 18 m), the stage-1 prior dominates in gaps
(→ no ringing). `prior_alpha≈0.1` minimizes 3D RMS.

**Cross-cutting fixes** (each a specific failure of the old SGD PINN):
1. **Inertial frame.** SPP/ODCP positions are in the rotating **ECEF** frame,
   where `r'' = a_grav` is wrong by ~0.6 m/s² (Coriolis + centrifugal). Rotate
   to inertial (`Rz(+ω_E·Δt)`), fit, rotate back on output.
2. **float64 phases.** Over ~470 orbital cycles, float32 phase rounding alone
   is a ~300–600 m floor. The basis runs in float64 end-to-end; evaluators must
   pass `t_norm` as float64.
3. **J2 sign** corrected to `+1.5·J2/r⁵` in `src/physics/orbits.py`.

Run: `python main.py` → writes `data/processed/pinn_smoother.pth` (and
`pinn_best.pth`). Report: `python scripts/stage1_report.py`.

## Note on superseded docs

`CHANGES.md`, `PDE_FIX_SUMMARY.md`, and `scripts/verify_pde_fix.py` describe the
earlier SGD/PDE-scaling fix, which this rewrite **supersedes** — they no longer
match the code and can be removed.
