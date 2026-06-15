# Stage 2 (variant A) — Purely Dynamical Gap-Bridging

Answers one question: across a data gap, what can a **known-force propagation
alone** do, versus the Stage-1 spectral smoother? No neural residual yet — this
is the J2-only baseline that motivates adding learned/empirical forces.

## Idea

Represent a short arc by its initial state `z0 = [r0, v0]` (6 numbers) at a
reference epoch and propagate `r'' = a(r)` with a known force model. In a gap the
trajectory is just the propagated orbit — constrained by **6 parameters**, not
the smoother's ~1000 spectral coefficients. Same inertial frame, `rotate_z`
convention, float64, and reused `OrbitPhysics.get_j2_acceleration` as Stage 1.

## Result (in-gap 3D RMS vs ODCP truth, metres)

Both methods **blind to the gap** (re-fit on the same masked ±2.5 h local arc):

| gap | center h | Stage-1 (smoother, rings) | **Stage-2 (J2 dynamical)** | factor |
|---|---|---|---|---|
| 30 min | 120 | 3614.6 | **120.0** | 30× |
| 30 min | 240 |  947.0 | **151.6** |  6× |
| 30 min | 360 | 3917.0 | **117.5** | 33× |
| 30 min | 480 | 4999.7 | **182.2** | 27× |
| 60 min | 120 | 2035.3 | **134.7** | 15× |
| 60 min | 240 | 4621.5 | **171.0** | 27× |
| 60 min | 360 | 9766.4 | **149.2** | 65× |
| 60 min | 480 | 6806.6 | **200.2** | 34× |

A 6-parameter dynamical orbit bridges gaps **6–65× better** than the spectral
basis, which rings to km-level once it loses its data. Mean Stage-2 gap error is
**along-track-dominated** (R/A/C ≈ 27 / 144 / 45 m) — the signature that J2-only
is the limit and the next forces to add are **drag + RTN empirical accelerations**.

## Method (`src/training/fit_dynamics.py`)

- **`DynamicalOrbit`** — learnable `z0` (non-dim, inertial); `forward(t_nd)`
  integrates `r''=J2` with **torchdiffeq** (dopri5, rtol/atol 1e-9, float64). The
  canonical propagator for scoring. `accel()` reuses `get_j2_acceleration` and
  has labeled `# TODO` hooks + a `params` dict for drag / empirical-RTN forces.
- **Fitting `z0` is the classic orbit-determination differential-correction
  problem**, not an SGD problem: position vs. velocity have wildly different
  leverage over an arc, which a single-learning-rate optimiser (Adam) cannot
  condition — it gave 7 km fit RMS in 36 min. Instead we use **robust
  Gauss-Newton** (`scipy.least_squares`, soft-L1) over the 6 non-dim parameters,
  on a fast numpy DOP853 mirror of the *same* J2 force (~0.12 s/integration,
  ~60 s/fit). A second hard-reject pass removes the ~25% **+1.000 s timetag
  outliers** (a ~7.7 km along-track shift) that survive the IQR magnitude filter.
- **Initialisation from data:** `r0` = nearest inertial point to `t_ref`; `v0` =
  derivative of a degree-2 fit to ~5 min of inertial track. A good `v0` is
  essential for convergence.

## Fair-comparison caveat

`data/processed/pinn_best.pth` is the full 30-day smoother trained **with** all
data, so evaluating it inside a synthetic gap is *interpolation* (~10–17 m), not
bridging — the "645 m / 1236 m" strings in `stage1_report.py` are hardcoded
titles from a separate masked experiment. The table above therefore re-fits a
**gap-blind** Stage-1 (`train_pinn` → temp checkpoint, `prior_alpha=0`, the
ring-free variant) on the same masked arc; the loaded full model is reported only
as a labeled interpolation reference.

Run: `python scripts/stage2_gap_compare.py` → table + LVLH diagnostic, figures to
`data/processed/report/stage2/`. Does **not** modify any Stage-1 checkpoint/script.
