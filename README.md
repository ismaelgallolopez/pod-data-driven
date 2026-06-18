# Precise Orbit Determination: Data-Driven Methods for LEO Satellites

A machine learning application for smoothing noisy single-point-positioning (SPP) fixes into continuous, accurate orbits for LEO satellites against ODCP ground truth.

## Summary

The project develops a two-stage approach combining PINN-based spectral smoothing (Stage 1) with dynamical gap-bridging (Stage 2) to achieve ~33 m RMS accuracy (128× improvement over raw 4.2 km SPP noise).

### Key Results

| Method | RMS | Median | p99 |
|--------|-----|--------|-----|
| Raw SPP | 4248 m | — | — |
| **Stage 1: PINN Smoother** | **~33 m** | **~18 m** | **~68 m** |
| Stage 2: Dynamical (J2) | Bridges gaps 6–65× better than spectral basis |

## Architecture

### Stage 1: PINN Spectral Smoother
- **Basis**: Fixed spectral features (Fourier + degree-5 polynomial, float64)
- **Fit**: Two-stage Least-Trimmed-Squares + Gauss-Newton (robust to outliers)
- **Physics penalty**: J2 constraint prevents edge ringing
- **Output**: Smooth orbit `r(t) = W·φ(t)` with ~33 m RMS

### Stage 2: Dynamical Gap-Bridging  
- **Method**: 6-parameter orbit propagation (inertial frame, J2 dynamics)
- **Solver**: Robust Gauss-Newton (scipy.least_squares) + torchdiffeq
- **Gap performance**: 30–65× better than spectral basis alone
- **Bottleneck**: Along-track dominated; next forces are drag + empirical accelerations

## Data & Setup

- **Satellite**: CHAMP (synthetic data from realistic GHOST simulator)
- **Input**: Raw SPP (~2.6 m good data + 7.7 km gross outliers)
- **Truth**: ODCP ground truth (GGM02S d/o 100, tides, 3-body, drag, SRP, IERS EOP)
- **Arc**: 30-day continuous mission

## Key Technical Insights

1. **Inertial frame**: SPP/ODCP are ECEF (rotating); fitting in inertial removes ~0.6 m/s² Coriolis/centrifugal error
2. **float64 precision**: float32 phases over 470 cycles cause 300–600 m RMS floor; all basis operations use float64
3. **Robust outlier handling**: LTS recovers timetag errors (outliers are +1.000 s shifts, not noise)
4. **Two-stage refinement**: Rich spectral basis alone rings catastrophically in gaps; regularizing toward Stage-1 prior controls it

## Files & Execution

- `main.py` — Train Stage 1 PINN smoother
- `scripts/stage1_report.py` — Generate Stage 1 diagnostic plots
- `scripts/stage2_gap_compare.py` — Benchmark Stage 2 dynamical bridging vs Stage 1
- `scripts/analysis_plots.py` — Generate final analysis figures
- Models: `data/processed/pinn_smoother.pth`, `pinn_best.pth`

## Project Status

**Closed for now.** The two-stage approach establishes the accuracy floor under current force models. Further improvement requires: (a) empirical drag/RTN accelerations in Stage 2, or (b) learned residual force fields (UDE / Path B), which would exceed the J2-only physical ceiling (~2.5 m).
