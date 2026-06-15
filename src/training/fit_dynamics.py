"""
Stage 2 (variant A) — PURELY DYNAMICAL orbit fit.

Represent a short orbital arc by its initial state z0 = [r0, v0] (6 numbers) at a
reference epoch and propagate it with a KNOWN force model (J2-only here), then
fit z0 to the surrounding SPP data by robust least-squares. In a data gap the
trajectory is just the propagated orbit — constrained by 6 parameters instead of
the smoother's ~1000 spectral coefficients. This file answers one question: what
can a known-force propagation alone do across a gap?

No neural residual yet. The force model is deliberately J2-only; drag and
empirical (RTN) accelerations have clearly-marked TODO hooks in `accel()` so the
next step (learned/added forces) drops straight in.

Two integrators, ONE force (get_j2_acceleration), both float64 dopri5/DOP853 at
rtol/atol 1e-9 — so they agree to ~1e-6 relative:
  - DynamicalOrbit.forward uses torchdiffeq odeint and is the canonical
    propagator for evaluation/propagate_ecef.
  - fitting z0 is the classic orbit-determination differential-correction
    problem (position vs. velocity have wildly different leverage over an arc),
    which a single-learning-rate optimiser like Adam cannot condition. We solve
    it the textbook way — robust Gauss-Newton (scipy.least_squares, soft-L1) over
    the 6 non-dim parameters — using a fast numpy mirror of the SAME J2 force.
    The fitted z0 is then loaded into the torch model.

Frame & units (must match the Stage 1 smoother EXACTLY — see train_pinn.py and
the project memory on the ECEF/inertial frame bug):
  - Positions are non-dimensionalised by L_star = R_earth, velocities by
    L_star / T_star, time by T_star (so non-dim mu = 1).
  - The SPP/ODCP data is ECEF. r'' = a_grav is wrong there by ~0.6 m/s^2, so we
    rotate ECEF -> inertial about z by +OMEGA_EARTH*(t - t_ref) (the same
    `rotate_z` convention as Stage 1), fit/propagate the dynamics in inertial,
    and rotate inertial -> ECEF by the inverse for any comparison vs SPP/ODCP.
  - Everything is float64 end to end.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.physics.orbits import OrbitPhysics
# Reuse the EXACT Stage-1 rotation convention and Earth rate so the frames match.
from src.training.train_pinn import rotate_z, OMEGA_EARTH

try:
    from torchdiffeq import odeint
except Exception as e:                                    # pragma: no cover
    raise RuntimeError(
        "torchdiffeq is required for Stage 2 dynamical fitting. "
        "Install it with `pip install torchdiffeq`."
    ) from e


_PHYS = OrbitPhysics()                                    # single shared instance


# ── force model ──────────────────────────────────────────────────────────────

def accel(r, v, t, params):
    """Non-dimensional acceleration of a LEO satellite at inertial position `r`.

    r, v : (3,) float64 tensors in non-dim units (L_star, L_star/T_star).
    t    : scalar non-dim time (unused by J2; kept for drag/empirical hooks).
    params : dict — carries the force-model configuration. Today only the J2
             two-body term is implemented; `params` and the hooks below exist so
             drag and empirical (RTN) accelerations can be added WITHOUT changing
             the caller or the integrator.

    Returns (3,) non-dim acceleration [L_star / T_star^2].
    """
    phys = params.get('phys', _PHYS)

    # --- two-body + J2 (the only force implemented) -------------------------
    # OrbitPhysics.get_j2_acceleration expects (N,3) and returns (N,3); it already
    # carries the corrected +1.5*J2/r^5 sign and the non-dim mu=1 convention.
    a = phys.get_j2_acceleration(r.reshape(1, 3)).reshape(3)

    # --- TODO hook: atmospheric drag ---------------------------------------
    # a_drag = -0.5 * params['BC'] * rho(r) * |v_rel| * v_rel  (needs v, density
    # model, and the co-rotating atmosphere velocity). Add to `a` here.

    # --- TODO hook: empirical / learned RTN accelerations ------------------
    # Build the RTN (radial/along/cross) unit triad from r, v and add
    # params['emp_R/T/N'] (constant or a small learned/Fourier term). This is
    # where a Stage-2 neural residual force will plug in.

    return a


class DynamicalOrbit(nn.Module):
    """A single orbital arc parameterised by its initial state z0 = [r0, v0].

    z0 is a learnable (6,) float64 tensor in NON-DIM units at the reference epoch
    (t_nd = 0). forward(t_nd) integrates r'' = accel(r) from t=0 to the sorted
    non-dim evaluation times and returns inertial non-dim positions (N, 3).
    """

    def __init__(self, z0_init, params=None, rtol=1e-9, atol=1e-9, method='dopri5'):
        super().__init__()
        z0 = torch.as_tensor(z0_init, dtype=torch.float64).reshape(6).clone()
        self.z0 = nn.Parameter(z0)
        self.params = params if params is not None else {'phys': _PHYS}
        self.rtol, self.atol, self.method = rtol, atol, method

    def _rhs(self, t, z):
        r, v = z[:3], z[3:]
        a = accel(r, v, t, self.params)
        return torch.cat([v, a])

    def forward(self, t_nd):
        """Integrate from t_nd=0 to each (sorted, >=0) non-dim time; return (N,3).

        odeint needs a STRICTLY increasing grid starting at the integration epoch
        (t=0, where z0 lives), but the observation times can contain duplicates.
        We integrate on the unique grid (with 0 prepended if absent) and gather
        the solution back onto every requested epoch.
        """
        t_nd = torch.as_tensor(t_nd, dtype=torch.float64).reshape(-1)
        uniq, inv = torch.unique(t_nd, sorted=True, return_inverse=True)
        prepend = bool(uniq[0] > 0)
        grid = torch.cat([uniq.new_zeros(1), uniq]) if prepend else uniq
        sol = odeint(self._rhs, self.z0, grid,
                     rtol=self.rtol, atol=self.atol, method=self.method)
        pos = sol[1:, :3] if prepend else sol[:, :3]       # (n_uniq, 3)
        return pos[inv]


# ── helpers ──────────────────────────────────────────────────────────────────

def _iqr_mask(r_km):
    """The same IQR rule used elsewhere (q25/q75, 10*IQR on |r|)."""
    r_mag = np.linalg.norm(r_km, axis=1)
    q25, q75 = np.quantile(r_mag, 0.25), np.quantile(r_mag, 0.75)
    iqr = q75 - q25
    return (r_mag >= q25 - 10 * iqr) & (r_mag <= q75 + 10 * iqr)


def _accel_np(r_nd, params):
    """numpy mirror of accel(): non-dim J2 acceleration via the SAME torch
    get_j2_acceleration (so the fit and the torchdiffeq propagator share one
    force). r_nd: (3,) -> (3,)."""
    phys = params.get('phys', _PHYS)
    r = torch.from_numpy(np.ascontiguousarray(r_nd)).reshape(1, 3)
    return phys.get_j2_acceleration(r).reshape(3).numpy()


def _integrate_np(z0, t_nd, params, rtol=1e-9, atol=1e-9):
    """Propagate z0 with DOP853 to the (sorted, >=0) non-dim times; return (N,3)
    inertial nd positions. Integrates the unique grid and gathers back, so
    duplicate observation epochs are fine."""
    t_nd = np.asarray(t_nd, dtype=np.float64).ravel()
    uniq, inv = np.unique(t_nd, return_inverse=True)
    grid = uniq if uniq[0] == 0.0 else np.concatenate([[0.0], uniq])

    def rhs(t, z):
        return np.concatenate([z[3:], _accel_np(z[:3], params)])

    sol = solve_ivp(rhs, (grid[0], grid[-1]), z0, t_eval=grid,
                    rtol=rtol, atol=atol, method='DOP853')
    pos = sol.y[:3].T                                      # (len(grid), 3)
    if grid[0] != uniq[0] or len(grid) != len(uniq):       # 0 was prepended
        pos = pos[1:]
    return pos[inv]


def propagate_ecef(model, t_seconds, t_ref, T_star, L_star, omega=OMEGA_EARTH):
    """Propagate the fitted orbit to `t_seconds` and rotate inertial -> ECEF.

    Returns ECEF positions in METRES (np.ndarray, (N,3)). Handles unsorted input.
    """
    t = np.asarray(t_seconds, dtype=np.float64).ravel()
    order = np.argsort(t)
    inv = np.argsort(order)
    t_sorted = t[order]
    t_nd = (t_sorted - t_ref) / T_star
    with torch.no_grad():
        inert_nd = model(torch.from_numpy(t_nd)).numpy()        # inertial nd
    inert_m = inert_nd * L_star
    # inverse of rotate_z(+omega*(t-t_ref)): rotate by -omega*(t-t_ref).
    ecef_m = rotate_z(t_sorted, inert_m, -omega, t_ref)
    return ecef_m[inv]


# ── fit ──────────────────────────────────────────────────────────────────────

def fit_dynamics(t_seconds, r_km, t_ref_seconds=None, max_nfev=200,
                 huber_delta_m=5.0, init_window_s=300.0, init_poly_deg=2,
                 params=None, rtol=1e-9, atol=1e-9, verbose=True):
    """Fit z0 = [r0, v0] of a J2-propagated arc to the SPP positions.

    t_seconds : (N,) epochs (s).  r_km : (N,3) ECEF positions (km).
    t_ref_seconds : reference epoch (default: arc start).
    huber_delta_m : soft-L1 robustness scale (the ~few-metre SPP noise knee).

    z0 is found by robust Gauss-Newton (scipy.least_squares, soft-L1) over the 6
    non-dim parameters — the orbit-determination differential-correction method —
    then loaded into the torchdiffeq DynamicalOrbit returned in `model`.

    Returns dict with:
      model (DynamicalOrbit), t_ref, L_star, T_star, rot_omega (= OMEGA_EARTH),
      n_used, z0 (fitted, nd), fit_rms_m.
    """
    def log(m):
        if verbose:
            print(m, flush=True)

    L_star, T_star = _PHYS.L_star, _PHYS.T_star
    if params is None:
        params = {'phys': _PHYS}

    t = np.asarray(t_seconds, dtype=np.float64).ravel()
    r_ecef_km = np.asarray(r_km, dtype=np.float64).reshape(-1, 3)

    # sort + IQR prefilter (rotation preserves |r|, so filter on the ECEF input)
    order = np.argsort(t)
    t, r_ecef_km = t[order], r_ecef_km[order]
    keep = _iqr_mask(r_ecef_km)
    t, r_ecef_km = t[keep], r_ecef_km[keep]
    n_used = len(t)

    t_ref = float(t[0]) if t_ref_seconds is None else float(t_ref_seconds)

    # ECEF -> inertial (same convention as Stage 1), then non-dimensionalise.
    r_ecef_m = r_ecef_km * 1000.0
    r_inert_m = rotate_z(t, r_ecef_m, +OMEGA_EARTH, t_ref)
    r_inert_nd = r_inert_m / L_star
    t_nd = (t - t_ref) / T_star

    # ── initialise z0 FROM THE DATA (a good v0 is essential for convergence) ──
    # r0: inertial nd point nearest t_ref.
    i0 = int(np.argmin(np.abs(t - t_ref)))
    r0 = r_inert_nd[i0].copy()
    # v0: derivative at t_ref of a low-order poly fit to a short window of the
    # inertial nd track (nd velocity = d r_nd / d t_nd).
    win = np.abs(t - t_ref) <= init_window_s
    if win.sum() < init_poly_deg + 1:                       # fall back: widen
        idx = np.argsort(np.abs(t - t_ref))[:max(init_poly_deg + 1, 5)]
        win = np.zeros(len(t), bool); win[idx] = True
    tw = t_nd[win]
    v0 = np.empty(3)
    for k in range(3):
        c = np.polyfit(tw, r_inert_nd[win, k], init_poly_deg)
        v0[k] = np.polyval(np.polyder(c), 0.0)             # derivative at t_nd=0
    z0_init = np.concatenate([r0, v0])

    log(f"fit_dynamics | N_used={n_used} | span={(t[-1]-t[0])/3600:.2f} h | "
        f"t_ref={t_ref/3600:.3f} h | frame: inertial")
    log(f"  z0 init: |r0|={np.linalg.norm(r0)*L_star/1e3:.1f} km  "
        f"|v0|={np.linalg.norm(v0)*L_star/T_star:.1f} m/s")

    # ── robust Gauss-Newton on z0 (differential correction) ─────────────────
    # Residuals are the inertial position misfit in METRES; the fast numpy
    # integrator shares the J2 force with the torch propagator.
    #
    # Robustness in two passes. The IQR prefilter removes only magnitude
    # outliers, NOT the ~25% SPP epochs carrying a +1.000 s timetag error (a
    # ~7.7 km ALONG-TRACK shift at normal |r| — see project memory). Those would
    # bias a plain least-squares velocity, so: (1) soft-L1 fit to lock onto the
    # good cluster, then (2) hard-reject the gross outliers and refit clean. The
    # reported fit RMS is over the inliers, the meaningful number.
    def residuals_at(z0, mask=None):
        pred_nd = _integrate_np(z0, t_nd, params, rtol=rtol, atol=atol)
        d = (pred_nd - r_inert_nd) * L_star
        return (d[mask] if mask is not None else d).reshape(-1)

    sol = least_squares(residuals_at, z0_init, method='trf', loss='soft_l1',
                        f_scale=huber_delta_m, max_nfev=max_nfev,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, verbose=0)

    rn = np.linalg.norm(residuals_at(sol.x).reshape(-1, 3), axis=1)
    mad = 1.4826 * np.median(rn)
    inl = rn < max(200.0, 6.0 * mad)                       # keep the good cluster
    sol2 = least_squares(lambda z0: residuals_at(z0, inl), sol.x, method='trf',
                         loss='linear', max_nfev=max_nfev,
                         xtol=1e-12, ftol=1e-12, gtol=1e-12, verbose=0)

    z0_fit = sol2.x
    res_in = residuals_at(z0_fit, inl).reshape(-1, 3)
    fit_rms = float(np.sqrt(np.mean(np.sum(res_in ** 2, axis=1))))
    log(f"  GN converged: nfev={sol.nfev}+{sol2.nfev}  inliers={inl.mean()*100:.0f}% "
        f"fit RMS={fit_rms:7.1f} m  (|v0| {np.linalg.norm(z0_fit[3:])*L_star/T_star:.1f} m/s)")

    model = DynamicalOrbit(z0_fit, params=params, rtol=rtol, atol=atol)
    model.eval()
    return {
        'model': model, 't_ref': t_ref, 'L_star': L_star, 'T_star': T_star,
        'rot_omega': OMEGA_EARTH, 'n_used': n_used, 'fit_rms_m': fit_rms,
        'z0': z0_fit.copy(),
    }
