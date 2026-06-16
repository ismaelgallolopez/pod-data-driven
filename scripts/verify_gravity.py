"""
Correctness gate for the additive zonal gravity (J2..J6) in OrbitPhysics.

Three checks over ~1000 random LEO positions; ALL must PASS before any sweep:
  1. get_zonal_acceleration(n_max=2) == get_j2_acceleration  (J2 regression / sign).
  2. a_pert = accel(6) - accel(2)  [J3-J6 only]  matches the central finite
     difference of U_pert = U(6) - U(2)  (validates J3-J6 specifically).
  3. RMS ||a_pert|| in SI lands in [1e-6, 1e-4] m/s^2  (gross unit-error catch).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.physics.orbits import OrbitPhysics

torch.manual_seed(0)


def main():
    phys = OrbitPhysics()
    L_star, T_star = phys.L_star, phys.T_star

    # ~1000 random LEO positions: |r| in [6.6e6, 7.0e6] m, random directions -> nd.
    N = 1000
    direction = torch.randn(N, 3, dtype=torch.float64)
    direction = direction / torch.norm(direction, dim=1, keepdim=True)
    r_si = (6.6e6 + (7.0e6 - 6.6e6) * torch.rand(N, 1, dtype=torch.float64))
    r_nd = (direction * r_si) / L_star

    print("=" * 70)
    print("ZONAL GRAVITY CORRECTNESS GATE  (J2..J6)")
    print("=" * 70)
    passed = True

    # ── check 1: n_max=2 reproduces get_j2_acceleration exactly ──────────────
    a_zonal2 = phys.get_zonal_acceleration(r_nd, n_max=2)
    a_j2 = phys.get_j2_acceleration(r_nd)
    d1 = (a_zonal2 - a_j2).abs().max().item()
    ok1 = d1 < 1e-12
    passed &= ok1
    print(f"\n[1] get_zonal_acceleration(n_max=2) vs get_j2_acceleration")
    print(f"    max abs diff = {d1:.3e}   (threshold 1e-12)   -> {'PASS' if ok1 else 'FAIL'}")

    # ── check 2: a_pert (J3-J6) vs central FD of U_pert ──────────────────────
    a_pert = phys.get_zonal_acceleration(r_nd, n_max=6) \
        - phys.get_zonal_acceleration(r_nd, n_max=2)

    def U_pert(r):
        # U(6) - U(2): the 1/r and J2 terms cancel analytically, so summing ONLY
        # the J3-J6 terms is mathematically identical but avoids the catastrophic
        # cancellation of subtracting two ~1/r (~0.15) numbers to get a ~1e-6
        # perturbation (which otherwise floors the FD at ~1e-4 rel error).
        rn = torch.norm(r, dim=1)
        s = r[:, 2] / rn
        U = torch.zeros_like(rn)
        for n in (3, 4, 5, 6):
            P, _ = phys._legendre_P_Pp(n, s)
            U = U - phys._zonal_J(n) * rn ** (-(n + 1)) * P
        return U

    h = 1e-6
    a_fd = torch.zeros_like(r_nd)
    for k in range(3):
        rp = r_nd.clone(); rp[:, k] += h
        rm = r_nd.clone(); rm[:, k] -= h
        a_fd[:, k] = (U_pert(rp) - U_pert(rm)) / (2 * h)        # a = +grad U
    rel = (torch.norm(a_fd - a_pert, dim=1)
           / torch.norm(a_pert, dim=1)).max().item()
    ok2 = rel < 1e-5
    passed &= ok2
    print(f"\n[2] a_pert (J3-J6) vs central FD of U_pert  (h={h})")
    print(f"    max per-point rel err = {rel:.3e}   (threshold 1e-5)   "
          f"-> {'PASS' if ok2 else 'FAIL'}")

    # ── check 3: magnitude sanity in SI ──────────────────────────────────────
    a_pert_si = a_pert * (L_star / T_star**2)
    rms = torch.sqrt(torch.mean(torch.sum(a_pert_si**2, dim=1))).item()
    ok3 = 1e-6 <= rms <= 1e-4
    passed &= ok3
    print(f"\n[3] RMS ||a_pert|| (J3-J6) in SI")
    print(f"    = {rms:.3e} m/s^2   (expect ~1e-5, allow [1e-6,1e-4])   "
          f"-> {'PASS' if ok3 else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"GATE: {'PASS — safe to run the gravity sweep' if passed else 'FAIL — do NOT run the sweep'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
