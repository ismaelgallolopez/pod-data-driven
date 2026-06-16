import torch
import math

class OrbitPhysics:
    def __init__(self):
        self.mu      = 3.986004418e14
        self.R_earth = 6378137.0
        self.J2      = 1.08262668e-3
        # higher zonal harmonics (standard EGM unnormalized; signs matter:
        # J3,J4,J5 negative, J6 positive). Used ONLY by the additive
        # get_zonal_acceleration below — get_j2_acceleration is untouched.
        self.J3      = -2.53265649e-6
        self.J4      = -1.61962159e-6
        self.J5      = -2.27296083e-7
        self.J6      = +5.40681239e-7
        self.L_star  = self.R_earth
        self.T_star  = math.sqrt(self.L_star**3 / self.mu)  # ~806.8 s

    def get_j2_acceleration(self, r_nd):
        """
        J2 acceleration in non-dimensional units (L_star, T_star).
        r_nd: (N, 3) tensor, positions non-dimensionalised by L_star.
        Returns acceleration in [L_star / T_star^2].
        """
        r_mag = torch.norm(r_nd, dim=1, keepdim=True)          # (N,1)
        z     = r_nd[:, 2:3]                                    # (N,1)

        # Two-body (non-dim mu = 1 by construction)
        a_kep = -r_nd / r_mag**3                                # (N,3)

        # J2 — additive correction, NOT multiplicative through a_kep
        # a_j2_i = +(3/2)*J2*(1/r^5) * r_i * (5*(z/r)^2 - 1)  for i=x,y
        # a_j2_z = +(3/2)*J2*(1/r^5) * z   * (5*(z/r)^2 - 3)
        # Sign check against the potential V = (1/r)(1 - J2 P2(sin phi)/r^2),
        # a = grad V: at the equator the J2 perturbation points INWARD
        # (a_x = -1.5 J2 x / r^5 there), which requires c > 0 below.
        c      = +1.5 * self.J2 / r_mag**5                     # (N,1)
        zr2    = (z / r_mag)**2                                 # (N,1)

        a_j2_x = c * r_nd[:, 0:1] * (5*zr2 - 1)
        a_j2_y = c * r_nd[:, 1:2] * (5*zr2 - 1)
        a_j2_z = c * z            * (5*zr2 - 3)

        a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

        return a_kep + a_j2

    # ── zonal gravity J2..J6 (ADDITIVE; get_j2_acceleration stays frozen) ──────

    @staticmethod
    def _legendre_P_Pp(n, s):
        """Unnormalized Legendre polynomial P_n(s) and its derivative P_n'(s)."""
        s2 = s * s
        if n == 2:
            return (3*s2 - 1) / 2, 3*s
        if n == 3:
            return (5*s2*s - 3*s) / 2, (15*s2 - 3) / 2
        if n == 4:
            return (35*s2*s2 - 30*s2 + 3) / 8, (35*s2*s - 15*s) / 2
        if n == 5:
            return (63*s2*s2*s - 70*s2*s + 15*s) / 8, (315*s2*s2 - 210*s2 + 15) / 8
        if n == 6:
            return (231*s2**3 - 315*s2*s2 + 105*s2 - 5) / 16, \
                   (1386*s2*s2*s - 1260*s2*s + 210*s) / 16
        raise ValueError(f"zonal degree {n} not supported (2..6)")

    def _zonal_J(self, n):
        return {2: self.J2, 3: self.J3, 4: self.J4, 5: self.J5, 6: self.J6}[n]

    def zonal_potential(self, r_nd, n_max=6):
        """Non-dim gravitational potential including zonal terms J2..J_{n_max}.

        U(r) = (1/r) [ 1 - sum_{n=2}^{n_max} J_n (1/r^n) P_n(z/r) ]
             = 1/r - sum_{n} J_n r^{-(n+1)} P_n(s),   s = z/|r|, R_earth_nd = 1.
        r_nd: (N,3) -> (N,) potential. Acceleration is +grad U.
        """
        r = torch.norm(r_nd, dim=1)                            # (N,)
        s = r_nd[:, 2] / r
        U = 1.0 / r
        for n in range(2, n_max + 1):
            P, _ = self._legendre_P_Pp(n, s)
            U = U - self._zonal_J(n) * r ** (-(n + 1)) * P
        return U

    def get_zonal_acceleration(self, r_nd, n_max=6):
        """Non-dim acceleration = two-body + zonal J2..J_{n_max}, a = +grad U.

        n_max=2 reproduces get_j2_acceleration EXACTLY (verified in the gate).
        Closed-form Cartesian gradient of U_n = -J_n r^{-(n+1)} P_n(s):
          grad U_n = -J_n ( [-(n+1) P_n r^{-(n+3)} - P_n' z r^{-(n+4)}] r_vec
                            + [P_n' r^{-(n+2)}] e_z ).
        """
        x = r_nd[:, 0:1]; y = r_nd[:, 1:2]; z = r_nd[:, 2:3]
        r = torch.norm(r_nd, dim=1, keepdim=True)              # (N,1)
        s = z / r

        ax = -x / r**3                                         # two-body
        ay = -y / r**3
        az = -z / r**3
        for n in range(2, n_max + 1):
            Jn = self._zonal_J(n)
            P, Pp = self._legendre_P_Pp(n, s)
            cr = -Jn * (-(n + 1) * P * r ** (-(n + 3)) - Pp * z * r ** (-(n + 4)))
            ce = -Jn * (Pp * r ** (-(n + 2)))                  # e_z component only
            ax = ax + cr * x
            ay = ay + cr * y
            az = az + cr * z + ce
        return torch.cat([ax, ay, az], dim=1)

    def si_to_nd_pos(self, r_si):
        return r_si / self.L_star

    def nd_to_si_pos(self, r_nd):
        return r_nd * self.L_star