import torch
import math

class OrbitPhysics:
    def __init__(self):
        self.mu      = 3.986004418e14
        self.R_earth = 6378137.0
        self.J2      = 1.08262668e-3
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
        # Standard formula: a_j2_i = -(3/2)*J2*(1/r^5) * r_i * (5*(z/r)^2 - 1)  for i=x,y
        #                   a_j2_z = -(3/2)*J2*(1/r^5) * z   * (5*(z/r)^2 - 3)
        c      = -1.5 * self.J2 / r_mag**5                     # (N,1)
        zr2    = (z / r_mag)**2                                 # (N,1)

        a_j2_x = c * r_nd[:, 0:1] * (5*zr2 - 1)
        a_j2_y = c * r_nd[:, 1:2] * (5*zr2 - 1)
        a_j2_z = c * z            * (5*zr2 - 3)

        a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

        return a_kep + a_j2

    def si_to_nd_pos(self, r_si):
        return r_si / self.L_star

    def nd_to_si_pos(self, r_nd):
        return r_nd * self.L_star