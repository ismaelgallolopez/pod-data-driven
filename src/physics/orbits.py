import torch
import math

class OrbitPhysics:
    def __init__(self):
        # Physical Constants (SI Units)
        self.mu = 3.986004418e14        # Earth's gravitational parameter [m^3/s^2]
        self.R_earth = 6378137.0         # Earth's equatorial radius [m]
        self.J2 = 1.08262668e-3         # J2 perturbation coefficient [-]

        # Non-dimensionalization factors (for better NN convergence)
        self.L_star = self.R_earth       # Distance unit: 1 Earth Radius
        self.T_star = math.sqrt(self.L_star**3 / self.mu) # Time unit: ~806.8 seconds
        
    def get_j2_acceleration(self, r_vec):
        """
        Calculates acceleration including J2 perturbation in ECI/ECEF frame.
        r_vec: PyTorch tensor of shape (N, 3) in NON-DIMENSIONAL units.
        """
        # Distance from center
        r_mag = torch.norm(r_vec, dim=1, keepdim=True)
        z = r_vec[:, 2:3]
        
        # Standard Two-Body Gravity (Keplerian)
        # a_kep = -mu * r / r^3. In non-dim units, mu becomes 1.
        a_kep = -r_vec / r_mag**3
        
        # J2 Perturbation (additive term) -- Vallado formulation
        # In SI: a_j2 = (3/2) * J2 * mu * Re^2 / r^5 * [ x*(5*(z/r)^2 - 1), y*(5*(z/r)^2 - 1), z*(5*(z/r)^2 - 3) ]
        # In our non-dimensional units (Re -> 1, mu -> 1) this becomes:
        # factor = 1.5 * J2 / r^5
        z_sq = z * z
        common_ratio = (5.0 * z_sq / (r_mag**2))
        factor = 1.5 * self.J2 / (r_mag**5)

        ax_j2 = factor * r_vec[:, 0:1] * (common_ratio - 1.0)
        ay_j2 = factor * r_vec[:, 1:2] * (common_ratio - 1.0)
        az_j2 = factor * r_vec[:, 2:3] * (common_ratio - 3.0)

        a_j2 = torch.cat([ax_j2, ay_j2, az_j2], dim=1)

        return a_kep + a_j2

    def nd_to_si_pos(self, r_nd):
        return r_nd * self.L_star

    def si_to_nd_pos(self, r_si):
        return r_si / self.L_star