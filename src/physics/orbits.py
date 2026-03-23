import torch

class OrbitPhysics:
    def __init__(self):
        # Physical Constants (SI Units)
        self.mu = 3.986004418e14        # Earth's gravitational parameter [m^3/s^2]
        self.R_earth = 6378137.0         # Earth's equatorial radius [m]
        self.J2 = 1.08262668e-3         # J2 perturbation coefficient [-]

        # Non-dimensionalization factors (for better NN convergence)
        self.L_star = self.R_earth       # Distance unit: 1 Earth Radius
        self.T_star = torch.sqrt(self.L_star**3 / self.mu) # Time unit: ~806.8 seconds
        
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
        
        # J2 Perturbation terms
        # Factor: (3/2) * J2 * (R_e / r)^2
        # Since r is already non-dimensionalized by R_e, R_e/r becomes 1/r_mag
        j2_factor = 1.5 * self.J2 * (1.0 / r_mag**2)
        
        z_ratio_sq = (z / r_mag)**2
        
        ax_j2 = a_kep[:, 0:1] * j2_factor * (1.0 - 5.0 * z_ratio_sq)
        ay_j2 = a_kep[:, 1:2] * j2_factor * (1.0 - 5.0 * z_ratio_sq)
        az_j2 = a_kep[:, 2:3] * j2_factor * (3.0 - 5.0 * z_ratio_sq)
        
        a_j2 = torch.cat([ax_j2, ay_j2, az_j2], dim=1)
        
        return a_kep + a_j2

    def nd_to_si_pos(self, r_nd):
        return r_nd * self.L_star

    def si_to_nd_pos(self, r_si):
        return r_si / self.L_star