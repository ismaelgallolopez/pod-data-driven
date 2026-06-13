import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    """
    Maps t ∈ [0,1] to sinusoidal features.

    Two modes:
    - Explicit `freqs` (new): features are [sin(2πft), cos(2πft), 1, t, ..., t^n_poly].
      Phases are computed in float64: at ~470 orbital cycles, float32 phase
      rounding alone produces a ~300-600 m position error floor.
    - Legacy grids (freqs=None): reproduces the original dual-grid embedding
      [sin, cos, t] so old checkpoints stay loadable.
    """
    def __init__(self, num_frequencies=128, t_scale=None, freqs=None, n_poly=5):
        super().__init__()
        self.n_poly = n_poly

        if freqs is not None:
            f = torch.as_tensor(np.asarray(freqs), dtype=torch.float64)
            self.legacy = False
            self.out_dim = 2 * len(f) + n_poly + 1
        else:
            self.legacy = True
            self.out_dim = 2 * num_frequencies + 1
            if t_scale is None:
                t_scale = 30 * 86400
                print("Warning: t_scale not provided to FourierEmbedding; using default 30 days")
            n_orbits = t_scale / 5520.0
            if num_frequencies == 64:
                f = torch.logspace(0, np.log10(512), 64).double()
            else:
                freqs_linear = torch.linspace(1.0, n_orbits + 10.0, num_frequencies - 16)
                freqs_log = torch.logspace(-1.0, 0.0, 16)
                f = torch.cat([freqs_log, freqs_linear]).double()

        self.register_buffer('freqs', f)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)                              # (N, 1)
        t = t.double()
        angles = t * self.freqs * (2.0 * np.pi)              # (N, F) float64
        if self.legacy:
            return torch.cat([torch.sin(angles), torch.cos(angles), t], dim=1)
        poly = torch.cat([t ** p for p in range(self.n_poly + 1)], dim=1)  # (N, n_poly+1)
        return torch.cat([torch.sin(angles), torch.cos(angles), poly], dim=1)


class KinematicPINN(nn.Module):
    """
    r_ecef(t) = Rz(-theta(t)) [ Linear(features) + MLP(features) ]

    The network represents the trajectory in the INERTIAL frame, where the
    r'' = a_J2(r) physics actually holds (the SPP/ODCP data is in the rotating
    ECEF frame: applying inertial gravity there is wrong by ~0.6 m/s^2, i.e.
    7% of g). theta(t) = rot_omega * t_norm is the Earth rotation angle since
    t_min; rot_omega = omega_earth * t_scale. The phase origin is irrelevant
    because two-body + J2 dynamics are invariant under rotations about z.

    The linear head over the Fourier/poly features carries the bulk of the
    trajectory (it is solved in closed form by the physics-regularized
    trainer); the MLP is an optional nonlinear correction. Both run on the
    float64 features; the linear head stays float64 end to end.
    """
    def __init__(self, num_frequencies=128, hidden=256, depth=5, t_scale=None,
                 freqs=None, n_poly=5, use_mlp=True, rot_omega=None):
        super().__init__()
        self.embedding = FourierEmbedding(num_frequencies=num_frequencies,
                                          t_scale=t_scale, freqs=freqs, n_poly=n_poly)
        self.register_buffer('rot_omega',
                             torch.tensor(float(rot_omega) if rot_omega is not None else 0.0,
                                          dtype=torch.float64))
        in_dim = self.embedding.out_dim
        self.legacy = self.embedding.legacy

        if self.legacy:
            self.linear = None
        else:
            self.linear = nn.Linear(in_dim, 3, bias=False).double()
            nn.init.zeros_(self.linear.weight)

        self.use_mlp = use_mlp
        if use_mlp:
            layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, 3)]
            self.net = nn.Sequential(*layers)
            self._init_weights()
        else:
            self.net = None

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # last layer near zero: the MLP starts as a small correction
        if not self.legacy:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        x = self.embedding(t)                                # float64
        if self.legacy:
            return self.net(x.float())
        out = self.linear(x)
        if self.net is not None:
            out = out + self.net(x.float()).double()
        # rotate inertial -> ECEF by -theta about z
        if float(self.rot_omega) != 0.0:
            th = self.rot_omega * t.double().reshape(-1)
            c, s = torch.cos(th), torch.sin(th)
            out = torch.stack([c*out[:, 0] + s*out[:, 1],
                               -s*out[:, 0] + c*out[:, 1],
                               out[:, 2]], dim=1)
        return out

    def forward_inertial(self, t):
        """Trajectory in the inertial frame (used by the physics residual)."""
        x = self.embedding(t)
        if self.legacy:
            return self.net(x.float())
        out = self.linear(x)
        if self.net is not None:
            out = out + self.net(x.float()).double()
        return out

    @classmethod
    def from_checkpoint(cls, ckpt):
        """Rebuild a model from a checkpoint dict saved by train_pinn."""
        state = ckpt['model_state']
        freqs = state.get('embedding.freqs')
        if 'linear.weight' in state:                         # new spectral format
            model = cls(freqs=freqs.numpy(),
                        n_poly=ckpt.get('n_poly', 5),
                        use_mlp=any(k.startswith('net.') for k in state),
                        rot_omega=float(state.get('rot_omega', 0.0)))
            model.load_state_dict(state)
        else:                                                # legacy SGD format
            model = cls(num_frequencies=len(freqs), t_scale=ckpt.get('t_scale'))
            # old checkpoints predate the rot_omega buffer (they were ECEF-frame,
            # i.e. no rotation); strict=False lets it default to 0.
            model.load_state_dict(state, strict=False)
        model.eval()
        return model
