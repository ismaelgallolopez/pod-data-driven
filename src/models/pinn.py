import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    """
    Maps t ∈ [0,1] to sinusoidal features spanning the relevant frequency range.

    Builds dual-grid embedding:
    - Linear grid: 1 to n_orbits+10 (orbital oscillations)
    - Logspace grid: 0.1 to 1.0 (secular drift)
    """
    def __init__(self, num_frequencies=128, t_scale=None):
        super().__init__()

        if t_scale is None:
            # Fallback: assume 30 days
            t_scale = 30 * 86400
            print("Warning: t_scale not provided to FourierEmbedding; using default 30 days")

        # Compute number of orbital cycles (92 min = 5520 s per orbit)
        n_orbits = t_scale / 5520.0

        # For num_frequencies=64 (old checkpoint), use legacy format
        if num_frequencies == 64:
            freqs = torch.logspace(0, np.log10(512), 64)
        else:
            # Dual-grid for 128+ frequencies
            # Linear grid for orbital band: 1 to n_orbits+10
            freqs_linear = torch.linspace(1.0, n_orbits + 10.0, num_frequencies - 16)
            # Logspace grid for secular drift: 0.1 to 1.0
            freqs_log = torch.logspace(-1.0, 0.0, 16)
            # Concatenate both grids
            freqs = torch.cat([freqs_log, freqs_linear])

        self.register_buffer('freqs', freqs)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)                          # (N, 1)
        angles = t * self.freqs * 2.0 * np.pi           # (N, F)
        return torch.cat([torch.sin(angles),
                          torch.cos(angles),
                          t], dim=1)                     # (N, 2F+1)


class KinematicPINN(nn.Module):
    def __init__(self, num_frequencies=128, hidden=256, depth=5, t_scale=None):
        super().__init__()
        in_dim = 2 * num_frequencies + 1

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]

        self.embedding = FourierEmbedding(num_frequencies=num_frequencies, t_scale=t_scale)
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        x = self.embedding(t)
        return self.net(x)