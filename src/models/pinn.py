import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    """
    Maps t ∈ [0,1] to sinusoidal features spanning the relevant frequency range.
    
    For CHAMP over 30 days:
      - Orbital period ~92 min → ~470 cycles in [0,1]
      - We need frequencies from 1 (secular drift) up to ~512 (sub-orbital)
    """
    def __init__(self, num_frequencies=64):
        super().__init__()
        freqs = torch.logspace(0, np.log10(512), num_frequencies)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)                          # (N, 1)
        angles = t * self.freqs * 2.0 * np.pi           # (N, F)
        return torch.cat([torch.sin(angles),
                          torch.cos(angles),
                          t], dim=1)                     # (N, 2F+1)


class KinematicPINN(nn.Module):
    def __init__(self, num_frequencies=64, hidden=256, depth=5):
        super().__init__()
        in_dim = 2 * num_frequencies + 1

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]

        self.embedding = FourierEmbedding(num_frequencies)
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