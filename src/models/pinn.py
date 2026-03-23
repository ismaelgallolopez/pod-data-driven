import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / 30, 
                                            np.sqrt(6 / self.in_features) / 30)

    def forward(self, x):
        return torch.sin(30 * self.linear(x))

class KinematicPINN(nn.Module):
    def __init__(self, hidden_features=256, hidden_layers=4):
        super().__init__()
        
        layers = []
        # Input: time (1D) -> Output: Position (3D)
        layers.append(SineLayer(1, hidden_features, is_first=True))
        
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features))
            
        layers.append(nn.Linear(hidden_features, 3))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # Time needs to be (N, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(t)