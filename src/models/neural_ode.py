import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint_adjoint as odeint
except Exception:
    odeint = None


class VectorField(nn.Module):
    """Vector field network for Neural ODEs. Input z = [r, v] concatenated.

    Produces dz/dt = f_theta(z, t).
    """
    def __init__(self, dim=6, hidden=128, layers=3):
        super().__init__()
        net = []
        in_dim = dim
        for _ in range(layers):
            net.append(nn.Linear(in_dim, hidden))
            net.append(nn.GELU())
            in_dim = hidden
        net.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*net)

    def forward(self, t, z):
        # z: (batch, dim) or (dim,) depending on integrator
        return self.net(z)


def integrate_trajectory(vector_field, z0, t_span, method='rk4', atol=1e-6, rtol=1e-6):
    """Integrate the ODE dz/dt = f(t, z) from t_span[0] to t_span[-1].

    Args:
        vector_field: instance of VectorField
        z0: initial state tensor (batch, dim) or (dim,)
        t_span: 1D tensor of times at which to evaluate the solution
    Returns:
        z_t: tensor of shape (len(t_span), batch, dim) if batch present
    """
    if odeint is None:
        raise RuntimeError("torchdiffeq is not installed. Install via `pip install torchdiffeq` to use Neural ODE integration.")

    # odeint expects func(t, z) signature
    z_t = odeint(vector_field, z0, t_span, atol=atol, rtol=rtol, method=method)
    return z_t


# Simple training sketch (to be integrated into the project's training loops)
class NeuralODEModel(nn.Module):
    def __init__(self, dim=6, hidden=128, layers=3):
        super().__init__()
        self.func = VectorField(dim=dim, hidden=hidden, layers=layers)

    def forward(self, z0, t_span):
        return integrate_trajectory(self.func, z0, t_span)


if __name__ == '__main__':
    print('neural_ode module: vector field + integrate_trajectory helper (requires torchdiffeq)')
