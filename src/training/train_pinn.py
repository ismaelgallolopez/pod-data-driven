import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics

def train_pinn(t_train, r_train, epochs=2000, batch_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = OrbitPhysics()
    model = KinematicPINN().to(device)
    # Fallback simple optimizer to avoid torch._dynamo import issues in some environments
    class SimpleSGD:
        def __init__(self, params, lr=1e-3):
            self.params = list(params)
            self.lr = lr
        def zero_grad(self):
            for p in self.params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        def step(self):
            for p in self.params:
                if p.grad is None:
                    continue
                p.data = p.data - self.lr * p.grad

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create DataLoader for batching
    dataset = TensorDataset(t_train, r_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_t, batch_r in loader:
            batch_t = batch_t.to(device).requires_grad_(True)
            batch_r = batch_r.to(device)
            
            optimizer.zero_grad()
            
            # Non-dimensionalize the target
            r_target_nd = physics.si_to_nd_pos(batch_r * 1000.0)

            # 1. Predict
            r_pred = model(batch_t)
            
            # 2. Physics Gradient (dr/dt, d2r/dt2)
            # Compute derivatives per coordinate to get vector-valued time derivatives
            v_components = []
            for i in range(r_pred.shape[1]):
                # r_pred[:, i] has shape (N,)
                grad_i = torch.autograd.grad(r_pred[:, i], batch_t,
                                             grad_outputs=torch.ones_like(r_pred[:, i]),
                                             create_graph=True)[0]
                v_components.append(grad_i.view(-1, 1))
            # Concatenate to (N,3)
            v_pred = torch.cat(v_components, dim=1)

            a_components = []
            for i in range(v_pred.shape[1]):
                grad2_i = torch.autograd.grad(v_pred[:, i], batch_t,
                                              grad_outputs=torch.ones_like(v_pred[:, i]),
                                              create_graph=True)[0]
                a_components.append(grad2_i.view(-1, 1))
            a_pred = torch.cat(a_components, dim=1)
            
            # 3. Physics Loss (Residual of J2)
            a_physics = physics.get_j2_acceleration(r_pred)
            loss_pde = torch.mean((a_pred - a_physics)**2)
            
            # 4. Data Loss
            loss_data = torch.mean((r_pred - r_target_nd)**2)
            
            # 5. Combine
            total_loss = loss_data + 1e-4 * loss_pde
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        if epoch % 100 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.2e}")
            
    return model