import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics

def train_pinn(t_train, r_train, epochs=2000, batch_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = OrbitPhysics()
    model = KinematicPINN().to(device)
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
            # Flatten is needed for SineLayer autograd compatibility
            v_pred = torch.autograd.grad(r_pred, batch_t, grad_outputs=torch.ones_like(r_pred),
                                         create_graph=True)
            a_pred = torch.autograd.grad(v_pred, batch_t, grad_outputs=torch.ones_like(v_pred),
                                         create_graph=True)
            
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