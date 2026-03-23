import torch
import torch.optim as optim
from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics

def train_pinn(t_train, r_train, epochs=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = OrbitPhysics()
    model = KinematicPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Move data to device and ensure gradient tracking for time
    t_train = t_train.to(device).requires_grad_(True)
    r_train = r_train.to(device)
    
    # Non-Dimensionalize positions for the loss
    r_train_nd = physics.si_to_nd_pos(r_train * 1000.0) # SP3 is in km

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Position Prediction
        r_pred = model(t_train)
        
        # 2. Compute Derivatives via Autograd
        # Velocity v = dr/dt
        v_pred = torch.autograd.grad(r_pred, t_train, grad_outputs=torch.ones_like(r_pred),
                                     create_graph=True)
        # Acceleration a = dv/dt
        a_pred = torch.autograd.grad(v_pred, t_train, grad_outputs=torch.ones_like(v_pred),
                                     create_graph=True)
        
        # 3. Physics Loss (Residual of J2 Dynamics)
        a_physics = physics.get_j2_acceleration(r_pred)
        loss_pde = torch.mean((a_pred - a_physics)**2)
        
        # 4. Data Loss (Fit to noisy SPP)
        loss_data = torch.mean((r_pred - r_train_nd)**2)
        
        # 5. Total Weighted Loss
        total_loss = loss_data + 0.01 * loss_pde
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.2e} (Data: {loss_data.item():.2e}, PDE: {loss_pde.item():.2e})")
            
    return model