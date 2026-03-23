import os
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics

def train_pinn(t_train, r_train, epochs=2000, batch_size=4096, resume=True,
               checkpoint_dir='data/processed', save_freq=5, pde_weight=1.0):
    # Suppress specific noisy warnings about torch.load and DML operator fallbacks
    warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*not currently supported on the DML backend.*", category=UserWarning)

    # Prefer available GPU-like accelerators in order: DirectML -> CUDA -> MPS -> CPU
    device = None
    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML:", device)
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        else:
            # Apple MPS (macOS) fallback
            try:
                if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                    device = torch.device('mps')
                    print('Using Apple MPS')
                else:
                    device = torch.device('cpu')
                    print('Using CPU')
            except Exception:
                device = torch.device('cpu')
                print('Using CPU')
    physics = OrbitPhysics()
    model = KinematicPINN().to(device)
    print(f"Trainer: {device} | batch_size: {batch_size}")
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

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'pinn_checkpoint.pth')

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        ckpt = None
        try:
            # Prefer safe weights-only load when supported
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except TypeError:
                # older torch doesn't accept weights_only kwarg
                ckpt = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            # Handle the new weights-only allowlist requirement by adding the needed global
            err = str(e)
            if 'Weights only load failed' in err or 'WeightsUnpickler' in err or 'allowlist' in err:
                try:
                    # allowlist the specific helper used by some saved checkpoints
                    from torch._utils import _rebuild_device_tensor_from_numpy
                    torch.serialization.add_safe_globals([_rebuild_device_tensor_from_numpy])
                    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                except Exception:
                    # Last resort: load without weights_only if the file is trusted
                    try:
                        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    except Exception as e2:
                        print(f"Failed to load checkpoint after allowlist: {e2}. Starting from scratch.")
                        ckpt = None
            else:
                print(f"Failed to load checkpoint: {e}. Starting from scratch.")

        if ckpt is not None:
            try:
                model.load_state_dict(ckpt.get('model_state', {}))
                # ensure model is on the chosen device
                model.to(device)
                optim_state = ckpt.get('optim_state', None)
                if optim_state is not None:
                    try:
                        optimizer.load_state_dict(optim_state)
                    except Exception:
                        pass
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to apply checkpoint: {e}. Starting from scratch.")
    
    # Prepare tensors and DataLoader for efficient GPU training when available
    t_train = t_train.float()
    r_train = r_train.float()
    pin_memory = True if getattr(device, 'type', None) == 'cuda' else False
    num_workers = 2 if pin_memory else 0

    dataset = TensorDataset(t_train, r_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=pin_memory, num_workers=num_workers)

    for epoch in range(start_epoch, epochs):
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
            
            # 5. Combine (expose PDE weight for tuning)
            total_loss = loss_data + pde_weight * loss_pde
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.2e}")

        # Periodically save checkpoint so we can resume later
        if (epoch + 1) % save_freq == 0:
            try:
                ckpt = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                }
                torch.save(ckpt, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path} (epoch {epoch})")
            except Exception as e:
                print(f"Warning: failed to save checkpoint: {e}")
            
    # Save final model and final checkpoint
    final_model_path = os.path.join(checkpoint_dir, 'pinn_smoother.pth')
    torch.save(model.state_dict(), final_model_path)
    try:
        torch.save({'epoch': epochs - 1, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, checkpoint_path)
    except Exception:
        pass
    print(f"Final model saved to {final_model_path}")
    return model