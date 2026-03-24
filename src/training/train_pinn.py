import os
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics


def train_pinn(t_train, r_train, epochs=2000, batch_size=4096, resume=True,
               checkpoint_dir='data/processed', save_freq=5,
               pde_weight=1e-4, data_only_epochs=200):

    warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*not currently supported on the DML backend.*", category=UserWarning)

    # ── Device selection (unchanged) ─────────────────────────────────────────
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
            try:
                if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                    device = torch.device('mps')
                    print('Using Apple MPS')
                else:
                    device = torch.device('cpu')
                    print('Using CPU')
            except Exception:
                device = torch.device('cpu')
                print('Using CPU')

    physics = OrbitPhysics()
    model   = KinematicPINN().to(device)
    print(f"Trainer: {device} | batch_size: {batch_size}")

    # ── Normalise BEFORE anything touches the network ────────────────────────
    # Time: map [t_min, t_max] → [0, 1]
    t_min   = t_train.min()
    t_max   = t_train.max()
    t_scale = (t_max - t_min).item()          # seconds spanned; keep as Python float
    t_norm  = ((t_train - t_min) / t_scale).float()   # in [0, 1]

    # Position: km → m → non-dimensional (÷ R_earth)
    r_nd = (r_train * 1000.0 / physics.L_star).float()   # ≈ 1.0–1.1 for LEO

    # Note: do NOT scale the physics by pde_scale (this explodes).
    # We'll convert network derivatives (w.r.t. t_norm) back to physical time units
    # using the chain rule when computing the PDE residual.

    # ── DataLoader ───────────────────────────────────────────────────────────
    dataset = TensorDataset(t_norm, r_nd)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Learning rate scheduler to reduce LR on plateau of data loss
    # Scheduler tries to rescue from plateaus before early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6, verbose=True
    )

    # Early stopping bookkeeping
    best_loss = float('inf')
    best_epoch = 0
    plateau_count = 0
    PATIENCE = 100
    MIN_DELTA = 1e-7

    # ── Checkpoint resume ────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'pinn_checkpoint.pth')
    start_epoch = 0

    if resume and os.path.exists(checkpoint_path):
        try:
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except TypeError:
                ckpt = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(ckpt.get('model_state', {}))
            model.to(device)
            optim_state = ckpt.get('optim_state', None)
            if optim_state:
                try:
                    optimizer.load_state_dict(optim_state)
                except Exception:
                    pass
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_loss_data = 0.0
        epoch_loss_pde  = 0.0

        for batch_t, batch_r in loader:
            # t ∈ [0,1],  r in non-dim units — network lives entirely in this space
            batch_t = batch_t.to(device).requires_grad_(True)
            batch_r = batch_r.to(device)

            optimizer.zero_grad()

            r_pred = model(batch_t)   # (N, 3), non-dim

            # ── First derivative: dr / dt_norm ───────────────────────────────
            v_list = []
            for i in range(3):
                gi = torch.autograd.grad(
                    r_pred[:, i], batch_t,
                    grad_outputs=torch.ones(batch_t.shape[0], device=device),
                    create_graph=True)[0]
                v_list.append(gi.unsqueeze(1))
            v_pred = torch.cat(v_list, dim=1)   # (N, 3)

            # ── Second derivative: d²r / dt_norm² ────────────────────────────
            a_list = []
            for i in range(3):
                gi2 = torch.autograd.grad(
                    v_pred[:, i], batch_t,
                    grad_outputs=torch.ones(batch_t.shape[0], device=device),
                    create_graph=True)[0]
                a_list.append(gi2.unsqueeze(1))
            a_pred = torch.cat(a_list, dim=1)   # (N, 3)  [non-dim / t_norm²]

            # ── Physics residual: convert derivatives back to physical time
            # a_pred is d²r / dt_norm² (units: L* / t_norm²)
            # Convert to physical seconds: a_pred_phys = a_pred / t_scale²  (L* / s²)
            t_scale_tensor = torch.tensor(t_scale, dtype=torch.float32, device=device)
            a_pred_phys = a_pred / (t_scale_tensor ** 2)

            # a_physics (from model) is in non-dim / T*² units -> convert to L*/s²
            a_physics_raw = physics.get_j2_acceleration(r_pred)   # [L* / T*²]
            T_star_tensor = torch.tensor(physics.T_star, dtype=torch.float32, device=device)
            a_physics_phys = a_physics_raw / (T_star_tensor ** 2)

            loss_data = torch.mean((r_pred - batch_r) ** 2)
            loss_pde = torch.mean((a_pred_phys - a_physics_phys) ** 2)

            # Two-phase weighting: data-only warmup, then enable PDE with pde_weight
            omega_pde = 0.0 if epoch < data_only_epochs else float(pde_weight)
            total_loss = loss_data + omega_pde * loss_pde

            total_loss.backward()

            # Clip gradients — prevents a single bad batch from diverging
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss_data += loss_data.item()
            epoch_loss_pde  += loss_pde.item()

        # Step scheduler on the data loss (avg per-batch)
        n = len(loader)
        avg_data = epoch_loss_data / n
        avg_pde = epoch_loss_pde / n
        scheduler.step(avg_data)

        # Early-stopping: track best avg_data and count plateaus
        if avg_data < best_loss - MIN_DELTA:
            best_loss = avg_data
            best_epoch = epoch
            plateau_count = 0
            # Save best model snapshot
            try:
                torch.save({
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    't_min':       t_min.item(),
                    't_scale':     t_scale,
                }, os.path.join(checkpoint_dir, 'pinn_best.pth'))
            except Exception:
                pass
        else:
            plateau_count += 1

        if epoch % 10 == 0:
            phase = "data-only" if epoch < data_only_epochs else "physics"
            print(f"Epoch {epoch:4d} [{phase}] | "
                  f"data={avg_data:.3e} | pde={avg_pde:.3e} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # Check early-stopping condition
        if plateau_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement since epoch {best_epoch}, best loss={best_loss:.3e})")
            break

        if (epoch + 1) % save_freq == 0:
            try:
                ckpt = {
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    't_min':       t_min.item(),
                    't_scale':     t_scale,
                }
                torch.save(ckpt, checkpoint_path)
            except Exception as e:
                print(f"Warning: checkpoint save failed: {e}")

    # ── Save final model ─────────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, 'pinn_smoother.pth')
    torch.save({
        'model_state': model.state_dict(),
        't_min':       t_min.item(),
        't_scale':     t_scale,
        'L_star':      physics.L_star,
        'T_star':      physics.T_star,
    }, final_path)
    # If we saved a best model earlier, load that instead of last epoch
    best_path = os.path.join(checkpoint_dir, 'pinn_best.pth')
    if os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state'])
            print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} (best data loss={best_loss:.3e})")
        except Exception:
            print("Warning: failed to load best model; returning final model")

    print(f"Final model saved to {final_path}")
    return model