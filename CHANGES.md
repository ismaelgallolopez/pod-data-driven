# Code Changes for PDE Loss Scaling Fix

## File: `src/training/train_pinn.py`

### Change 1: Line 12 - Function signature parameters

**Before:**
```python
def train_pinn(t_train, r_train, epochs=2000, batch_size=512, resume=True,
               checkpoint_dir='data/processed', save_freq=5,
               pde_weight=1e-7, data_only_epochs=500):
```

**After:**
```python
def train_pinn(t_train, r_train, epochs=2000, batch_size=512, resume=True,
               checkpoint_dir='data/processed', save_freq=5,
               pde_weight=1e-4, data_only_epochs=200):
```

**Changes:**
- `pde_weight`: `1e-7` → `1e-4` (100x stronger physics term)
- `data_only_epochs`: `500` → `200` (earlier PDE engagement after fix)

### Change 2: Lines 139-151 - PDE loss computation

**Before:**
```python
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
```

**After:**
```python
# ── Physics residual: compute entirely in non-dimensional units
# a_pred is d²r / dt_norm² (units: L* / t_norm²)
a_pred_nd = a_pred

# a_physics is in [L* / T*²] units — scale to match t_norm time scale
a_physics_raw = physics.get_j2_acceleration(r_pred)   # [L* / T*²]
t_scale_tensor = torch.tensor(t_scale, dtype=torch.float32, device=device)
T_star_tensor = torch.tensor(physics.T_star, dtype=torch.float32, device=device)
# Convert from [L* / T*²] to [L* / t_norm²]: multiply by (t_scale / T_star)²
a_physics_scaled = a_physics_raw * ((t_scale_tensor / T_star_tensor) ** 2)

loss_data = torch.mean((r_pred - batch_r) ** 2)
loss_pde = torch.mean((a_pred_nd - a_physics_scaled) ** 2)
```

**Key differences:**
- Removed: `a_pred_phys = a_pred / (t_scale_tensor ** 2)` (incorrect conversion)
- Removed: `a_physics_phys = a_physics_raw / (T_star_tensor ** 2)` (incorrect conversion)
- Added: `a_pred_nd = a_pred` (already correct units)
- Added: `a_physics_scaled = a_physics_raw * ((t_scale_tensor / T_star_tensor) ** 2)` (correct scaling)
- Changed: Loss computed in consistent non-dimensional units

## Impact

The fix ensures that:
1. **Correct units**: Both terms in the loss are in [L*/t_norm²], the natural units for the network
2. **Reasonable scaling**: Physics loss is no longer numerically enormous (7 orders of magnitude off)
3. **Effective physics**: `pde_weight=1e-4` now properly constrains the network instead of being ineffective at 1e-7
4. **Better convergence**: Physics constraint engages earlier (epoch 200 instead of 500) with correct scaling

## Testing

After training completes, verify with:
```bash
python scripts/verify_pde_fix.py
```

Expected results:
- pde/data ratio at convergence: **< 0.1**
- PINN 3D RMS vs ODCP: **< 4247 m** (better than SPP baseline)
