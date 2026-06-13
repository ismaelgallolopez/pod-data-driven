# PDE Loss Scaling Fix - Summary

## Problem

The PDE residual was computed by converting both predicted and physics accelerations to physical units (L*/s²), but with different time denominators:
- Predicted acceleration: `a_pred_phys = a_pred / t_scale²` (where t_scale ~ 2.6e6 seconds)
- Physics acceleration: `a_physics_phys = a_physics / T_star²` (where T_star ~ 807 seconds)

**Magnitude mismatch**: `t_scale² ≈ 6.7e12` vs `T_star² ≈ 6.5e5` → ~7 orders of magnitude difference!

This caused:
- Physics loss (PDE term) to be numerically enormous
- Data loss to be completely overshadowed
- `pde_weight=1e-7` was compensating for the broken scaling
- Physics was not actually constraining the network

## Solution

Compute the PDE residual entirely in non-dimensional units (the units the network operates in):

### Before (incorrect):
```python
# Lines 139-151 in src/training/train_pinn.py
a_pred_phys = a_pred / (t_scale_tensor ** 2)  # [L*/s²]
a_physics_phys = a_physics_raw / (T_star_tensor ** 2)  # [L*/s²]
loss_pde = torch.mean((a_pred_phys - a_physics_phys) ** 2)
```

### After (correct):
```python
# Lines 139-151 in src/training/train_pinn.py
a_pred_nd = a_pred  # Already in [L*/t_norm²]
a_physics_raw = physics.get_j2_acceleration(r_pred)  # [L*/T*²]
# Scale physics to match prediction time scale
a_physics_scaled = a_physics_raw * ((t_scale_tensor / T_star_tensor) ** 2)
loss_pde = torch.mean((a_pred_nd - a_physics_scaled) ** 2)
```

The scaling factor `(t_scale / T_star)² = (2.6e6 / 807)² ≈ 1.0e7` brings physics acceleration to the correct magnitude **in the network's non-dimensional units**.

## Additional Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `pde_weight` | 1e-7 | 1e-4 | 100x stronger physics (was compensating for unit mismatch) |
| `data_only_epochs` | 500 | 200 | Earlier PDE engagement (fixed scaling allows physics sooner) |

## Files Modified

1. **src/training/train_pinn.py**
   - Lines 12, 139-151: Fixed PDE loss scaling
   - Lines 12: Updated default parameters

## Expected Results

After retraining from scratch:
- ✓ No t_scale warning from FourierEmbedding (because t_scale is passed in constructor)
- ✓ `pde/data` ratio at convergence should be **< 0.1** (physics is now reasonable, not dominant)
- ✓ PINN 3D RMS should be **< 4247 m** (baseline SPP vs ODCP) - physics should improve accuracy
- ✓ Smoother training dynamics (physics term is well-scaled)

## Verification

Run after training completes:
```bash
python scripts/verify_pde_fix.py
```

Or check manually:
```bash
python stage1_report.py  # Should show improved metrics
```

## Training Status

- Current: Epoch 49/2000 (in progress)
- Checkpoints: `data/processed/pinn_checkpoint.pth` (latest), `pinn_best.pth` (best)
- Final model: `data/processed/pinn_smoother.pth`

---

**Commit-ready changes**: The fixes in `train_pinn.py` are minimal, focused, and address the root cause of the numerical instability. No other files needed modification (stage1_report.py was already fixed separately for LVLH computation).
