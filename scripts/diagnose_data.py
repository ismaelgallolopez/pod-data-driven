# scripts/diagnose_data.py
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

# ── Load (use weights_only when available; handle missing files) ──────────────
def _safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        # Older torch versions don't support weights_only kwarg
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
            return torch.load(path)
    except FileNotFoundError:
        return None

spp = _safe_torch_load('data/processed/noisy_spp.pt')
if spp is None:
    raise SystemExit('Missing data/processed/noisy_spp.pt — run sp3 extraction first')

truth = _safe_torch_load('data/processed/truth_odcp.pt')

t_spp   = spp[:, 0].numpy()
r_spp   = spp[:, 1:4].numpy()   # km

t_tru = None
r_tru = None
if truth is not None:
    t_tru = truth[:, 0].numpy()
    r_tru = truth[:, 1:4].numpy()

# ── 1. Basic sanity ───────────────────────────────────────────────────────────
print("=== SPP (noisy) ===")
print(f"  Samples      : {len(t_spp)}")
print(f"  Time span    : {t_spp[0]:.1f} → {t_spp[-1]:.1f} s  ({(t_spp[-1]-t_spp[0])/86400:.1f} days)")
print(f"  Time step    : {np.median(np.diff(t_spp)):.1f} s median")
print(f"  X range [km] : {r_spp[:,0].min():.1f} → {r_spp[:,0].max():.1f}")
print(f"  Y range [km] : {r_spp[:,1].min():.1f} → {r_spp[:,1].max():.1f}")
print(f"  Z range [km] : {r_spp[:,2].min():.1f} → {r_spp[:,2].max():.1f}")

r_mag_spp = np.linalg.norm(r_spp, axis=1)
print(f"  |r| mean [km]: {r_mag_spp.mean():.1f}  ±  {r_mag_spp.std():.1f}")
print(f"  |r| min/max  : {r_mag_spp.min():.1f} / {r_mag_spp.max():.1f}")

# CHAMP is ~450 km altitude → |r| ≈ 6828 km
R_earth = 6378.137
alt_spp = r_mag_spp - R_earth
print(f"  Altitude [km]: {alt_spp.mean():.1f}  ±  {alt_spp.std():.1f}  (expected ~450 km)")

if truth is not None:
    print("\n=== ODCP (truth) ===")
    print(f"  Samples      : {len(t_tru)}")
    print(f"  Time span    : {t_tru[0]:.1f} → {t_tru[-1]:.1f} s  ({(t_tru[-1]-t_tru[0])/86400:.1f} days)")
    print(f"  Time step    : {np.median(np.diff(t_tru)):.1f} s median")

    r_mag_tru = np.linalg.norm(r_tru, axis=1)
    alt_tru = r_mag_tru - R_earth
    print(f"  Altitude [km]: {alt_tru.mean():.1f}  ±  {alt_tru.std():.1f}")

# ── 2. Time continuity — are there unexpected gaps? ───────────────────────────
dt_spp = np.diff(t_spp)
gaps   = np.where(dt_spp > 60)[0]   # gaps > 60 s
print(f"\n=== Time gaps > 60 s in SPP ===")
print(f"  Count: {len(gaps)}")
for i in gaps[:10]:   # show first 10
    print(f"  t={t_spp[i]/3600:.2f} h → {t_spp[i+1]/3600:.2f} h  "
          f"(gap={dt_spp[i]/3600:.2f} h)")

# ── 3. Does |r| stay physical? ────────────────────────────────────────────────
bad = np.where((r_mag_spp < 6500) | (r_mag_spp > 7500))[0]
print(f"\n=== Outliers (|r| outside 6500–7500 km): {len(bad)} points ===")
if len(bad):
    print("  First few:", r_mag_spp[bad[:5]])

# ── 4. SPP vs truth positional difference (where epochs overlap) ──────────────
# Match by nearest timestamp
from scipy.interpolate import interp1d

if truth is not None and len(t_tru) > 10:
    # Interpolate truth onto SPP timestamps (only within truth span)
    mask = (t_spp >= t_tru[0]) & (t_spp <= t_tru[-1])
    t_common = t_spp[mask]

    interp_x = interp1d(t_tru, r_tru[:,0], kind='linear')
    interp_y = interp1d(t_tru, r_tru[:,1], kind='linear')
    interp_z = interp1d(t_tru, r_tru[:,2], kind='linear')

    r_tru_interp = np.column_stack([
        interp_x(t_common),
        interp_y(t_common),
        interp_z(t_common),
    ])
    r_spp_common = r_spp[mask]

    diff_m = (r_spp_common - r_tru_interp) * 1000   # km → m
    dist_m = np.linalg.norm(diff_m, axis=1)
    print(f"\n=== SPP vs ODCP 3D error ===")
    print(f"  Mean  : {dist_m.mean():.1f} m")
    print(f"  Median: {np.median(dist_m):.1f} m")
    print(f"  95th % : {np.percentile(dist_m, 95):.1f} m")
    print(f"  Max   : {dist_m.max():.1f} m")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
t_h = t_spp / 3600

for i, (ax, label) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
    ax.plot(t_h, r_spp[:, i], '.', ms=0.5, alpha=0.4, label='SPP (noisy)')
    if truth is not None and len(t_tru) > 10:
        ax.plot(t_tru/3600, r_tru[:, i], '-', lw=0.8, color='red', label='ODCP (truth)')
    ax.set_ylabel(f'{label} [km]')
    ax.legend(loc='upper right', markerscale=5)

axes[-1].set_xlabel('Time [hours]')
axes[0].set_title('ECI Position: SPP vs ODCP ground truth')
plt.tight_layout()
plt.savefig('data/processed/diag_position.png', dpi=150)
print("\nSaved: data/processed/diag_position.png")

# ── 6. |r| over time ─────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 3))
ax2.plot(t_h, r_mag_spp, '.', ms=0.5, alpha=0.4, label='SPP')
if truth is not None and len(t_tru) > 10:
    ax2.plot(t_tru/3600, r_mag_tru, '-', lw=0.8, color='red', label='ODCP')
ax2.axhline(R_earth + 400, color='gray', ls='--', lw=0.8, label='400 km alt')
ax2.axhline(R_earth + 500, color='gray', ls='--', lw=0.8, label='500 km alt')
ax2.set_ylabel('|r| [km]')
ax2.set_xlabel('Time [hours]')
ax2.set_title('Orbital radius over time')
ax2.legend(markerscale=5)
plt.tight_layout()
plt.savefig('data/processed/diag_radius.png', dpi=150)
print("Saved: data/processed/diag_radius.png")

plt.show()