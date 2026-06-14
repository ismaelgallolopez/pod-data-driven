"""
Comprehensive analysis and metrics plots for the Stage-1 two-stage PINN smoother.

Panels produced:
  Fig 1 — Data overview: SPP positions, radius time-series, outlier map
  Fig 2 — Smoothing quality: 3D error time-series, SPP vs PINN vs truth
  Fig 3 — Error histograms: 3D-RMS and per-component (X/Y/Z)
  Fig 4 — LVLH frame analysis: radial / along-track / cross-track errors
  Fig 5 — Percentile sweep: CDF of 3D errors for SPP, PINN (and stage-1 only if present)
  Fig 6 — Spectral analysis: FFT of position errors (residual frequency content)
  Fig 7 — Physics residual: J2 PDE residual magnitude along the orbit

Run from the repo root:
    python scripts/analysis_plots.py
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

from src.models.pinn import KinematicPINN
from src.physics.orbits import OrbitPhysics


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path):
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.load(path, map_location="cpu")


def _iqr_mask(r):
    r_mag = torch.norm(r, dim=1)
    q25, q75 = r_mag.quantile(0.25), r_mag.quantile(0.75)
    iqr = q75 - q25
    return (r_mag >= q25 - 10 * iqr) & (r_mag <= q75 + 10 * iqr)


def _predict(model, t, t_min, t_scale, L_star):
    """Run PINN in float64 normalised time -> km."""
    with torch.no_grad():
        t_norm = (t.double() - float(t_min)) / float(t_scale)
        r_nd = model(t_norm)
    return (r_nd * L_star / 1000.0).float().numpy()   # km


def _interp_truth(t_np, odcp):
    t_tru = odcp[:, 0].numpy()
    r_tru = odcp[:, 1:4].numpy()
    inside = (t_np >= t_tru[0]) & (t_np <= t_tru[-1])
    r_out = np.full((len(t_np), 3), np.nan)
    for i in range(3):
        r_out[inside, i] = interp1d(t_tru, r_tru[:, i], kind="linear")(t_np[inside])
    return r_out, inside


def _lvlh(r_km, dt_s=30.0):
    """
    Project position error into the Local Vertical / Local Horizontal frame.
    Returns (radial, along_track, cross_track) unit vectors at each epoch.
    Velocity estimated by central differences on the reference orbit.
    """
    v = np.gradient(r_km, dt_s, axis=0)     # km/s, finite diff
    r_hat = r_km / np.linalg.norm(r_km, axis=1, keepdims=True)
    v_hat = v / np.linalg.norm(v, axis=1, keepdims=True)
    c_hat = np.cross(r_hat, v_hat)
    c_hat /= np.linalg.norm(c_hat, axis=1, keepdims=True)
    # recompute along-track orthogonal to both
    a_hat = np.cross(c_hat, r_hat)
    return r_hat, a_hat, c_hat


def _j2_residual_nd(model, t, t_min, t_scale):
    """
    ||r''_pred - a_J2(r_pred)||  in non-dimensional units.
    Uses autograd on the inertial-frame output.
    """
    phys = OrbitPhysics()
    s2 = (float(t_scale) / phys.T_star) ** 2

    t_norm = ((t.double() - float(t_min)) / float(t_scale)).requires_grad_(False)

    # sample a subset for speed
    idx = torch.linspace(0, len(t_norm) - 1, min(2000, len(t_norm))).long()
    t_sub = t_norm[idx].detach().requires_grad_(True)

    r_inertial = model.forward_inertial(t_sub)               # (N,3)
    # first derivative via autograd
    v = torch.autograd.grad(r_inertial.sum(), t_sub,
                            create_graph=True)[0]             # scalar grad trick
    # actually grad each component
    r_out = model.forward_inertial(t_sub)
    drdt = []
    for c in range(3):
        g = torch.autograd.grad(r_out[:, c].sum(), t_sub,
                                create_graph=True, retain_graph=True)[0]
        drdt.append(g)
    drdt = torch.stack(drdt, dim=1)                          # (N,3)
    d2rdt2 = []
    for c in range(3):
        g2 = torch.autograd.grad(drdt[:, c].sum(), t_sub,
                                 retain_graph=(c < 2))[0]
        d2rdt2.append(g2)
    d2rdt2 = torch.stack(d2rdt2, dim=1).detach()            # (N,3)

    a_j2 = phys.get_j2_acceleration(r_out.detach().float().double()) * s2
    res = (d2rdt2 - a_j2).float()
    # to SI (m/s²): multiply by L_star / T_star^2 — but keep relative units here
    return t[idx].numpy() / 3600.0, res.norm(dim=1).numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    spp = _load("data/processed/noisy_spp.pt")
    odcp = _load("data/processed/truth_odcp.pt")
    ckpt = _load("data/processed/pinn_best.pth")

    t_all = spp[:, 0]
    r_all = spp[:, 1:4]

    mask = _iqr_mask(r_all)
    n_out = (~mask).sum().item()
    t = t_all[mask]
    r_spp = r_all[mask].numpy()     # km

    t_np = t.numpy()
    t_h = t_np / 3600.0

    print(f"  SPP total:   {len(t_all):,} epochs")
    print(f"  IQR removed: {n_out:,} ({100*n_out/len(t_all):.1f}%)  outliers")
    print(f"  Clean:       {len(t):,} epochs  ({t_h[-1]-t_h[0]:.1f} h arc)")

    # ── load model and predict ─────────────────────────────────────────────────
    print("Predicting with PINN smoother …")
    model = KinematicPINN.from_checkpoint(ckpt)
    t_min  = ckpt["t_min"]
    t_scale = ckpt["t_scale"]
    L_star = ckpt.get("L_star", 6378137.0)

    r_pred = _predict(model, t, t_min, t_scale, L_star)     # km

    # ── interpolate truth ─────────────────────────────────────────────────────
    r_truth, inside = _interp_truth(t_np, odcp)             # km, mask

    err_spp  = (r_spp  - r_truth) * 1000.0                  # m
    err_pinn = (r_pred - r_truth) * 1000.0                  # m

    err3d_spp  = np.linalg.norm(err_spp,  axis=1)
    err3d_pinn = np.linalg.norm(err_pinn, axis=1)

    # valid where truth available
    vi = inside
    rms_spp  = np.sqrt(np.nanmean(err3d_spp[vi] ** 2))
    rms_pinn = np.sqrt(np.nanmean(err3d_pinn[vi] ** 2))
    med_pinn = np.nanmedian(err3d_pinn[vi])
    p99_pinn = np.nanpercentile(err3d_pinn[vi], 99)

    print(f"\n  SPP   3D RMS vs truth : {rms_spp:.1f} m")
    print(f"  PINN  3D RMS vs truth : {rms_pinn:.1f} m")
    print(f"  PINN  median          : {med_pinn:.1f} m")
    print(f"  PINN  p99             : {p99_pinn:.1f} m")
    print(f"  Improvement           : {rms_spp/rms_pinn:.1f}×")

    # ── LVLH projection ───────────────────────────────────────────────────────
    dt_s = float(np.median(np.diff(t_np)))
    r_hat, a_hat, c_hat = _lvlh(r_truth, dt_s=dt_s)

    def project_lvlh(err_m):
        r_c = np.einsum("ni,ni->n", err_m / 1000.0, r_hat) * 1000.0   # m
        a_c = np.einsum("ni,ni->n", err_m / 1000.0, a_hat) * 1000.0
        c_c = np.einsum("ni,ni->n", err_m / 1000.0, c_hat) * 1000.0
        return r_c, a_c, c_c

    rad_spp,  alt_spp,  ct_spp  = project_lvlh(err_spp)
    rad_pinn, alt_pinn, ct_pinn = project_lvlh(err_pinn)

    def lvlh_rms(v):
        return np.sqrt(np.nanmean(v[vi] ** 2))

    print(f"\n  LVLH RMS (PINN, m):")
    print(f"    Radial       : {lvlh_rms(rad_pinn):.1f}")
    print(f"    Along-track  : {lvlh_rms(alt_pinn):.1f}")
    print(f"    Cross-track  : {lvlh_rms(ct_pinn):.1f}")

    # ── figure style ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.0,
    })
    coord_labels = ["X", "Y", "Z"]
    colors = dict(spp="#e07b54", pinn="#4b8bbe", truth="#2ecc71")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 1 — Data overview
    # ══════════════════════════════════════════════════════════════════════════
    print("\nFig 1 — Data overview …")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    r_mag_all = np.linalg.norm(r_all.numpy(), axis=1)
    r_mag_clean = np.linalg.norm(r_spp, axis=1)
    r_mag_truth = np.linalg.norm(odcp[:, 1:4].numpy(), axis=1)

    t_all_h = t_all.numpy() / 3600.0
    t_odcp_h = odcp[:, 0].numpy() / 3600.0

    # panel 0: SPP outlier scatter
    ax = axes[0]
    ax.scatter(t_all_h[~mask.numpy()], r_mag_all[~mask.numpy()],
               s=1, c="#e07b54", alpha=0.5, label=f"Outliers ({n_out:,})")
    ax.scatter(t_h, r_mag_clean,
               s=0.3, c="#4b8bbe", alpha=0.3, label=f"Clean SPP ({len(t):,})")
    ax.plot(t_odcp_h, r_mag_truth, "g-", lw=0.8, label="ODCP truth", zorder=3)
    ax.set_ylabel("|r| [km]")
    ax.legend(loc="upper right", markerscale=4, fontsize=9)
    ax.set_title("Orbital Radius — raw SPP vs ODCP truth", fontweight="bold")

    # panel 1: X component time-series (sampled)
    ax = axes[1]
    step = max(1, len(t) // 3000)
    ax.plot(t_h[::step], r_spp[::step, 0], ".", ms=0.8, c=colors["spp"],
            alpha=0.4, label="SPP")
    ax.plot(t_h[::step], r_pred[::step, 0], "-", lw=0.8, c=colors["pinn"],
            label="PINN")
    ax.plot(t_odcp_h, odcp[:, 1].numpy(), "-", lw=0.6, c=colors["truth"],
            label="ODCP truth", zorder=3)
    ax.set_ylabel("X [km]")
    ax.legend(loc="upper right", markerscale=6, fontsize=9)

    # panel 2: outlier fraction in rolling 6-h windows
    ax = axes[2]
    win_h = 6.0
    edges = np.arange(t_h[0], t_h[-1] + win_h, win_h)
    out_frac = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        in_win_all   = ((t_all_h >= e0) & (t_all_h < e1)).sum()
        in_win_clean = ((t_h      >= e0) & (t_h      < e1)).sum()
        out_frac.append(0.0 if in_win_all == 0
                        else (in_win_all - in_win_clean) / in_win_all * 100)
    ax.bar(edges[:-1], out_frac, width=win_h * 0.9, align="edge",
           color="#e07b54", alpha=0.7)
    ax.axhline(np.mean(out_frac), color="k", ls="--", lw=1,
               label=f"Mean {np.mean(out_frac):.1f}%")
    ax.set_ylabel("Outlier fraction [%]")
    ax.set_xlabel("Time [hours]")
    ax.legend(fontsize=9)
    ax.set_title("Outlier fraction per 6-hour window", fontweight="bold")

    fig.suptitle("Fig 1 — Data Overview (CHAMP SPP)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = out_dir / "fig1_data_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 2 — Smoothing quality: error time-series X/Y/Z
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 2 — Smoothing quality time-series …")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 3D error
    ax = axes[0]
    ax.plot(t_h, err3d_spp,  ".", ms=0.5, c=colors["spp"],  alpha=0.25,
            label=f"SPP  RMS={rms_spp:.0f} m")
    ax.plot(t_h, err3d_pinn, "-", lw=0.7, c=colors["pinn"],
            label=f"PINN RMS={rms_pinn:.0f} m  med={med_pinn:.0f} m")
    ax.set_ylabel("3D error [m]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.legend(fontsize=9, markerscale=4)
    ax.set_title(f"3D position error vs ODCP truth  (improvement {rms_spp/rms_pinn:.0f}×)",
                 fontweight="bold")

    # per-component
    for i, (ax, coord) in enumerate(zip(axes[1:], coord_labels)):
        step = max(1, len(t) // 4000)
        ax.plot(t_h[::step], err_spp[::step, i],  ".", ms=0.7,
                c=colors["spp"], alpha=0.3,
                label=f"SPP  RMS={np.sqrt(np.nanmean(err_spp[vi,i]**2)):.0f} m")
        ax.plot(t_h[::step], err_pinn[::step, i], "-", lw=0.8,
                c=colors["pinn"],
                label=f"PINN RMS={np.sqrt(np.nanmean(err_pinn[vi,i]**2)):.0f} m")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"Δ{coord} [m]")
        ax.legend(fontsize=9, markerscale=4)

    axes[-1].set_xlabel("Time [hours]")
    fig.suptitle("Fig 2 — Smoothing Quality: PINN vs raw SPP (vs ODCP truth)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = out_dir / "fig2_smoothing_quality.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 3 — Error histograms
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 3 — Error histograms …")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _hist(ax, data_spp, data_pinn, title, xlabel, clip=None):
        d_s = data_spp[vi][np.isfinite(data_spp[vi])]
        d_p = data_pinn[vi][np.isfinite(data_pinn[vi])]
        if clip is not None:
            d_s = np.clip(d_s, -clip, clip)
            d_p = np.clip(d_p, -clip, clip)
        rng = (min(d_s.min(), d_p.min()), max(d_s.max(), d_p.max()))
        bins = np.linspace(rng[0], rng[1], 120)
        ax.hist(d_s, bins=bins, color=colors["spp"],  alpha=0.5,
                label=f"SPP  σ={d_s.std():.0f} m")
        ax.hist(d_p, bins=bins, color=colors["pinn"], alpha=0.7,
                label=f"PINN σ={d_p.std():.0f} m")
        ax.axvline(np.median(d_p), color=colors["pinn"], ls="--", lw=1.5,
                   label=f"PINN med={np.median(d_p):.0f} m")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    _hist(axes[0, 0], err3d_spp, err3d_pinn, "3D position error", "error [m]", clip=15000)
    _hist(axes[0, 1], err_spp[:, 0], err_pinn[:, 0], "ΔX error", "ΔX [m]",   clip=8000)
    _hist(axes[1, 0], err_spp[:, 1], err_pinn[:, 1], "ΔY error", "ΔY [m]",   clip=8000)
    _hist(axes[1, 1], err_spp[:, 2], err_pinn[:, 2], "ΔZ error", "ΔZ [m]",   clip=8000)

    fig.suptitle("Fig 3 — Error Distributions (SPP vs PINN vs ODCP truth)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "fig3_error_histograms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 4 — LVLH frame errors
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 4 — LVLH frame errors …")
    lvlh_labels  = ["Radial", "Along-track", "Cross-track"]
    lvlh_spp  = [rad_spp,  alt_spp,  ct_spp]
    lvlh_pinn = [rad_pinn, alt_pinn, ct_pinn]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    step = max(1, len(t) // 3000)
    for i, (label, ls, lp) in enumerate(zip(lvlh_labels, lvlh_spp, lvlh_pinn)):
        rms_s = lvlh_rms(ls)
        rms_p = lvlh_rms(lp)

        # time series
        ax = axes[i, 0]
        ax.plot(t_h[::step], ls[::step], ".", ms=0.7, c=colors["spp"], alpha=0.25,
                label=f"SPP  RMS={rms_s:.0f} m")
        ax.plot(t_h[::step], lp[::step], "-", lw=0.7, c=colors["pinn"],
                label=f"PINN RMS={rms_p:.0f} m")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"{label} [m]")
        ax.legend(fontsize=8, markerscale=4)
        if i == 2:
            ax.set_xlabel("Time [hours]")
        ax.set_title(f"{label} — time series", fontweight="bold")

        # histogram
        ax = axes[i, 1]
        d_s = np.clip(ls[vi][np.isfinite(ls[vi])], -10000, 10000)
        d_p = lp[vi][np.isfinite(lp[vi])]
        bins = np.linspace(min(d_s.min(), d_p.min()), max(d_s.max(), d_p.max()), 100)
        ax.hist(d_s, bins=bins, color=colors["spp"], alpha=0.5,
                label=f"SPP σ={d_s.std():.0f} m")
        ax.hist(d_p, bins=bins, color=colors["pinn"], alpha=0.7,
                label=f"PINN σ={d_p.std():.0f} m")
        ax.set_xlabel(f"{label} error [m]")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_title(f"{label} — histogram", fontweight="bold")

    fig.suptitle("Fig 4 — LVLH Frame Errors (Radial / Along-track / Cross-track)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "fig4_lvlh_errors.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 5 — CDF of 3D errors
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 5 — Error CDF …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (lo, hi, label) in zip(
        axes,
        [(0, 20000, "Full range (0–20 km)"), (0, 500, "Zoom: 0–500 m (PINN)")],
    ):
        for name, data, c in [
            ("SPP",  err3d_spp[vi],  colors["spp"]),
            ("PINN", err3d_pinn[vi], colors["pinn"]),
        ]:
            d = np.sort(data[np.isfinite(data)])
            ax.plot(d, np.linspace(0, 1, len(d)), color=c, lw=1.5, label=name)
        ax.axhline(0.50, color="k", ls=":", lw=0.8)
        ax.axhline(0.95, color="k", ls=":", lw=0.8)
        ax.text(hi * 0.98, 0.51, "p50", ha="right", fontsize=8)
        ax.text(hi * 0.98, 0.96, "p95", ha="right", fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel("3D error [m]")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(label, fontweight="bold")
        ax.legend()

    # annotate key percentiles
    for pct, label_sfx in [(50, "p50"), (95, "p95"), (99, "p99")]:
        val = np.nanpercentile(err3d_pinn[vi], pct)
        axes[1].axvline(val, ls="--", color=colors["pinn"], lw=0.8, alpha=0.7)
        axes[1].text(val + 3, 0.05 + pct / 400,
                     f"PINN {label_sfx}={val:.0f} m", fontsize=7,
                     color=colors["pinn"])

    fig.suptitle("Fig 5 — Cumulative Distribution of 3D Position Error",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "fig5_error_cdf.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 6 — Spectral analysis of PINN residual error
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 6 — Spectral analysis …")
    # Use PINN error on a uniform-ish grid (truth available everywhere)
    # Interpolate PINN error onto uniform time grid for FFT
    t_uniform = np.linspace(t_np[vi][0], t_np[vi][-1], 8192)
    err_x_uniform = interp1d(t_np[vi], err_pinn[vi, 0], kind="linear")(t_uniform)
    err_y_uniform = interp1d(t_np[vi], err_pinn[vi, 1], kind="linear")(t_uniform)
    err_z_uniform = interp1d(t_np[vi], err_pinn[vi, 2], kind="linear")(t_uniform)

    fs = 1.0 / (t_uniform[1] - t_uniform[0])    # Hz
    freqs_hz = rfftfreq(len(t_uniform), d=1.0 / fs)
    freqs_per_orbit = freqs_hz / (1.0 / 5520.0)  # orbital cycles

    def psd(x):
        F = rfft(x - x.mean())
        return np.abs(F) ** 2 / len(x)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, err_u, coord in zip(axes,
                                [err_x_uniform, err_y_uniform, err_z_uniform],
                                coord_labels):
        ax.semilogy(freqs_per_orbit, psd(err_u), lw=0.8, c=colors["pinn"])
        ax.set_ylabel(f"PSD Δ{coord} [m²/cycle]")
        ax.set_title(f"Δ{coord} residual power spectrum", fontweight="bold")
        # mark orbital harmonics
        for n in range(1, 5):
            ax.axvline(n, color="red", ls="--", lw=0.6, alpha=0.7)
        ax.text(1.02, ax.get_ylim()[1] * 0.5, "n=1", color="red", fontsize=7)

    axes[-1].set_xlabel("Frequency [orbital cycles]")
    axes[-1].set_xlim(0, 10)
    fig.suptitle("Fig 6 — Spectral Content of PINN Position Error\n"
                 "(red dashed = orbital harmonics)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "fig6_spectral_analysis.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 7 — J2 physics residual along the orbit
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 7 — J2 physics residual …")
    try:
        t_res_h, res_nd = _j2_residual_nd(model, t, t_min, t_scale)
        phys = OrbitPhysics()
        # convert to m/s² (non-dim accel * L_star/T_star²)
        res_si = res_nd * phys.L_star / phys.T_star ** 2

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogy(t_res_h, res_si, ".", ms=2, c=colors["pinn"], alpha=0.6)
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel("||r'' − a_J2|| [m/s²]")
        ax.set_title("Fig 7 — J2 Physics Residual Magnitude (lower = better physics fit)",
                     fontweight="bold")
        ax.axhline(1e-4, color="k", ls="--", lw=0.8,
                   label="0.1 mm/s² reference")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = out_dir / "fig7_physics_residual.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved {p}")
        plt.close(fig)
    except Exception as exc:
        print(f"  Skipped Fig 7 (autograd on linear model not supported): {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fig 8 — Summary table (text figure)
    # ══════════════════════════════════════════════════════════════════════════
    print("Fig 8 — Summary table …")
    rows = [
        ("Metric", "Raw SPP", "PINN Smoother", "Improvement"),
        ("3D RMS [m]", f"{rms_spp:.0f}", f"{rms_pinn:.1f}", f"{rms_spp/rms_pinn:.0f}×"),
        ("3D Median [m]", "—", f"{med_pinn:.1f}", "—"),
        ("3D p99 [m]", "—", f"{p99_pinn:.1f}", "—"),
        ("ΔX RMS [m]",
         f"{np.sqrt(np.nanmean(err_spp[vi,0]**2)):.0f}",
         f"{np.sqrt(np.nanmean(err_pinn[vi,0]**2)):.1f}", "—"),
        ("ΔY RMS [m]",
         f"{np.sqrt(np.nanmean(err_spp[vi,1]**2)):.0f}",
         f"{np.sqrt(np.nanmean(err_pinn[vi,1]**2)):.1f}", "—"),
        ("ΔZ RMS [m]",
         f"{np.sqrt(np.nanmean(err_spp[vi,2]**2)):.0f}",
         f"{np.sqrt(np.nanmean(err_pinn[vi,2]**2)):.1f}", "—"),
        ("Radial RMS [m]",
         f"{lvlh_rms(rad_spp):.0f}",
         f"{lvlh_rms(rad_pinn):.1f}", "—"),
        ("Along-track RMS [m]",
         f"{lvlh_rms(alt_spp):.0f}",
         f"{lvlh_rms(alt_pinn):.1f}", "—"),
        ("Cross-track RMS [m]",
         f"{lvlh_rms(ct_spp):.0f}",
         f"{lvlh_rms(ct_pinn):.1f}", "—"),
        ("Arc length", f"{(t_np[-1]-t_np[0])/86400:.1f} days", "", ""),
        ("Clean epochs", f"{len(t):,}", "", ""),
        ("Outlier rate", f"{100*n_out/len(t_all):.1f}%", "", ""),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r_i, c_i), cell in tbl.get_celld().items():
        if r_i == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_i % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        if c_i == 2 and r_i > 0:
            cell.set_text_props(color="#1a5276", fontweight="bold")

    fig.suptitle("Fig 8 — Stage-1 PINN Smoother: Results Summary\n"
                 "(CHAMP 2005, SPP → PINN → vs ODCP truth)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = out_dir / "fig8_summary_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig)

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("All figures saved to data/processed/")
    print(f"  fig1_data_overview.png       — data quality & outlier map")
    print(f"  fig2_smoothing_quality.png   — error time-series (3D + X/Y/Z)")
    print(f"  fig3_error_histograms.png    — error distributions")
    print(f"  fig4_lvlh_errors.png         — radial / along / cross errors")
    print(f"  fig5_error_cdf.png           — cumulative error distributions")
    print(f"  fig6_spectral_analysis.png   — residual frequency content")
    print(f"  fig7_physics_residual.png    — J2 PDE residual")
    print(f"  fig8_summary_table.png       — results summary table")
    print("=" * 65)
    print(f"\nKey result: PINN RMS {rms_pinn:.1f} m vs SPP {rms_spp:.0f} m "
          f"({rms_spp/rms_pinn:.0f}× improvement)")


if __name__ == "__main__":
    main()
