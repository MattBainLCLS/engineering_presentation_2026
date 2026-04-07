"""
generate_time_bandwidth.py
Visual demonstration of the time-bandwidth product:
broader spectral bandwidth → shorter laser pulse.

Physical parameters: central wavelength 1030 nm (Yb laser), time in fs.

Left panel  : 9 sinusoidal components at different frequencies, stacked
              vertically, all peaked at t=0. Colour-coded blue→red.
              X-axis zoomed to ±20 fs to show individual carrier oscillations.
Right top   : Narrow-bandwidth superposition (3 components) → ~30 fs pulse.
Right bottom: Broad-bandwidth superposition (9 components) → ~10 fs pulse.
              Both panels show the oscillating field + normalised envelope.
"""
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato', 'Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BORDER   = "#44546A"
NARROW_C = "#E04F39"   # SPIRITED — narrow bandwidth / long pulse
BROAD_C  = "#4298B5"   # SKY      — broad bandwidth / short pulse

# ── Physical parameters ───────────────────────────────────────────────────────
c       = 0.3            # µm / fs  (speed of light)
lam0    = 1.030          # µm  (1030 nm, Yb laser)
omega_0 = 2 * np.pi * c / lam0   # ≈ 1.827 rad/fs

# Component spacing: gives ~31 fs (N=3) and ~10 fs (N=9) FWHM pulses
d_omega = 0.06           # rad/fs  (~9.5 THz between adjacent components)

N_stack  = 9
N_narrow = 15   # Gaussian-weighted → ~28 fs pulse, minimal side lobes
N_broad  = 31   # Gaussian-weighted → ~10 fs pulse, clean Gaussian

SIGMA_NARROW = 1.5   # Gaussian width in component index units
SIGMA_BROAD  = 4.5

# High-resolution time array (dt << carrier period 3.44 fs)
t = np.linspace(-80.0, 80.0, 30000)   # femtoseconds

def make_components(n, t, sigma=None):
    """Return (components [n×t], normalised sum, omegas).
    If sigma is given, components are Gaussian-weighted to suppress side lobes."""
    half   = (n - 1) / 2.0
    ks     = np.arange(n) - half
    omegas = omega_0 + d_omega * ks
    if sigma is not None:
        weights = np.exp(-ks**2 / (2 * sigma**2))
    else:
        weights = np.ones(n)
    comps  = weights[:, None] * np.cos(np.outer(omegas, t))
    total  = comps.sum(axis=0)
    total /= total.max()
    return comps, total, omegas

comps_stack, _,           omegas_stack = make_components(N_stack,  t)
_,           sum_narrow,  _            = make_components(N_narrow, t, sigma=SIGMA_NARROW)
_,           sum_broad,   _            = make_components(N_broad,  t, sigma=SIGMA_BROAD)

# Analytic envelopes via Hilbert transform
env_narrow = np.abs(hilbert(sum_narrow))
env_broad  = np.abs(hilbert(sum_broad))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.4))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.32)

# ─────────────────────────────────────────────────────────────────────────────
# LEFT: stacked individual components (zoomed to ±20 fs)
# ─────────────────────────────────────────────────────────────────────────────
ax_l = fig.add_subplot(gs[0])

cmap        = plt.cm.coolwarm
offset_step = 2.6
t_left_lim  = (-20.0, 20.0)

top_offset  = (N_stack - 1) * offset_step
mid_offset  = top_offset / 2

for i in range(N_stack):
    offset = (N_stack - 1 - i) * offset_step
    color  = cmap(1 - i / (N_stack - 1))
    ax_l.plot(t, comps_stack[i] + offset, color=color, lw=1.1, alpha=0.9)

# t = 0 dashed line
ax_l.axvline(0, color='gray', lw=1.0, linestyle='--', alpha=0.65, zorder=0)

# Annotation to the right of the t=0 line, pointing at mid-height of the stack
ax_l.annotate("All waves\nin phase here",
              xy=(0, mid_offset),
              xytext=(10, mid_offset + offset_step * 1.5),
              fontsize=8.5, color='black', va='center', ha='left',
              arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                              connectionstyle='arc3,rad=-0.25'))

ax_l.set_xlim(*t_left_lim)
ax_l.set_ylim(-2.2, top_offset + 2.4)
ax_l.set_yticks([])
ax_l.set_xlabel("Time (fs)", fontsize=9, color=BORDER)
ax_l.set_title("Components at different frequencies\n"
               "— constructive interference only at $t = 0$",
               fontsize=9.5, color=BORDER, pad=7)
ax_l.spines[['top', 'right', 'left']].set_visible(False)
ax_l.tick_params(left=False, labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT: narrow vs broad bandwidth — field + envelope
# ─────────────────────────────────────────────────────────────────────────────
gs_r     = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.65)
t_r_lim  = (-70.0, 70.0)

def plot_pulse(ax, t, field, envelope, color, title_str):
    # Oscillating field — thin, semi-transparent
    ax.plot(t, field,    color=color, lw=0.7, alpha=0.35)
    # Normalised envelope — solid
    ax.plot(t, envelope,  color=color, lw=2.0, label='envelope')
    ax.plot(t, -envelope, color=color, lw=2.0)
    ax.fill_between(t, envelope, -envelope, alpha=0.12, color=color)

    # FWHM arrow on the envelope
    peak = envelope.max()
    idx  = np.where(envelope >= 0.5 * peak)[0]
    if len(idx) >= 2:
        tl, tr  = t[idx[0]], t[idx[-1]]
        fwhm_fs = tr - tl
        y_arr   = peak * 0.62
        ax.annotate('', xy=(tr, y_arr), xytext=(tl, y_arr),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.4,
                                   mutation_scale=9))
        ax.text((tl + tr) / 2, y_arr + 0.44,
                f"$\\Delta t \\approx {fwhm_fs:.0f}$ fs",
                ha='center', va='bottom', fontsize=8.5, color='black')

    ax.axvline(0, color='gray', lw=0.6, linestyle='--', alpha=0.45)
    ax.set_xlim(*t_r_lim)
    ax.set_ylim(-1.25, 1.45)
    ax.set_yticks([])
    ax.set_xlabel("Time (fs)", fontsize=8, color=BORDER)
    ax.set_title(title_str, fontsize=8.5, color=color, pad=18)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False, labelsize=7)

plot_pulse(fig.add_subplot(gs_r[0]),
           t, sum_narrow, env_narrow, NARROW_C,
           "Narrow bandwidth  (fewer components) → long pulse")

plot_pulse(fig.add_subplot(gs_r[1]),
           t, sum_broad, env_broad, BROAD_C,
           "Broad bandwidth  (many components) → short pulse")

# ── Save ──────────────────────────────────────────────────────────────────────
out = "figures/time_bandwidth.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
