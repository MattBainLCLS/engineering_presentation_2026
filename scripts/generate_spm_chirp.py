"""
generate_spm_chirp.py
Two-panel figure for the "SPM creates a chirped pulse" slide:
  Left  — SPM instantaneous frequency (I(t) + δω(t)) with red/blue region shading
  Right — Stacked: transform-limited pulse (top) vs positively-chirped pulse (bottom)
          Chirped pulse field is colour-coded red → blue to match the δω annotation.
Output: figures/spm_chirp.svg
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

BORDER = "#44546A"
RED_C  = "#C62828"
BLUE_C = "#1565C0"

# ── SPM physics ───────────────────────────────────────────────────────────────
t  = np.linspace(-2.5, 2.5, 1000)
I  = np.exp(-2 * np.log(2) * t**2)
B  = 5 * np.pi
dI = np.gradient(I, t)
dw = -B * dI   # δω < 0 = red-shift (leading edge), δω > 0 = blue-shift (trailing)

# ── Pulse parameters ──────────────────────────────────────────────────────────
t_pulse = np.linspace(-5.5, 5.5, 8000)
omega_0 = 10.0     # normalised carrier (~4 cycles per FWHM_TL)
tau_TL  = 0.75     # TL pulse FWHM
tau_ch  = 2.8      # Chirped pulse FWHM (≈ 3.7× stretch from GDD/SPM)
beta    = 0.85     # Positive chirp rate (ω_inst = ω₀ + 2β·t)

A_TL  = np.exp(-2 * np.log(2) * t_pulse**2 / tau_TL**2)
E_TL  = A_TL * np.cos(omega_0 * t_pulse)

A_ch  = np.exp(-2 * np.log(2) * t_pulse**2 / tau_ch**2)
E_ch  = A_ch * np.cos(omega_0 * t_pulse + beta * t_pulse**2)
w_inst = omega_0 + 2 * beta * t_pulse   # positive: reds at t<0, blues at t>0

# Colour map: low freq → red, high freq → blue
cmap_rb = LinearSegmentedColormap.from_list('rb', [RED_C, '#9C27B0', BLUE_C])

# Coloured LineCollection for chirped pulse field
sig_mask = A_ch > 0.03
t_m   = t_pulse[sig_mask]
E_m   = E_ch[sig_mask]
w_m   = w_inst[sig_mask]
w_norm = (w_m - w_m.min()) / (w_m.max() - w_m.min())   # 0=red, 1=blue

pts  = np.array([t_m, E_m]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
seg_colors = cmap_rb(w_norm[:-1])

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 4.0))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.38)

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT: SPM concept — I(t) + δω(t)
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()

# Tinted background regions
ax1.axvspan(-2.5, 0, alpha=0.07, color=RED_C,  zorder=0)
ax1.axvspan(0, 2.5,  alpha=0.07, color=BLUE_C, zorder=0)

# Intensity envelope
ax1.fill_between(t, I, alpha=0.18, color='#E65100')
ax1.plot(t, I, color='#E65100', lw=2.2, label="$I(t)$")
ax1.set_ylabel("Intensity  $I(t)$", fontsize=8, color='#E65100')
ax1.set_ylim(-0.08, 1.72)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["0", "$I_0$"], fontsize=8, color='#E65100')
ax1.tick_params(axis='y', colors='#E65100', length=3)
ax1.spines['left'].set_color('#E65100')
ax1.spines[['top', 'right', 'bottom']].set_visible(False)

# δω curve
peak_dw = np.max(np.abs(dw))
ax2.plot(t, dw, color='#1565C0', lw=2.2, linestyle='--')
ax2.axhline(0, color='#1565C0', lw=0.6, linestyle=':', alpha=0.5)
ax2.set_ylabel(r"$\delta\omega(t)$", fontsize=8, color='#1565C0')
ax2.set_ylim(-peak_dw * 1.72, peak_dw * 1.72)
ax2.set_yticks([-peak_dw, 0, peak_dw])
ax2.set_yticklabels(["blue\nshift", "0", "red\nshift"], fontsize=7.5, color='#1565C0')
ax2.tick_params(axis='y', colors='#1565C0', length=3)
ax2.spines['right'].set_color('#1565C0')
ax2.spines[['top', 'left', 'bottom']].set_visible(False)

# Annotations
ax1.annotate("new low-$\\omega$\n(red-shifted)",
             xy=(-0.65, 0.5), xytext=(-2.3, 1.38),
             fontsize=7.5, color=RED_C, ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color=RED_C, lw=1.1,
                             connectionstyle='arc3,rad=0.25'))
ax1.annotate("new high-$\\omega$\n(blue-shifted)",
             xy=(0.65, 0.5), xytext=(2.2, 1.38),
             fontsize=7.5, color=BLUE_C, ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color=BLUE_C, lw=1.1,
                             connectionstyle='arc3,rad=-0.25'))

# Equation
ax1.text(0, 1.64,
         r"$\delta\omega(t)=-\frac{n_2\omega_0 L}{c}\frac{dI}{dt}$",
         ha='center', va='bottom', fontsize=8, color=BORDER,
         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', lw=0.7))

ax1.set_xlim(-2.5, 2.5)
ax1.set_xticks([])
ax1.set_xlabel("Time  →", fontsize=8, color=BORDER)
ax1.spines['bottom'].set_visible(True)
ax1.spines['bottom'].set_color('#cccccc')
ax1.tick_params(axis='x', bottom=False)
ax1.set_title("SPM: Instantaneous Frequency Shift", fontsize=9.5, color=BORDER, pad=6)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT: TL pulse (top) vs Chirped pulse (bottom)
# ═══════════════════════════════════════════════════════════════════════════════
gs_r  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=1.0)
ax_tl = fig.add_subplot(gs_r[0])
ax_ch = fig.add_subplot(gs_r[1])

xlim = (-5.5, 5.5)

# ── TL pulse ─────────────────────────────────────────────────────────────────
ax_tl.plot(t_pulse, E_TL, color=BORDER, lw=0.9, alpha=0.40)
ax_tl.plot(t_pulse,  A_TL, color=BORDER, lw=2.0)
ax_tl.plot(t_pulse, -A_TL, color=BORDER, lw=2.0)
ax_tl.fill_between(t_pulse, A_TL, -A_TL, alpha=0.10, color=BORDER)
ax_tl.axvline(0, color='gray', lw=0.7, linestyle='--', alpha=0.55)

# FWHM arrow
tl_half = tau_TL / 2
ax_tl.annotate('', xy=(tl_half, 1.05), xytext=(-tl_half, 1.05),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.2, mutation_scale=8))
ax_tl.text(0, 1.22, r"$\Delta t_{\rm TL}$", ha='center', va='bottom', fontsize=8.5, color='black')

ax_tl.set_xlim(*xlim)
ax_tl.set_ylim(-1.55, 2.10)
ax_tl.set_yticks([])
ax_tl.set_xticks([])
ax_tl.spines[['top', 'right', 'left']].set_visible(False)
ax_tl.spines['bottom'].set_visible(True)
ax_tl.spines['bottom'].set_color('#cccccc')
ax_tl.set_xlabel("Time  →", fontsize=7.5, color=BORDER)
ax_tl.set_title("Transform-limited  (all freqs in phase)", fontsize=8.5, color=BORDER, pad=10)

# ── Chirped pulse ─────────────────────────────────────────────────────────────
lc = LineCollection(segs, colors=seg_colors, lw=1.5, alpha=0.88, zorder=3)
ax_ch.add_collection(lc)
ax_ch.plot(t_pulse,  A_ch, color=BORDER, lw=2.0, zorder=4)
ax_ch.plot(t_pulse, -A_ch, color=BORDER, lw=2.0, zorder=4)
ax_ch.fill_between(t_pulse, A_ch, -A_ch, alpha=0.06, color=BORDER, zorder=2)

# FWHM arrow
ch_half = tau_ch / 2
ax_ch.annotate('', xy=(ch_half, 1.05), xytext=(-ch_half, 1.05),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.2, mutation_scale=8))
ax_ch.text(0, 1.22, r"$\Delta t_{\rm chirped}\gg\Delta t_{\rm TL}$",
           ha='center', va='bottom', fontsize=8, color='black')

# Red / blue frequency labels on the pulse
ax_ch.text(-4.0, -1.28, "low $\\omega$\n(red)", ha='center', fontsize=7.5,
           color=RED_C, fontweight='bold')
ax_ch.text( 4.0, -1.28, "high $\\omega$\n(blue)", ha='center', fontsize=7.5,
           color=BLUE_C, fontweight='bold')

ax_ch.set_xlim(*xlim)
ax_ch.set_ylim(-1.55, 2.10)
ax_ch.set_yticks([])
ax_ch.set_xticks([])
ax_ch.spines[['top', 'right', 'left']].set_visible(False)
ax_ch.spines['bottom'].set_visible(True)
ax_ch.spines['bottom'].set_color('#cccccc')
ax_ch.set_xlabel("Time  →", fontsize=7.5, color=BORDER)
ax_ch.set_title("After SPM: positively chirped  (reds at front, blues at back)",
                fontsize=8.5, color=BORDER, pad=10)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "figures/spm_chirp.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
