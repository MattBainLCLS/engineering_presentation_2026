"""
generate_pump_probe.py
Pump-probe schematic combining spatial geometry and temporal structure.

Left  : Experimental geometry with beams drawn as wave packets
        (Gaussian-envelope × carrier), not CW waves.
Right : Temporal pulse diagram with updated durations (50 fs pump, 20 fs X-ray).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato', 'Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

BORDER   = "#44546A"
PUMP_C   = "#E04F39"   # SPIRITED — optical pump
XRAY_C   = "#53284F"   # PURPLE   — X-ray probe
SAMPLE_C = "#7F7776"   # STONE    — detector (repurposed)
WATER_FILL = "#85C1E9"  # light blue — sample fill
WATER_EDGE = "#1A5276"  # dark navy  — sample stroke
SIGNAL_C = "#279989"   # PALO VERDE — sample response
BG       = "white"

fig = plt.figure(figsize=(13, 5.76))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.58, 1.0], wspace=0.28)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: draw a Gaussian pulse envelope along a straight beam path
# ═══════════════════════════════════════════════════════════════════════════════
def draw_pulse(ax, p0, p1, env_fwhm_frac=0.35,
               amplitude=0.30, color='black', lw=2.0, center_frac=0.55,
               alpha_fill=0.25):
    """
    Draw a filled Gaussian pulse shape travelling along the path p0 → p1.
    env_fwhm_frac : Gaussian FWHM as a fraction of the path length.
    center_frac   : where along the path the pulse centre sits (0–1).
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    vec  = p1 - p0
    L    = np.linalg.norm(vec)
    uhat = vec / L
    nhat = np.array([-uhat[1], uhat[0]])

    s      = np.linspace(0, L, 2000)
    s_ctr  = center_frac * L
    sigma  = (env_fwhm_frac * L) / (2 * np.sqrt(2 * np.log(2)))
    env    = np.exp(-0.5 * ((s - s_ctr) / sigma) ** 2)

    # Spine (flat beam axis)
    spine = p0 + np.outer(s, uhat)
    # Gaussian bump above the spine
    bump  = spine + np.outer(amplitude * env, nhat)

    # Flat baseline
    ax.plot(spine[:, 0], spine[:, 1], color=color, lw=lw,
            solid_capstyle='round', zorder=3)
    # Filled bump
    ax.fill_betweenx(bump[:, 1], bump[:, 0], spine[:, 0],
                     alpha=alpha_fill, color=color, zorder=2)
    ax.plot(bump[:, 0], bump[:, 1], color=color, lw=lw * 0.9,
            solid_capstyle='round', zorder=3)

    # Arrowhead at terminus along the beam axis
    ax.annotate('', xy=p1, xytext=p1 - 0.35 * uhat,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=13), zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT: Geometry with wave-packet beams
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 10)
ax.set_ylim(0.5, 7.5)
ax.axis('off')
ax.set_title("Experimental Geometry", fontsize=10.5, color=BORDER, pad=8)

# ── Sample ────────────────────────────────────────────────────────────────────
sx, sy  = 5.1, 3.8
sw, sh  = 1.0, 1.5
sample  = mpatches.FancyBboxPatch((sx - sw/2, sy - sh/2), sw, sh,
                                   boxstyle="round,pad=0.08",
                                   fc=WATER_FILL, ec=WATER_EDGE, lw=1.8, zorder=5)
ax.add_patch(sample)
ax.text(sx, sy, "Sample", ha='center', va='center', fontsize=8.5,
        color='white', fontweight='bold', zorder=6)

# ── Optical pump — wave packet, from top-left at ~40° ────────────────────────
pump_src = [0.6, 6.0]
pump_end = [sx - sw/2 - 0.05, sy + sh * 0.28]

# Wider pulse (50 fs pump)
draw_pulse(ax, pump_src, pump_end,
           env_fwhm_frac=0.14, amplitude=0.52,
           color=PUMP_C, lw=2.2, center_frac=0.52, alpha_fill=0)

ax.text(0.45, 7.25,
        "Optical pump  (1030 nm, ~50 fs)",
        fontsize=8, color=PUMP_C, ha='left', va='bottom', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=PUMP_C,
                  lw=0.7, alpha=0.88))

# ── X-ray probe — wave packet, horizontal, arriving later (Δt) ───────────────
# Positioned slightly further from sample to indicate it arrives after pump
xray_src = [0.4, sy - 0.05]
xray_end = [sx - sw/2 - 0.05, sy - 0.05]

# Narrower pulse (20 fs X-ray)
draw_pulse(ax, xray_src, xray_end,
           env_fwhm_frac=0.07, amplitude=0.44,
           color=XRAY_C, lw=2.0, center_frac=0.42)

ax.text(0.3, sy - 0.65,
        "X-ray probe  (LCLS, ~20 fs)",
        fontsize=8, color=XRAY_C, ha='left', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=XRAY_C,
                  lw=0.7, alpha=0.88))

# ── Transmitted X-ray → detector ─────────────────────────────────────────────
det_x = 8.55
draw_pulse(ax, [sx + sw/2 + 0.05, sy - 0.05], [det_x - 0.05, sy - 0.05],
           env_fwhm_frac=0.18, amplitude=0.40,
           color=XRAY_C, lw=1.8, center_frac=0.50)

det = mpatches.FancyBboxPatch((det_x, sy - 0.75), 0.62, 1.45,
                               boxstyle="round,pad=0.06",
                               fc=SAMPLE_C, ec=BORDER, lw=1.5, zorder=5)
ax.add_patch(det)
ax.text(det_x + 0.31, sy - 0.02, "Detector", ha='center', va='center',
        fontsize=7.5, color='white', rotation=90, fontweight='bold', zorder=6)

# ── Δt callout ────────────────────────────────────────────────────────────────
ax.annotate(r"delay  $\Delta t$",
            xy=(sx - sw/2 - 0.7, sy + 0.15),
            xytext=(sx + sw/2 + 0.5, sy + sh/2 + 1.0),
            fontsize=9.5, color=BORDER, ha='left', va='bottom',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.1,
                            connectionstyle='arc3,rad=-0.28'),
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=BORDER, lw=0.9, alpha=0.92))


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT: Temporal pulse diagram
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])

t = np.linspace(-200, 500, 6000)

def gauss(t, centre, fwhm):
    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((t - centre) / sig) ** 2)

pump_env = gauss(t,   0, 50)
xray_env = gauss(t, 150, 20)

tau  = 220.0
resp = np.where(t >= 0, 0.58 * np.exp(-t / tau), 0.0)

ax2.fill_between(t, pump_env, alpha=0.18, color=PUMP_C)
ax2.plot(t, pump_env, color=PUMP_C, lw=2.2, label="Optical pump  (~50 fs)")

ax2.fill_between(t, xray_env, alpha=0.25, color=XRAY_C)
ax2.plot(t, xray_env, color=XRAY_C, lw=2.2, label="X-ray probe  (~20 fs)")

ax2.plot(t, resp, color=SIGNAL_C, lw=1.8, linestyle='--',
         label="Sample excitation")

# Δt double-headed arrow
y_dt = 1.10
ax2.annotate('', xy=(150, y_dt), xytext=(0, y_dt),
             arrowprops=dict(arrowstyle='<->', color=BORDER, lw=1.6,
                             mutation_scale=11))
ax2.text(75, y_dt + 0.05, r"$\Delta t$  (scannable)",
         ha='center', va='bottom', fontsize=10, color=BORDER, fontweight='bold')

ax2.axvline(0, color=PUMP_C, lw=0.8, linestyle=':', alpha=0.6)
ax2.text(4, -0.04, "pump\narrives", ha='left', va='top',
         fontsize=7, color=PUMP_C, alpha=0.85)

ax2.set_xlim(-180, 480)
ax2.set_ylim(-0.08, 1.32)
ax2.set_xlabel("Time (fs)", fontsize=9, color=BORDER)
ax2.set_ylabel("Intensity  /  Excitation (norm.)", fontsize=8.5, color=BORDER)
ax2.set_title("Temporal Structure", fontsize=10.5, color=BORDER, pad=8)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(labelsize=8)
ax2.legend(fontsize=7.8, loc='upper right', framealpha=0.85, handlelength=1.8)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "figures/pump_probe.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
