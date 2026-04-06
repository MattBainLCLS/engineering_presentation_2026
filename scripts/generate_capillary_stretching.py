"""
generate_capillary_stretching.py
Figure illustrating how stretching a hollow capillary improves transmission
by increasing the angle of incidence (towards grazing) at the glass wall.

Left panel  : Fresnel reflectance Rs and Rp vs angle of incidence (air→glass),
              with shaded regions for curved vs stretched fibre operating angles.
Right panel : Two fibre cross-section schematics (curved / straight) showing
              how the bounce angle changes with fibre geometry.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Arc, FancyArrowPatch

BORDER  = "#44546A"
GLASS   = "#B0BEC5"
CURVED_C  = "#E65100"   # orange — curved fibre / lower AOI
STRETCH_C = "#2E7D32"   # green  — stretched fibre / grazing AOI
RAY_C   = "#FDD835"     # yellow ray

# ── Fresnel equations (air → glass) ──────────────────────────────────────────
def fresnel(theta_deg, n2=1.45):
    ti = np.radians(theta_deg)
    sin_tt = np.clip(np.sin(ti) / n2, -1, 1)
    tt = np.arcsin(sin_tt)
    ci, ct = np.cos(ti), np.cos(tt)
    Rs = ((ci - n2 * ct) / (ci + n2 * ct)) ** 2
    Rp = ((ct - n2 * ci) / (ct + n2 * ci)) ** 2
    return Rs, Rp

theta = np.linspace(0, 89.8, 1000)
Rs, Rp = fresnel(theta)
theta_B = np.degrees(np.arctan(1.45))   # Brewster angle ≈ 55.4°

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4.0))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.3], wspace=0.38)

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Fresnel reflectance
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])

ax.plot(theta, Rs, color=BORDER,    lw=2.2, label=r'$R_s$')
ax.plot(theta, Rp, color="#607D8B", lw=2.2, linestyle='--', label=r'$R_p$')

# Brewster's angle marker
ax.axvline(theta_B, color='gray', lw=0.9, linestyle=':')
ax.text(theta_B - 1.2, 0.35, f"Brewster\n({theta_B:.0f}°)",
        ha='right', va='center', fontsize=7.5, color='gray')

# Shaded operating regions
ax.axvspan(70, 80, alpha=0.13, color=CURVED_C,  zorder=0)
ax.axvspan(83, 89.8, alpha=0.13, color=STRETCH_C, zorder=0)
ax.text(75, 0.60, "Curved\nfibre", ha='center', fontsize=7.5,
        color=CURVED_C, fontweight='bold')
ax.text(80.5, 0.88, "Stretched\nfibre", ha='center', fontsize=7.5,
        color=STRETCH_C, fontweight='bold')

ax.set_xlabel("Angle of incidence from normal (°)", fontsize=9, color=BORDER)
ax.set_ylabel("Reflectance", fontsize=9, color=BORDER)
ax.set_xlim(0, 90)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='upper left', framealpha=0.7)
ax.set_title("Fresnel Reflectance: Air–Glass", fontsize=10,
             color=BORDER, pad=8)
ax.tick_params(labelsize=8)
ax.spines[['top', 'right']].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — fibre schematics
# ═══════════════════════════════════════════════════════════════════════════════
gs_r = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.6)

WALL = 0.10   # wall thickness in data coords

# ── helper: draw a straight or curved hollow fibre ───────────────────────────
def draw_fiber(ax, x_arr, y_centre, half_gap, color_title, title_str):
    """
    x_arr    : 1-D array of x positions
    y_centre : 1-D array of fibre centre y positions (allows curvature)
    half_gap : half inner diameter
    """
    y_top_in  = y_centre + half_gap
    y_top_out = y_top_in  + WALL
    y_bot_in  = y_centre - half_gap
    y_bot_out = y_bot_in  - WALL

    ax.fill_between(x_arr, y_top_in,  y_top_out, color=GLASS,  lw=0)
    ax.fill_between(x_arr, y_bot_out, y_bot_in,  color=GLASS,  lw=0)
    ax.plot(x_arr, y_top_out, color=BORDER, lw=1.0)
    ax.plot(x_arr, y_bot_out, color=BORDER, lw=1.0)
    # faint inner edges
    ax.plot(x_arr, y_top_in, color=BORDER, lw=0.4, alpha=0.5)
    ax.plot(x_arr, y_bot_in, color=BORDER, lw=0.4, alpha=0.5)

    ax.set_title(title_str, fontsize=8.5, color=color_title, pad=5)
    ax.axis('off')

# ── CURVED fibre ──────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs_r[0])
ax2.set_xlim(0, 10)
ax2.set_ylim(-1.6, 1.6)

N = 400
x = np.linspace(0.3, 9.7, N)
sag = 0.55                          # peak sag of the curve
y_c = sag * np.sin(np.pi * (x - 0.3) / 9.4)   # gentle arc

draw_fiber(ax2, x, y_c, half_gap=0.45,
           color_title=CURVED_C,
           title_str="Curved capillary — reduced AOI, higher loss per bounce")

# Ray bouncing inside curved fibre
# Bounce x-positions; y alternates between top & bottom inner wall
def wall_y(xi, which, x_arr=x, y_c=y_c, hg=0.45):
    yc = np.interp(xi, x_arr, y_c)
    return yc + (hg - 0.02) * (1 if which == 'top' else -1)

bx = [0.6, 2.2, 3.9, 5.6, 7.3, 9.0]
by = [wall_y(bx[i], 'bot' if i % 2 == 0 else 'top') for i in range(len(bx))]

for i in range(len(bx) - 1):
    ax2.annotate('', xy=(bx[i+1], by[i+1]), xytext=(bx[i], by[i]),
                 arrowprops=dict(arrowstyle='->', color=RAY_C, lw=1.6,
                                 mutation_scale=10))
    ax2.plot([bx[i], bx[i+1]], [by[i], by[i+1]], color=RAY_C, lw=1.6)

# Annotate one bounce with AOI arc and label
# At bounce index 2 (on bottom wall): draw normal and show shallow angle
bxi, byi = bx[2], by[2]
# Normal points downward from bottom wall → direction (0, -1) in local coords
ax2.annotate("Smaller AOI\n→ lower R per bounce",
             xy=(bxi, byi), xytext=(bxi + 1.1, byi - 0.75),
             fontsize=7, color=CURVED_C, ha='left',
             arrowprops=dict(arrowstyle='->', color=CURVED_C, lw=0.8))

# ── STRAIGHT (stretched) fibre ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs_r[1])
ax3.set_xlim(0, 10)
ax3.set_ylim(-1.6, 1.6)

x_s = np.linspace(0.3, 9.7, N)
y_s = np.zeros(N)   # flat

draw_fiber(ax3, x_s, y_s, half_gap=0.45,
           color_title=STRETCH_C,
           title_str="Stretched capillary — near-grazing AOI, lower cumulative loss")

# Ray bouncing at very shallow angle (many widely-spaced bounces)
hg = 0.43
bx2 = [0.5, 2.6, 4.7, 6.8, 9.0]
by2 = [(-hg if i % 2 == 0 else hg) for i in range(len(bx2))]

for i in range(len(bx2) - 1):
    ax3.annotate('', xy=(bx2[i+1], by2[i+1]), xytext=(bx2[i], by2[i]),
                 arrowprops=dict(arrowstyle='->', color=RAY_C, lw=1.6,
                                 mutation_scale=10))
    ax3.plot([bx2[i], bx2[i+1]], [by2[i], by2[i+1]], color=RAY_C, lw=1.6)

# Annotate with grazing AOI
bxi, byi = bx2[2], by2[2]
ax3.annotate("Grazing AOI\n→ R → 1 per bounce",
             xy=(bxi, byi), xytext=(bxi + 1.1, byi - 0.75),
             fontsize=7, color=STRETCH_C, ha='left',
             arrowprops=dict(arrowstyle='->', color=STRETCH_C, lw=0.8))

# ── Save ──────────────────────────────────────────────────────────────────────
out = "figures/capillary_stretching.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
