"""
generate_fiber_schematics.py
Draws cross-section schematics of four hollow-core fiber types and saves
the figure to figures/hcf_comparison.svg
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe

SILICA  = "#B0BEC5"   # glass
AIR     = "#FFFFFF"   # air / hollow regions
BORDER  = "#44546A"   # SLAC dark
STRUT   = "#78909C"   # thinner internal borders

def setup(ax, title):
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=BORDER,
                 pad=10)

# ── Helper: regular polygon patch ─────────────────────────────────────────────
def reg_poly(cx, cy, r, n, angle0=0, **kw):
    angles = np.linspace(angle0, angle0 + 2*np.pi, n, endpoint=False)
    verts  = np.column_stack([cx + r*np.cos(angles),
                              cy + r*np.sin(angles)])
    return mpatches.Polygon(verts, closed=True, **kw)

# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.05)

# ── 1. SOLID CORE FIBRE ───────────────────────────────────────────────────────
ax = axes[0]
setup(ax, "Solid-Core Fibre")
ax.add_patch(Circle((0,0), 1.0,   fc=SILICA, ec=BORDER,  lw=1.8, zorder=1))
ax.add_patch(Circle((0,0), 0.30,  fc="#607D8B", ec=BORDER, lw=1.2, zorder=2))
ax.text(0, -1.18, "Silica core\n(n ≈ 1.45)",
        ha="center", va="top", fontsize=7.5, color=BORDER)

# ── 2. KAGOME HC-PCF ─────────────────────────────────────────────────────────
ax = axes[1]
setup(ax, "Kagome HC-PCF")
ax.add_patch(Circle((0,0), 1.0, fc=SILICA, ec=BORDER, lw=1.8, zorder=1))

# Triangular lattice of large air holes — thin silica struts between them
a    = 0.285          # lattice constant
r_h  = 0.115          # air hole radius (large → thin struts)
core_r = 0.30

for i in range(-6, 7):
    for j in range(-6, 7):
        x = a * i + a * 0.5 * j
        y = a * (np.sqrt(3)/2) * j
        dist = np.sqrt(x**2 + y**2)
        if dist > core_r + r_h*0.6 and dist + r_h < 0.92:
            ax.add_patch(Circle((x, y), r_h, fc=AIR, ec=STRUT, lw=0.5, zorder=2))

ax.add_patch(Circle((0,0), core_r, fc=AIR, ec=BORDER, lw=1.4, zorder=3))
ax.text(0, -1.18, "Hollow core\nKagome web cladding",
        ha="center", va="top", fontsize=7.5, color=BORDER)

# ── 3. ANTI-RESONANT HCF ─────────────────────────────────────────────────────
ax = axes[2]
setup(ax, "Anti-Resonant HCF")
ax.add_patch(Circle((0,0), 1.0, fc=SILICA, ec=BORDER, lw=1.8, zorder=1))

n_tubes  = 6
r_ring   = 0.565
r_tube   = 0.23
wall     = 0.025      # tube wall thickness

for i in range(n_tubes):
    angle = np.pi/2 + 2*np.pi * i / n_tubes   # start at top
    cx, cy = r_ring * np.cos(angle), r_ring * np.sin(angle)
    ax.add_patch(Circle((cx, cy), r_tube,        fc=SILICA, ec=STRUT, lw=0.8, zorder=2))
    ax.add_patch(Circle((cx, cy), r_tube - wall, fc=AIR,    ec=STRUT, lw=0.5, zorder=3))

core_r = r_ring - r_tube - 0.06
ax.add_patch(Circle((0,0), core_r, fc=AIR, ec=BORDER, lw=1.4, zorder=4))
ax.text(0, -1.18, "Hollow core\nAnti-resonant capillaries",
        ha="center", va="top", fontsize=7.5, color=BORDER)

# ── 4. HOLLOW CAPILLARY ───────────────────────────────────────────────────────
ax = axes[3]
setup(ax, "Hollow Capillary")
ax.add_patch(Circle((0,0), 1.0,  fc=SILICA, ec=BORDER, lw=1.8, zorder=1))
ax.add_patch(Circle((0,0), 0.72, fc=AIR,    ec=BORDER, lw=1.4, zorder=2))
ax.text(0, -1.18, "Thick glass wall\nLarge hollow core",
        ha="center", va="top", fontsize=7.5, color=BORDER)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "figures/hcf_comparison.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
