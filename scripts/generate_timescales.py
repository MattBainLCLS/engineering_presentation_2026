"""
generate_timescales.py
Horizontal logarithmic timeline of photoinitiated molecular/electronic processes,
spanning attoseconds to tens of picoseconds.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BORDER = "#44546A"
BG     = "white"

# ── Process definitions (t_min_fs, t_max_fs, label, bar_color, label_side) ───
# label_side: 'left' = before bar, 'right' = after bar
processes = [
    (0.01,   0.8,   "Electron wavepacket\ndynamics",          "#6A1B9A", 'left'),
    (0.05,   5.0,   "Proton / H-atom\ntransfer",              "#1565C0", 'left'),
    (10,     400,   "Electronic relaxation\n(internal conversion)", "#00838F", 'right'),
    (50,     8000,  "Molecular motion, vibrations\n& bond breaking", "#2E7D32", 'right'),
    (2000,   1e5,   "Rotational &\nconformational dynamics",   "#B71C1C", 'right'),
]

fig, ax = plt.subplots(figsize=(12, 4.2))
fig.patch.set_facecolor(BG)

# ── Draw bars ─────────────────────────────────────────────────────────────────
bar_height = 0.55
y_gap      = 1.0
n          = len(processes)

for i, (tmin, tmax, label, color, side) in enumerate(processes):
    y = (n - 1 - i) * y_gap
    ax.barh(y, np.log10(tmax) - np.log10(tmin),
            left=np.log10(tmin), height=bar_height,
            color=color, alpha=0.82, zorder=3)
    if side == 'left':
        ax.text(np.log10(tmin) - 0.08, y, label,
                ha='right', va='center', fontsize=8.0, color=color,
                fontweight='bold', linespacing=1.4)
    else:
        ax.text(np.log10(tmax) + 0.08, y, label,
                ha='left', va='center', fontsize=8.0, color=color,
                fontweight='bold', linespacing=1.4)

# ── Vertical pulse-duration markers ──────────────────────────────────────────
# (value_fs, color, linestyle, label, ha, y_level)
# y_level 0 → -0.55,  y_level 1 → -1.15  (stagger close lines)
XRAY_C    = "#4527A0"   # purple  — X-ray
TIS_C     = "#C62828"   # dark red — Ti:Sapphire 800 nm
CARBIDE_C = "#E65100"   # amber    — Carbide 1030 nm

markers = [
    (0.2,  XRAY_C,    '--', "LCLS X-ray XLEAP\n(< 500 as)",         'left',  0),
    (7,    XRAY_C,    '--', "LCLS X-ray Low charge\n(< 10 fs)",      'right', 2),
    (30,   XRAY_C,    '--', "LCLS X-ray\n(30 fs)",                   'right', 0),
    (40,   TIS_C,     '-',  "Ti:S 800 nm\n(40 fs)",                  'left',  1),
    (300,  CARBIDE_C, '-',  "Carbide 1030 nm\n(300 fs)",             'left',  0),
]

y_levels = {0: -0.55, 1: -1.15, 2: 0.8}

for val_fs, color, ls, label, ha, level in markers:
    x = np.log10(val_fs)
    ax.axvline(x, color=color, lw=1.5, linestyle=ls, zorder=4, alpha=0.85)
    offset = -0.05 if ha == 'right' else 0.05
    ax.text(x + offset, y_levels[level], label,
            ha=ha, va='top', fontsize=7.2, color=color, fontweight='bold',
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=color, lw=0.8, alpha=0.92))

# ── X-axis: custom labels in as / fs / ps ────────────────────────────────────
tick_vals = [-2, -1, 0, 1, 2, 3, 4, 5]   # log10(fs)
tick_labels = ["10 as", "100 as", "1 fs", "10 fs", "100 fs", "1 ps", "10 ps", "100 ps"]

ax.set_xticks(tick_vals)
ax.set_xticklabels(tick_labels, fontsize=9, color=BORDER)
ax.set_xlim(-2.3, 5.5)
ax.set_ylim(-2.1, n * y_gap)

ax.set_yticks([])
ax.set_xlabel("Time after photoexcitation", fontsize=10, color=BORDER, labelpad=6)
ax.set_title("Timescales of Photoinitiated Processes", fontsize=11,
             color=BORDER, pad=10)

ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='x', length=4, color=BORDER)
ax.grid(axis='x', which='major', color='#ECEFF1', lw=0.8, zorder=0)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "figures/timescales.svg"
plt.tight_layout()
plt.savefig(out, format="svg", bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
