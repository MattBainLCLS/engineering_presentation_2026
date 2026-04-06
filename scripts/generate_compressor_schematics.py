"""
generate_compressor_schematics.py
Two side-by-side schematics:
  Left  — Prism compressor (two-prism layout, dispersed paths)
  Right — Chirped mirror principle (layer penetration depth vs wavelength)
Output: figures/compressor_schematics.svg
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon, Arc
from matplotlib.collections import PatchCollection

BORDER = "#44546A"
BG     = "white"

fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(10, 3.8))
fig.patch.set_facecolor(BG)
for ax in (ax_p, ax_c):
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG)


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT: Prism compressor
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_p
ax.set_xlim(-0.5, 12)
ax.set_ylim(-1.2, 5.2)
ax.set_title("Prism Compressor", fontsize=10, color=BORDER, pad=6)

PRISM_C = "#B0BEC5"

def draw_prism(ax, apex, base_left, base_right, color=PRISM_C):
    tri = Polygon([apex, base_left, base_right], closed=True,
                  facecolor=color, edgecolor=BORDER, linewidth=1.2, alpha=0.55, zorder=2)
    ax.add_patch(tri)

# Prism 1: apex up, base at y=0
P1_apex = (1.5, 4.0)
P1_bl   = (0.0, 0.0)
P1_br   = (3.0, 0.0)
draw_prism(ax, P1_apex, P1_bl, P1_br)

# Prism 2: flipped (apex DOWN at y=0, base across top at y=4)
P2_apex = (9.0,  0.0)
P2_bl   = (7.5,  4.0)
P2_br   = (10.5, 4.0)
draw_prism(ax, P2_apex, P2_bl, P2_br)

# ── Entry point and Brewster angle indicator ─────────────────────────────────
entry_x, entry_y = 0.75, 2.0

face_mag = np.sqrt(1.5**2 + 4.0**2)
nx, ny =  4.0/face_mag, -1.5/face_mag
ox, oy = -nx, -ny

beam_deg = 34.4
bx = np.cos(np.radians(beam_deg))
by = np.sin(np.radians(beam_deg))

beam_len = 1.4
ax.annotate('', xy=(entry_x, entry_y),
            xytext=(entry_x - beam_len*bx, entry_y - beam_len*by),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2.0))
ax.text(entry_x - beam_len*bx - 0.08,
        entry_y - beam_len*by - 0.18,
        "Input\n(chirped)", ha='center', va='top', fontsize=7, color=BORDER)

norm_ext = 0.85
ax.plot([entry_x + ox*norm_ext, entry_x + nx*norm_ext],
        [entry_y + oy*norm_ext, entry_y + ny*norm_ext],
        color='gray', lw=0.8, linestyle='--', alpha=0.7, zorder=3)

theta_n = np.degrees(np.arctan2(oy, ox))
theta_b = np.degrees(np.arctan2(-by, -bx))
arc55 = Arc((entry_x, entry_y), 0.58, 0.58,
            angle=0, theta1=theta_n, theta2=theta_b,
            color='gray', lw=0.9, zorder=4)
ax.add_patch(arc55)
mid_arc = np.radians((theta_n + theta_b) / 2)
ax.text(entry_x + 0.42*np.cos(mid_arc), entry_y + 0.42*np.sin(mid_arc),
        "55°", fontsize=7.5, color='gray', ha='center', va='center', zorder=5)

# ── Rays ──────────────────────────────────────────────────────────────────────
# Face x-coordinates for flipped P2 (apex y=0, base y=4):
def p1_right(y): return 1.5 + 1.5*(4.0 - y)/4.0
def p2_left(y):  return 7.5 + 1.5*(4.0 - y)/4.0
def p2_right(y): return 10.5 - 1.5*(4.0 - y)/4.0

# Both beams exit P1 going downward (symmetric about the reflection of the 34.4° input in
# the vertical axis ≈ -34.4°, but small enough to clear P2 geometry).
# Red: less refracted, exits P1 higher → bends less steeply down → enters thick (upper) P2
red_y2 = 2.5;  red_x2 = p1_right(red_y2)   # exits P1 at y=2.5
red_y3 = 1.8;  red_x3 = p2_left(red_y3)    # enters P2 at y=1.8  (upper → thick glass)
red_y4 = 2.2;  red_x4 = p2_right(red_y4)   # exits P2 at y=2.2
# Both beams travel at input angle (34.4°) after P2 toward mirror.
# Mirror centre (11.0, 3.0).  Red hits mirror at (pre-calculated):
red_mx,  red_my  = 11.000, 3.000

# Blue: more refracted, exits P1 slightly lower → bends more steeply down → enters thin (lower) P2
blue_y2 = 2.3;  blue_x2 = p1_right(blue_y2) # exits P1 at y=2.3
blue_y3 = 1.2;  blue_x3 = p2_left(blue_y3)  # enters P2 at y=1.2  (lower → thin glass)
blue_y4 = 1.5;  blue_x4 = p2_right(blue_y4) # exits P2 at y=1.5
blue_mx, blue_my = 11.243, 2.647

# Draw red ray
ax.annotate('', xy=(red_x2, red_y2), xytext=(entry_x, entry_y),
            arrowprops=dict(arrowstyle='-', color='#E53935', lw=1.8))
ax.plot([red_x2, red_x3],  [red_y2, red_y3],  color='#E53935', lw=1.8)
ax.plot([red_x3, red_x4],  [red_y3, red_y4],  color='#E53935', lw=1.8)
ax.annotate('', xy=(red_mx, red_my), xytext=(red_x4, red_y4),
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.8))

# Draw blue ray
ax.annotate('', xy=(blue_x2, blue_y2), xytext=(entry_x, entry_y),
            arrowprops=dict(arrowstyle='-', color='#1E88E5', lw=1.8))
ax.plot([blue_x2, blue_x3], [blue_y2, blue_y3], color='#1E88E5', lw=1.8)
ax.plot([blue_x3, blue_x4], [blue_y3, blue_y4], color='#1E88E5', lw=1.8)
ax.annotate('', xy=(blue_mx, blue_my), xytext=(blue_x4, blue_y4),
            arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=1.8))

# λ labels beside output beams (between P2 and mirror)
ax.text((red_x4  + red_mx )/2, (red_y4  + red_my )/2 + 0.22,
        "λ long (red)",   ha='center', fontsize=7.5, color='#E53935', fontweight='bold')
ax.text((blue_x4 + blue_mx)/2, (blue_y4 + blue_my)/2 - 0.22,
        "λ short (blue)", ha='center', fontsize=7.5, color='#1E88E5', fontweight='bold')

# Thick/thin labels next to P2
ax.text(red_x3  - 0.3, red_y3  + 0.50, "thick\nglass", ha='center',
        fontsize=6.5, color='#E53935', alpha=0.9, style='italic')
ax.text(blue_x3 - 0.3, blue_y3 - 0.55, "thin\nglass",  ha='center',
        fontsize=6.5, color='#1E88E5', alpha=0.9, style='italic')

# ── Retroreflecting mirror (perpendicular to beam = normal incidence) ──────────
mirror_cx, mirror_cy = 11.0, 3.0
mdx = -np.sin(np.radians(beam_deg))   # mirror surface direction x
mdy =  np.cos(np.radians(beam_deg))   # mirror surface direction y
half_m = 0.55
mx0 = mirror_cx + half_m*mdx;  my0 = mirror_cy + half_m*mdy
mx1 = mirror_cx - half_m*mdx;  my1 = mirror_cy - half_m*mdy
ax.plot([mx0, mx1], [my0, my1], color=BORDER, lw=3.5, solid_capstyle='butt', zorder=5)
# hatching on the back side (+beam direction)
for hf in np.linspace(0.05, 0.95, 7):
    hx = mx0 + hf*(mx1 - mx0);  hy = my0 + hf*(my1 - my0)
    ax.plot([hx, hx + 0.13*bx], [hy, hy + 0.13*by],
            color=BORDER, lw=0.9, alpha=0.65, zorder=5)
ax.text(mirror_cx + 0.38*bx, mirror_cy + 0.38*by,
        "normal incidence", ha='center', fontsize=6.5, color=BORDER, style='italic')

# Mechanism annotation and GDD label
ax.text(5.5, -0.85, "Red through thicker glass in P2 → more phase → arrives later",
        ha='center', va='top', fontsize=7.5, color=BORDER, style='italic')
ax.text(5.5, 4.8, "Net effect: negative GDD", ha='center', fontsize=8.5,
        color=BORDER, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT: Chirped mirror principle
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_c
ax.set_xlim(-1.0, 9.0)
ax.set_ylim(-0.5, 9.5)
ax.set_title("Chirped Mirror", fontsize=10, color=BORDER, pad=6)

# Draw alternating H/L layers — chirped: layers get thicker with depth
N_LAYERS = 18
y0       = 0.0
colors_HL = ["#CFD8DC", "#78909C"]   # light/dark grey for H and L layers
layer_tops = []
y = y0
for i in range(N_LAYERS):
    thickness = 0.54 - i * 0.022   # chirped: layers closer to substrate are thicker (longer λ penetrates deeper)
    color = colors_HL[i % 2]
    rect = mpatches.Rectangle((0, y), 6.0, thickness,
                                facecolor=color, edgecolor='none', zorder=2)
    ax.add_patch(rect)
    layer_tops.append(y + thickness)
    y += thickness

# Substrate below
substrate = mpatches.Rectangle((0, -0.5), 6.0, 0.5,
                                 facecolor='#546E7A', edgecolor=BORDER,
                                 linewidth=0.8, zorder=2)
ax.add_patch(substrate)
ax.text(3.0, -0.25, "Substrate", ha='center', va='center',
        fontsize=7, color='white', fontweight='bold')

# Top surface label
ax.text(3.0, y + 0.15, "Surface", ha='center', va='bottom',
        fontsize=7.5, color=BORDER)

# ── Blue ray: reflects near the surface (shallow) ────────────────────────────
blue_refl_y = layer_tops[13]  # near surface = shallow penetration
ax.annotate('', xy=(1.5, blue_refl_y), xytext=(1.5, y + 0.9),
            arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=2.0))
ax.annotate('', xy=(1.5, y + 0.9), xytext=(1.5, blue_refl_y),
            arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=2.0,
                            connectionstyle='arc3,rad=0.5'))
ax.text(0.2, y + 1.1, "λ short (blue)\nshallow penetration",
        ha='center', va='bottom', fontsize=7.5, color='#1E88E5', fontweight='bold')

# Horizontal dashed marker for blue reflection depth
ax.plot([-0.8, 2.2], [blue_refl_y, blue_refl_y],
        color='#1E88E5', lw=0.9, linestyle='--', alpha=0.7)

# ── Red ray: penetrates deeper ────────────────────────────────────────────────
red_refl_y = layer_tops[4]   # near substrate = deep penetration
ax.annotate('', xy=(4.5, red_refl_y), xytext=(4.5, y + 0.9),
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=2.0))
ax.annotate('', xy=(4.5, y + 0.9), xytext=(4.5, red_refl_y),
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=2.0,
                            connectionstyle='arc3,rad=-0.5'))
ax.text(5.8, y + 1.1, "λ long (red)\ndeeper penetration",
        ha='center', va='bottom', fontsize=7.5, color='#E53935', fontweight='bold')

ax.plot([3.8, 6.8], [red_refl_y, red_refl_y],
        color='#E53935', lw=0.9, linestyle='--', alpha=0.7)

# ── Multilayer mirror label on right ─────────────────────────────────────────
ax.text(7.6, y / 2, "Multilayer\nmirror", ha='center', va='center',
        fontsize=7.5, color=BORDER, rotation=90,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))

# ── Double-headed arrow showing path difference ───────────────────────────────
ax.annotate('', xy=(-0.7, red_refl_y), xytext=(-0.7, blue_refl_y),
            arrowprops=dict(arrowstyle='<->', color=BORDER, lw=1.3))
ax.text(-0.9, (red_refl_y + blue_refl_y) / 2,
        "ΔL", ha='right', va='center', fontsize=9, color=BORDER, fontweight='bold')

# GDD label
ax.text(3.0, 9.2, "Net effect: negative GDD", ha='center', fontsize=8.5,
        color=BORDER, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.8))

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=1.0)
out = "figures/compressor_schematics.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
