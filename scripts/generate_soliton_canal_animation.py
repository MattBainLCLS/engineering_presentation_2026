"""
generate_soliton_canal_animation.py
Canal wave animation showing two waves propagating at the same group velocity:
  Blue  — soliton (sech²): shape and amplitude preserved throughout
  Orange — non-soliton pulse: same initial shape but disperses (broadens, flattens)
Output: figures/soliton_canal_animation.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

BORDER  = "#44546A"
SOL_C   = "#1565C0"    # soliton: blue
DISP_C  = "#E65100"    # dispersing: orange
WATER   = "#B3D4F5"    # canal surface fill
FLOOR_C = "#B0BEC5"    # canal floor

N_FRAMES  = 80
FPS       = 12
X_MAX     = 14.0

x = np.linspace(0, X_MAX, 1400)

def sech2(x, x0, width):
    return (1.0 / np.cosh((x - x0) / width)) ** 2

fig, ax = plt.subplots(figsize=(9, 3.4), facecolor="white")
ax.set_facecolor("white")

def animate(frame):
    ax.clear()
    ax.set_facecolor("white")

    frac = frame / (N_FRAMES - 1)

    # Both waves travel at the same group velocity
    travel = frac * 9.5
    WIDTH0  = 0.85   # initial half-width for both

    # ── Soliton (left): sech², constant shape ────────────────────────────────
    x0_sol = 1.2 + travel
    sol = sech2(x, x0_sol, WIDTH0)

    # ── Dispersing wave (right): same initial shape, broadens over propagation ──
    # Width grows as sqrt(1 + (3*frac)^2), amplitude drops to conserve norm
    x0_disp = 7.0 + travel
    width_d  = WIDTH0 * np.sqrt(1.0 + (3.0 * frac) ** 2)
    amp_d    = WIDTH0 / width_d     # amplitude scales as 1/width (area conserved)
    disp = amp_d * sech2(x, x0_disp, width_d)

    water_level = 0.08

    # Canal water background tint
    ax.fill_between(x, -0.08, water_level, color=WATER, alpha=0.3, zorder=0)
    ax.axhline(water_level, color="#78909C", lw=0.8, alpha=0.5, zorder=1)

    # Dispersing wave
    ax.fill_between(x, water_level, disp + water_level,
                    alpha=0.30, color=DISP_C, zorder=2)
    ax.plot(x, disp + water_level, color=DISP_C, lw=2.2, zorder=3,
            label="Non-soliton pulse  (disperses)")

    # Soliton
    ax.fill_between(x, water_level, sol + water_level,
                    alpha=0.35, color=SOL_C, zorder=4)
    ax.plot(x, sol + water_level, color=SOL_C, lw=2.5, zorder=5,
            label="Soliton  (shape preserved)")

    # Canal floor
    ax.fill_between(x, -0.08, 0.0, color=FLOOR_C, alpha=0.45, zorder=0)

    # Labels tracking the peaks
    ax.text(min(x0_sol, X_MAX - 0.3), sol.max() + water_level + 0.07,
            "Soliton", ha='center', fontsize=9, color=SOL_C,
            fontweight='bold', zorder=6)
    if x0_disp < X_MAX - 0.5:
        ax.text(x0_disp, disp.max() + water_level + 0.07,
                "Non-soliton", ha='center', fontsize=9, color=DISP_C,
                fontweight='bold', zorder=6)

    ax.legend(fontsize=8, framealpha=0.9, loc='upper right',
              edgecolor='#cccccc', handlelength=1.4)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-0.10, 1.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(FLOOR_C)
    ax.set_xlabel("Position  →", fontsize=9, color=BORDER)
    ax.set_title("Wave of Translation — Union Canal, Edinburgh  (J. S. Russell, 1834)",
                 fontsize=10, color=BORDER, pad=8)

    fig.tight_layout(pad=0.7)

ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS)
out = "figures/soliton_canal_animation.gif"
ani.save(out, writer='pillow', fps=FPS, dpi=110)
print(f"Saved → {out}")
