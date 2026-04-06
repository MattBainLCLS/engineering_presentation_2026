"""
generate_soliton_animation.py
Two-panel animated GIF:
  Left  — "Wave of translation": sech² soliton gliding without changing shape (canal analogy)
  Right — Optical propagation: linear Gaussian broadening vs N=1 NLS soliton staying sharp
Output: figures/soliton_animation.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

BORDER    = "#44546A"
SOL_C     = "#1565C0"   # soliton blue
LIN_C     = "#78909C"   # linear pulse grey
WATER_C   = "#1565C0"   # canal water
CANAL_C   = "#90A4AE"   # canal surface

N_FRAMES  = 72
FPS       = 12
X_MAX     = 14.0

# ── Spatial coordinate ────────────────────────────────────────────────────────
x = np.linspace(0, X_MAX, 1200)

# ── Temporal coordinate for optical panel ────────────────────────────────────
tau = np.linspace(-5, 5, 1000)

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 3.6), facecolor='white')
gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
ax1 = fig.add_subplot(gs[0])   # canal soliton
ax2 = fig.add_subplot(gs[1])   # optical propagation

def sech2(x, x0, width=1.0):
    return (1.0 / np.cosh((x - x0) / width)) ** 2

def linear_gaussian(tau, z, tau0=0.75):
    """Gaussian pulse width grows as sqrt(1 + (z/Ld)^2), amplitude drops to conserve energy."""
    w = tau0 * np.sqrt(1.0 + z**2)
    return (tau0 / w) * np.exp(-tau**2 / (2 * w**2))

def optical_soliton(tau):
    """N=1 fundamental soliton envelope — independent of z."""
    return sech2(tau, 0.0, width=0.75)

def animate(frame):
    ax1.clear()
    ax2.clear()

    # ── Left panel: wave of translation ──────────────────────────────────────
    frac = frame / (N_FRAMES - 1)
    x0   = 1.5 + frac * (X_MAX - 3.0)   # soliton centre moves across canvas
    wave = sech2(x, x0, width=0.9)

    # Canal water fill (flat baseline with wave on top)
    water_level = 0.10
    ax1.fill_between(x, water_level, wave + water_level,
                     alpha=0.35, color=WATER_C, zorder=2)
    ax1.plot(x, wave + water_level, color=WATER_C, lw=2.2, zorder=3)
    ax1.axhline(water_level, color=CANAL_C, lw=1.0, alpha=0.6, zorder=1)

    ax1.set_xlim(0, X_MAX)
    ax1.set_ylim(-0.12, 1.6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines[['top', 'right', 'left']].set_visible(False)
    ax1.spines['bottom'].set_color(CANAL_C)
    ax1.set_xlabel("Position  →", fontsize=8, color=BORDER)
    ax1.set_title("Wave of Translation  (J. S. Russell, 1834)",
                  fontsize=9, color=BORDER, pad=7)
    ax1.text(X_MAX / 2, 1.45,
             "Shape and speed unchanged",
             ha='center', fontsize=8, color=SOL_C, style='italic')
    # Canal floor/bank suggestion
    ax1.fill_between(x, -0.12, 0.0, color='#B0BEC5', alpha=0.4, zorder=0)

    # ── Right panel: optical propagation ──────────────────────────────────────
    z = frac * 3.0   # 0 → 3 dispersion lengths

    lin = linear_gaussian(tau, z)
    sol = optical_soliton(tau)

    ax2.fill_between(tau, lin / sol.max(), alpha=0.20, color=LIN_C, zorder=1)
    ax2.plot(tau, lin / sol.max(), color=LIN_C, lw=2.0, zorder=2,
             label="Linear pulse  (GVD only)")
    ax2.fill_between(tau, sol / sol.max(), alpha=0.20, color=SOL_C, zorder=3)
    ax2.plot(tau, sol / sol.max(), color=SOL_C, lw=2.5, zorder=4,
             label="Soliton  (N = 1)")

    # Propagation label
    ax2.text(0, 1.36,
             f"z / L$_D$ = {z:.2f}",
             ha='center', fontsize=8, color=BORDER,
             bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#cccccc', lw=0.7))

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-0.06, 1.55)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[['top', 'right', 'left']].set_visible(False)
    ax2.spines['bottom'].set_color('#cccccc')
    ax2.set_xlabel("Time  →", fontsize=8, color=BORDER)
    ax2.set_title("Optical Soliton in Anomalous-GVD Fibre",
                  fontsize=9, color=BORDER, pad=7)
    ax2.legend(fontsize=7.5, framealpha=0.9, loc='upper right',
               edgecolor='#cccccc', handlelength=1.5)

    fig.tight_layout(pad=0.9)

ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS)
out = "figures/soliton_animation.gif"
ani.save(out, writer='pillow', fps=FPS, dpi=110)
print(f"Saved → {out}")
