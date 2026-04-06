"""
generate_soliton_optical_animation.py
Compact animated panel for embedding in the GVD slide:
  Gray  — linear Gaussian pulse broadening in anomalous GVD (dispersion only)
  Blue  — N=1 NLS soliton: envelope unchanged as z/L_D increases
Output: figures/soliton_optical_animation.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

BORDER = "#44546A"
SOL_C  = "#1565C0"
LIN_C  = "#78909C"

N_FRAMES = 72
FPS      = 12

tau = np.linspace(-5.5, 5.5, 1000)

def linear_gaussian(tau, z, tau0=0.75):
    w = tau0 * np.sqrt(1.0 + z ** 2)
    return (tau0 / w) * np.exp(-tau ** 2 / (2.0 * w ** 2))

def optical_soliton(tau):
    return (1.0 / np.cosh(tau / 0.75)) ** 2  # sech², same width as tau0=0.75

sol_peak = optical_soliton(tau).max()

fig, ax = plt.subplots(figsize=(5.2, 3.0), facecolor="white")
ax.set_facecolor("white")

def animate(frame):
    ax.clear()
    ax.set_facecolor("white")

    frac = frame / (N_FRAMES - 1)
    z    = frac * 3.0

    lin = linear_gaussian(tau, z)
    sol = optical_soliton(tau)

    ax.fill_between(tau, lin / sol_peak, alpha=0.18, color=LIN_C, zorder=1)
    ax.plot(tau, lin / sol_peak, color=LIN_C, lw=2.0, zorder=2,
            label="Linear pulse  (GVD only)")
    ax.fill_between(tau, sol / sol_peak, alpha=0.20, color=SOL_C, zorder=3)
    ax.plot(tau, sol / sol_peak, color=SOL_C, lw=2.5, zorder=4,
            label="Soliton  (N = 1)")

    ax.text(0, 1.30,
            f"z / L$_D$ = {z:.2f}",
            ha='center', fontsize=8.5, color=BORDER,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#cccccc', lw=0.7))

    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper right',
              edgecolor='#cccccc', handlelength=1.4)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.06, 1.52)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_xlabel("Time  →", fontsize=8.5, color=BORDER)
    ax.set_title("Anomalous GVD: soliton vs. linear pulse",
                 fontsize=9, color=BORDER, pad=7)

    fig.tight_layout(pad=0.6)

ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS)
out = "figures/soliton_optical_animation.gif"
ani.save(out, writer='pillow', fps=FPS, dpi=110)
print(f"Saved → {out}")
