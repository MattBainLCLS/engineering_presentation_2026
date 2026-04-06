"""
generate_spm_concept.py
Single-panel figure: Gaussian pulse I(t) with the instantaneous frequency
shift δω(t) = −dφ/dt = −B·dI/dt overlaid on a secondary y-axis.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BORDER  = "#44546A"
PULSE_C = "#E65100"   # orange — intensity
FREQ_C  = "#1565C0"   # blue   — instantaneous frequency shift

t  = np.linspace(-2.5, 2.5, 1000)
I  = np.exp(-2 * np.log(2) * t**2)   # Gaussian FWHM = 1 (normalised)
B  = 5 * np.pi                         # peak B-integral (~15.7 rad)
dI = np.gradient(I, t)                # dI/dt
dw = -B * dI                          # δω(t) = −B · dI/dt  (normalised units)

fig, ax1 = plt.subplots(figsize=(4.2, 2.6))
fig.patch.set_facecolor("white")

# ── Intensity ─────────────────────────────────────────────────────────────────
ax1.fill_between(t, I, alpha=0.18, color=PULSE_C)
ax1.plot(t, I, color=PULSE_C, lw=2.2, label="$I(t)$")
ax1.set_ylabel("Intensity  $I(t)$", fontsize=8, color=PULSE_C)
ax1.set_ylim(-0.05, 1.35)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["0", "$I_0$"], fontsize=8, color=PULSE_C)
ax1.tick_params(axis='y', colors=PULSE_C, length=3)
ax1.spines['left'].set_color(PULSE_C)
ax1.spines[['top', 'right', 'bottom']].set_visible(False)

# ── Instantaneous frequency shift (secondary y-axis) ─────────────────────────
ax2 = ax1.twinx()
ax2.plot(t, dw, color=FREQ_C, lw=2.2, linestyle='--', label=r"$\delta\omega(t)$")
ax2.axhline(0, color=FREQ_C, lw=0.6, linestyle=':', alpha=0.5)
ax2.set_ylabel(r"Freq. shift  $\delta\omega(t)$", fontsize=8, color=FREQ_C)
peak = np.max(np.abs(dw))
ax2.set_ylim(-peak * 1.5, peak * 1.5)
ax2.set_yticks([-peak, 0, peak])
ax2.set_yticklabels(["blue", "0", "red"], fontsize=7.5, color=FREQ_C)
ax2.tick_params(axis='y', colors=FREQ_C, length=3)
ax2.spines['right'].set_color(FREQ_C)
ax2.spines[['top', 'left', 'bottom']].set_visible(False)

# ── Equation label ────────────────────────────────────────────────────────────
ax1.text(0, 1.22, r"$\delta\omega(t) = -\frac{n_2\,\omega_0\,L}{c}\frac{dI}{dt}$",
         ha='center', va='bottom', fontsize=8.5, color=BORDER,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))

# ── x-axis ────────────────────────────────────────────────────────────────────
ax1.set_xlim(-2.5, 2.5)
ax1.set_xticks([])
ax1.set_xlabel("Time", fontsize=8, color=BORDER)
ax1.spines['bottom'].set_visible(True)
ax1.spines['bottom'].set_color('#cccccc')
ax1.tick_params(axis='x', bottom=False)

out = "figures/spm_concept.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
