import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── Time domain grid ───────────────────────────────────────────────────────────
N = 8192
t = np.linspace(-8, 8, N)           # dimensionless time (units of pulse width)
dt = t[1] - t[0]

tau = 1.0                            # normalised pulse width
E0 = np.exp(-t**2 / (2 * tau**2))   # Gaussian field envelope
I0 = E0**2                           # intensity (peak = 1)

# ── Frequency domain grid ──────────────────────────────────────────────────────
freq = np.fft.fftfreq(N, d=dt)
freq = np.fft.fftshift(freq)

# Display window sized for the transform-limited spectrum;
# wide enough that heavily broadened spectra still look good.
FREQ_LIMIT = 3.0
mask = np.abs(freq) < FREQ_LIMIT


def compute_spectrum(B: float) -> np.ndarray:
    """Return normalised power spectrum after SPM with B-integral = B rad."""
    E_spm = E0 * np.exp(1j * B * I0)
    S = np.abs(np.fft.fftshift(np.fft.fft(E_spm))) ** 2
    return S / S.max()


# ── Figure layout ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
plt.subplots_adjust(bottom=0.22)

S0 = compute_spectrum(0)
(line,) = ax.plot(freq[mask], S0[mask], color="steelblue", lw=2)

ax.set_xlim(-FREQ_LIMIT, FREQ_LIMIT)
ax.set_ylim(-0.03, 1.15)
ax.set_xlabel("Frequency (normalised to transform limit)", fontsize=12)
ax.set_ylabel("Spectral Intensity (normalised)", fontsize=12)
ax.set_title("Self-Phase Modulation — Spectral Broadening", fontsize=13)
ax.grid(True, alpha=0.3)

b_label = ax.text(
    0.98, 0.95, "B = 0.0π rad",
    transform=ax.transAxes, ha="right", va="top", fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.8),
)

# ── Slider ─────────────────────────────────────────────────────────────────────
ax_slider = plt.axes([0.15, 0.08, 0.7, 0.04])
slider = Slider(
    ax_slider, "B-integral", 0.0, 5 * np.pi,
    valinit=0.0, color="steelblue",
)
ax_slider.set_xlabel("(rad)", labelpad=2)


def update(val: float) -> None:
    B = slider.val
    S = compute_spectrum(B)
    line.set_ydata(S[mask])
    b_label.set_text(f"B = {B / np.pi:.2f}π rad")
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.show()
