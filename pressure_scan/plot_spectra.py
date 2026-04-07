"""
plot_spectra.py
Pressure-scan figure for presentation slide.
5 selected Ar pressures spanning the full FTL range.
Left: stacked normalised spectra.  Right: transform-limited pulse profiles.
Output: ../figures/pressure_scan_tile.svg
"""
import os, zipfile, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

PSI_TO_BAR = 0.0689476

SELECTED_PSI = [0, 100, 180, 240, 300]   # psi

# Viridis-like palette: dark purple → blue → teal → green → yellow-green
COLORS = ["#3B0F70", "#3969AC", "#11A579", "#7CBA5B", "#F2B701"]

BORDER = "#44546A"
BG     = "white"


# ── data loading ──────────────────────────────────────────────────────────────
def load_spz(filepath):
    with zipfile.ZipFile(filepath) as z:
        with z.open(z.namelist()[0]) as f:
            text = f.read().decode('utf-8', errors='replace')
    lines = text.splitlines()
    names = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('Name:'):
            names = line.split('\t')[1:]
        if line.startswith('Wavelength [nm]'):
            data_start = i + 1
            break
    wavelengths, intensities = [], [[] for _ in names]
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.split('\t')
        wavelengths.append(float(parts[0]))
        for j in range(len(names)):
            intensities[j].append(float(parts[j + 1]))
    wavelengths = np.array(wavelengths)
    intensities = [np.array(a) for a in intensities]
    spectra = {}
    for name, intensity in zip(names, intensities):
        m = re.search(r'(\d+(?:\.\d+)?)\s*psi', name, re.IGNORECASE)
        if m:
            spectra[float(m.group(1))] = intensity
    return wavelengths, spectra


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn)


def spectrum_to_tl(wavelengths_nm, intensity):
    """Return (time_fs centred, I_t normalised, fwhm_fs)."""
    c = 2.998e8
    lam = wavelengths_nm * 1e-9
    omega = 2 * np.pi * c / lam
    idx = np.argsort(omega)
    omega_s = omega[idx]
    int_s   = np.clip(intensity[idx], 0, None)

    N = 2048
    omega_u = np.linspace(omega_s[0], omega_s[-1], N)
    int_u   = np.interp(omega_u, omega_s, int_s)
    int_u[int_u < 0.1 * int_u.max()] = 0.0

    E_omega = np.sqrt(int_u)
    N_pad = 65536
    E_pad = np.zeros(N_pad, dtype=complex)
    E_pad[:N] = E_omega

    d_omega = omega_u[1] - omega_u[0]
    E_t = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_pad)))
    I_t = np.abs(E_t) ** 2

    t = np.fft.ifftshift(np.fft.fftfreq(N_pad, d=d_omega / (2 * np.pi))) * 1e15
    centre = t[np.argmax(I_t)]
    t -= centre

    I_t /= I_t.max()
    above = t[I_t >= 0.5]
    fwhm = float(above[-1] - above[0]) if len(above) >= 2 else float('nan')
    return t, I_t, fwhm


# ── load & merge files ────────────────────────────────────────────────────────
files = [
    "Argon ramp 2 low pressures 20.5 mW in.spz",
    "Argon ramp 2 20.5 mW in.spz",
]
all_spectra = {}
wavelengths = None
for fname in files:
    wl, sp = load_spz(fname)
    if wavelengths is None:
        wavelengths = wl
    # higher-pressure file takes priority for duplicate keys
    for k, v in sp.items():
        if k not in all_spectra:
            all_spectra[k] = v

# Wavelength window
mask   = (wavelengths >= 700) & (wavelengths <= 900)
wl_plt = wavelengths[mask]

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(9.0, 4.0))
fig.patch.set_facecolor(BG)

STACK_GAP = 1.1   # vertical offset between stacked spectra

for i, (psi, color) in enumerate(zip(SELECTED_PSI, COLORS)):
    bar = psi * PSI_TO_BAR
    label_bar = f"{bar:.1f} bar"

    raw  = all_spectra[float(psi)]
    norm = normalize(gaussian_filter1d(normalize(raw), sigma=3))
    spec = norm[mask]
    spec = spec / spec.max()

    offset = i * STACK_GAP

    # ── left: stacked spectrum ────────────────────────────────────────────
    ax_s.fill_between(wl_plt, offset, offset + spec,
                      color=color, alpha=0.55, zorder=3)
    ax_s.plot(wl_plt, offset + spec, color=color, lw=1.6, zorder=4)

    ax_s.text(898, offset + 0.07, label_bar,
              ha='right', va='bottom', fontsize=8, color=color, fontweight='bold')

    # ── right: TL pulse ───────────────────────────────────────────────────
    t, I_t, fwhm = spectrum_to_tl(wl_plt, spec)
    t_mask = np.abs(t) < 120
    ax_t.plot(t[t_mask], I_t[t_mask], color=color, lw=2.0,
              label=f"{label_bar}  →  {fwhm:.0f} fs", zorder=3)

# ── left panel cosmetics ──────────────────────────────────────────────────────
ax_s.set_xlim(700, 900)
ax_s.set_ylim(-0.15, len(SELECTED_PSI) * STACK_GAP + 0.2)
ax_s.set_xlabel("Wavelength (nm)", fontsize=9, color=BORDER)
ax_s.set_ylabel("Normalised Intensity (offset)", fontsize=9, color=BORDER)
ax_s.set_title("Spectra vs. Ar pressure", fontsize=10, color=BORDER, pad=6)
ax_s.set_yticks([])
ax_s.spines[['top', 'right', 'left']].set_visible(False)
ax_s.tick_params(colors=BORDER)
ax_s.spines['bottom'].set_color('#cccccc')
ax_s.grid(axis='x', color='#ECEFF1', lw=0.7, zorder=0)

# ── right panel cosmetics ─────────────────────────────────────────────────────
ax_t.set_xlim(-120, 120)
ax_t.set_ylim(-0.05, 1.18)
ax_t.set_xlabel("Time (fs)", fontsize=9, color=BORDER)
ax_t.set_ylabel("Normalised Intensity", fontsize=9, color=BORDER)
ax_t.set_title("Transform-limited pulse duration", fontsize=10, color=BORDER, pad=6)
ax_t.spines[['top', 'right']].set_visible(False)
ax_t.spines[['left', 'bottom']].set_color('#cccccc')
ax_t.tick_params(colors=BORDER)
ax_t.grid(color='#ECEFF1', lw=0.7, zorder=0)
ax_t.legend(title="Ar pressure", title_fontsize=8,
            fontsize=8, loc='upper right', framealpha=0.9,
            handlelength=1.6)

plt.tight_layout()

out = "../figures/pressure_scan_tile.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
