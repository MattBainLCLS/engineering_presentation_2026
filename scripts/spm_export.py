"""
spm_export.py
Generates spm_interactive.html — a Plotly interactive showing SPM spectral
broadening. X-axis: wavelength in nm centred at 1030 nm.
Slider: normalised peak intensity 0 → 1.
"""
import numpy as np
import plotly.graph_objects as go

# ── Physical constants & pulse parameters ─────────────────────────────────────
c_nm_THz = 2.99792458e5    # nm·THz  (speed of light)
lambda0  = 1030.0           # nm  (Yb laser)
nu0_THz  = c_nm_THz / lambda0  # THz ≈ 291 THz

tau_fwhm = 223.0            # fs  (pulse FWHM → 7 nm bandwidth at 1030 nm)
tau      = tau_fwhm / (2 * np.sqrt(np.log(2)))  # 1/e half-width in fs

B_max    = 5 * np.pi        # max B-integral (normalised intensity = 1)

# ── Time & frequency arrays ───────────────────────────────────────────────────
N      = 8192
t      = np.linspace(-1200.0, 1200.0, N)   # fs  (wider window for longer pulse)
dt     = t[1] - t[0]

E0 = np.exp(-t**2 / (2 * tau**2))  # Gaussian field envelope
I0 = E0**2                           # normalised intensity profile

# Frequencies in THz (dt in fs → raw fftfreq in fs⁻¹ = 1000 THz)
freq_rel_THz = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 1e3  # THz offset from ν₀
nu_THz       = nu0_THz + freq_rel_THz                            # absolute THz
lam_nm       = c_nm_THz / nu_THz                                 # nm

# Display window and sort by increasing wavelength
lam_min, lam_max = 900.0, 1200.0
mask     = (lam_nm >= lam_min) & (lam_nm <= lam_max)
lam_disp = np.sort(lam_nm[mask])
sort_idx = np.argsort(lam_nm[mask])


def compute_spectrum(I_norm: float) -> np.ndarray:
    B    = I_norm * B_max
    E    = E0 * np.exp(1j * B * I0)
    S    = np.abs(np.fft.fftshift(np.fft.fft(E))) ** 2
    S   /= S.max()
    return S[mask][sort_idx]


# ── Pre-compute one trace per slider step ─────────────────────────────────────
I_values = np.linspace(0, 1, 60)

traces = []
for i, I_norm in enumerate(I_values):
    traces.append(
        go.Scatter(
            x=lam_disp,
            y=compute_spectrum(I_norm),
            mode="lines",
            line=dict(color="steelblue", width=2.5),
            visible=(i == 0),
            name=f"I = {I_norm:.2f}",
        )
    )

# ── Slider ────────────────────────────────────────────────────────────────────
steps = [
    dict(
        method="update",
        args=[{"visible": [j == i for j in range(len(traces))]}],
        label=f"{v:.2f}",
    )
    for i, v in enumerate(I_values)
]

sliders = [
    dict(
        active=0,
        steps=steps,
        currentvalue=dict(
            prefix="Normalised intensity = ",
            font=dict(size=14),
        ),
        pad=dict(t=65, b=10),
        len=0.9,
        x=0.05,
    )
]

# ── Layout ────────────────────────────────────────────────────────────────────
fig = go.Figure(data=traces)
fig.update_layout(
    sliders=sliders,
    xaxis=dict(
        title="Wavelength (nm)",
        range=[900, 1200],
        showgrid=True,
        gridcolor="#eeeeee",
    ),
    yaxis=dict(
        title="Spectral Intensity (normalised)",
        range=[-0.03, 1.15],
        showgrid=True,
        gridcolor="#eeeeee",
    ),
    title=dict(
        text="Self-Phase Modulation — Spectral Broadening",
        x=0.5,
        font=dict(size=16),
    ),
    showlegend=False,
    template="simple_white",
    margin=dict(t=50, b=130, l=60, r=20),
)

# ── Export ────────────────────────────────────────────────────────────────────
out = "interactives/spm_interactive.html"
fig.write_html(out, include_plotlyjs="cdn")
print(f"Saved → {out}")
