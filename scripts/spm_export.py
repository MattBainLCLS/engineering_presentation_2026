"""
Run this script once to generate spm_interactive.html.
The output file can then be embedded in the reveal.js presentation.
"""
import numpy as np
import plotly.graph_objects as go

# ── Physics (same as spm_demo.py) ─────────────────────────────────────────────
N = 8192
t = np.linspace(-8, 8, N)
dt = t[1] - t[0]
tau = 1.0
E0 = np.exp(-t**2 / (2 * tau**2))
I0 = E0**2

freq = np.fft.fftfreq(N, d=dt)
freq = np.fft.fftshift(freq)

FREQ_LIMIT = 3.0
mask = np.abs(freq) < FREQ_LIMIT
freq_display = freq[mask]


def compute_spectrum(B: float) -> np.ndarray:
    E_spm = E0 * np.exp(1j * B * I0)
    S = np.abs(np.fft.fftshift(np.fft.fft(E_spm))) ** 2
    return S / S.max()


# ── Pre-compute traces for each slider step ────────────────────────────────────
B_values = np.linspace(0, 5 * np.pi, 60)

traces = []
for i, B in enumerate(B_values):
    S = compute_spectrum(B)[mask]
    traces.append(
        go.Scatter(
            x=freq_display,
            y=S,
            mode="lines",
            line=dict(color="steelblue", width=2.5),
            visible=(i == 0),
            name=f"B = {B / np.pi:.2f}π rad",
        )
    )

# ── Slider ─────────────────────────────────────────────────────────────────────
steps = [
    dict(
        method="update",
        args=[{"visible": [j == i for j in range(len(traces))]}],
        label=f"{B / np.pi:.1f}π",
    )
    for i, B in enumerate(B_values)
]

sliders = [
    dict(
        active=0,
        steps=steps,
        currentvalue=dict(
            prefix="B-integral = ",
            suffix=" rad",
            font=dict(size=14),
        ),
        pad=dict(t=40, b=10),
        len=0.9,
        x=0.05,
    )
]

# ── Layout ─────────────────────────────────────────────────────────────────────
fig = go.Figure(data=traces)
fig.update_layout(
    sliders=sliders,
    xaxis=dict(
        title="Frequency (normalised to transform limit)",
        range=[-FREQ_LIMIT, FREQ_LIMIT],
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
        font=dict(size=18),
    ),
    showlegend=False,
    template="simple_white",
    margin=dict(t=60, b=100, l=60, r=20),
)

# ── Export ─────────────────────────────────────────────────────────────────────
out = "interactives/spm_interactive.html"
fig.write_html(out, include_plotlyjs="cdn")
print(f"Saved → {out}")
