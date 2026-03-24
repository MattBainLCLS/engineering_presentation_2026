"""
dispersion_export.py — generate dispersion_interactive.html

Interactive demo: Group Delay Dispersion (GDD) and Third Order Dispersion (TOD)
acting on a transform-limited Gaussian ultrafast laser pulse.

Two-panel Plotly figure:
  Left  — Spectral domain: intensity (left y) + spectral phase (right y)
  Right — Temporal domain: intensity vs time

Two HTML range sliders control GDD and TOD independently.
All (GDD × TOD) combinations are pre-computed in Python and embedded as JSON.
"""
import numpy as np
import json

# ── Pulse / grid parameters ───────────────────────────────────────────────────
N       = 4096
T_MAX   = 1000.0          # fs, half-width of time window
tau0    = 25.0            # fs, 1/e² intensity half-width (~30 fs FWHM)

t  = np.linspace(-T_MAX, T_MAX, N)
dt = t[1] - t[0]

# Transform-limited electric field (Gaussian envelope, no chirp)
E0 = np.exp(-t**2 / (2.0 * tau0**2))

# Angular-frequency axis (rad/fs), shifted so DC is at centre
omega = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dt))

# Spectral field of the TL pulse
E_omega = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E0)))

# Spectral intensity — stays constant under pure-phase dispersion
I_omega      = np.abs(E_omega) ** 2
I_omega_norm = I_omega / I_omega.max()

# Peak temporal intensity of TL pulse (used as the normalization reference)
I_TL_peak = np.abs(E0).max() ** 2   # = 1 by construction here

# ── Display masks ─────────────────────────────────────────────────────────────
OMEGA_LIM = 0.22   # rad/fs  (≈ ±5σ_ω for a 25 fs pulse)
T_LIM     = 700.0  # fs

mask_omega = np.abs(omega) <= OMEGA_LIM
mask_t     = np.abs(t)     <= T_LIM

omega_disp = omega[mask_omega]
t_disp     = t[mask_t]
I_omega_disp = I_omega_norm[mask_omega]

# Subsample to keep JSON size manageable
def subsample(arr, max_pts=400):
    if len(arr) <= max_pts:
        return arr
    idx = np.round(np.linspace(0, len(arr)-1, max_pts)).astype(int)
    return arr[idx]

omega_plot   = subsample(omega_disp, 400)
t_plot       = subsample(t_disp,     600)
I_omega_plot = subsample(I_omega_disp, 400)

# Mask indices for subsampled arrays (recompute from scratch for safety)
omega_idx = np.round(np.linspace(0, len(omega_disp)-1, len(omega_plot))).astype(int)
t_idx     = np.round(np.linspace(0, len(t_disp)-1,     len(t_plot))).astype(int)

# ── Slider grid ───────────────────────────────────────────────────────────────
N_GDD = 21
N_TOD = 21

gdd_values = np.linspace(0,    3000, N_GDD)   # fs²
tod_values = np.linspace(-30000, 30000, N_TOD)  # fs³; index 10 → TOD = 0

GDD_INIT = 0          # index
TOD_INIT = N_TOD // 2  # index → TOD = 0

# ── Pre-compute all (GDD, TOD) combinations ───────────────────────────────────
phase_data    = []   # [N_GDD][N_TOD][n_omega_plot]
temporal_data = []   # [N_GDD][N_TOD][n_t_plot]

for gdd in gdd_values:
    row_phase = []
    row_temp  = []
    for tod in tod_values:
        # Spectral phase applied by dispersive medium
        phi = 0.5 * gdd * omega**2 + (1.0/6.0) * tod * omega**3

        # Dispersed field in frequency domain
        E_disp = E_omega * np.exp(1j * phi)

        # Temporal field via inverse FFT
        E_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(E_disp)))
        I_t = np.abs(E_t)**2

        # Normalise temporal intensity to TL peak (shows intensity reduction)
        I_t_norm = I_t / I_TL_peak

        # Spectral phase in display window, subsampled
        phi_disp = phi[mask_omega][omega_idx]

        # Temporal intensity in display window, subsampled
        I_t_disp = I_t_norm[mask_t][t_idx]

        row_phase.append([round(v, 4) for v in phi_disp.tolist()])
        row_temp.append( [round(v, 6) for v in I_t_disp.tolist()])

    phase_data.append(row_phase)
    temporal_data.append(row_temp)

# ── Bundle data for JavaScript ────────────────────────────────────────────────
data_bundle = {
    "omega":      [round(v, 5) for v in omega_plot.tolist()],
    "t":          [round(v, 3) for v in t_plot.tolist()],
    "I_omega":    [round(v, 5) for v in I_omega_plot.tolist()],
    "phase":      phase_data,
    "temporal":   temporal_data,
    "gdd_values": [round(v, 1) for v in gdd_values.tolist()],
    "tod_values": [round(v, 1) for v in tod_values.tolist()],
    "N_GDD":      N_GDD,
    "N_TOD":      N_TOD,
    "gdd_init":   GDD_INIT,
    "tod_init":   TOD_INIT,
}

data_json = json.dumps(data_bundle, separators=(",", ":"))

# ── Build HTML ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Lato', sans-serif;
    background: white;
    padding: 6px 10px 4px;
    overflow: hidden;
  }}
  #plot {{
    width: 100%;
    height: 440px;
  }}
  .controls {{
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 6px 0 2px;
  }}
  .slider-group {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .slider-label {{
    font-size: 13px;
    font-weight: 600;
    color: #44546A;
    min-width: 42px;
  }}
  input[type=range] {{
    width: 220px;
    accent-color: #8C1515;
    cursor: pointer;
  }}
  .slider-value {{
    font-size: 13px;
    color: #333;
    min-width: 88px;
    font-variant-numeric: tabular-nums;
  }}
</style>
</head>
<body>
<div id="plot"></div>
<div class="controls">
  <div class="slider-group">
    <span class="slider-label">GDD</span>
    <input type="range" id="gdd-slider"
           min="0" max="{N_GDD-1}" value="{GDD_INIT}" step="1"/>
    <span class="slider-value" id="gdd-label">0 fs²</span>
  </div>
  <div class="slider-group">
    <span class="slider-label">TOD</span>
    <input type="range" id="tod-slider"
           min="0" max="{N_TOD-1}" value="{TOD_INIT}" step="1"/>
    <span class="slider-value" id="tod-label">0 fs³</span>
  </div>
</div>

<script>
const D = {data_json};

/* ── Initial traces ────────────────────────────────────────────────────────── */
const traces = [
  /* 0 — spectral intensity (constant) */
  {{
    x: D.omega, y: D.I_omega,
    name: 'Spectral Intensity',
    type: 'scatter', mode: 'lines',
    line: {{color: '#8C1515', width: 2.5}},
    xaxis: 'x', yaxis: 'y',
    hovertemplate: '%{{x:.3f}} rad/fs  I=%{{y:.3f}}<extra>Spectral intensity</extra>',
  }},
  /* 1 — spectral phase (updates with sliders) */
  {{
    x: D.omega, y: D.phase[D.gdd_init][D.tod_init],
    name: 'Spectral Phase',
    type: 'scatter', mode: 'lines',
    line: {{color: '#44546A', width: 2, dash: 'dash'}},
    xaxis: 'x', yaxis: 'y2',
    hovertemplate: '%{{x:.3f}} rad/fs  φ=%{{y:.1f}} rad<extra>Spectral phase</extra>',
  }},
  /* 2 — temporal intensity (updates with sliders) */
  {{
    x: D.t, y: D.temporal[D.gdd_init][D.tod_init],
    name: 'Temporal Intensity',
    type: 'scatter', mode: 'lines',
    line: {{color: '#8C1515', width: 2.5}},
    xaxis: 'x2', yaxis: 'y3',
    fill: 'tozeroy',
    fillcolor: 'rgba(140,21,21,0.08)',
    hovertemplate: '%{{x:.1f}} fs  I=%{{y:.4f}}<extra>Temporal intensity</extra>',
  }},
];

/* ── Layout ────────────────────────────────────────────────────────────────── */
const layout = {{
  annotations: [
    {{
      text: 'Spectral Domain',
      x: 0.5, y: 1.06,
      xref: 'x domain', yref: 'paper',
      showarrow: false,
      font: {{size: 13, color: '#44546A', family: 'Lato, sans-serif'}},
    }},
    {{
      text: 'Temporal Domain',
      x: 0.5, y: 1.06,
      xref: 'x2 domain', yref: 'paper',
      showarrow: false,
      font: {{size: 13, color: '#44546A', family: 'Lato, sans-serif'}},
    }},
  ],
  xaxis: {{
    domain: [0, 0.43],
    title: {{ text: 'Δω (rad/fs)', font: {{size: 12}} }},
    showgrid: true, gridcolor: '#eeeeee',
    zeroline: true, zerolinecolor: '#cccccc', zerolinewidth: 1,
  }},
  yaxis: {{
    title: {{ text: 'Spectral Intensity (norm.)', font: {{size: 11, color: '#8C1515'}} }},
    range: [-0.03, 1.18],
    showgrid: true, gridcolor: '#eeeeee',
    tickfont: {{size: 10, color: '#8C1515'}},
    titlefont: {{color: '#8C1515'}},
    fixedrange: true,
  }},
  yaxis2: {{
    title: {{ text: 'Spectral Phase (rad)', font: {{size: 11, color: '#44546A'}} }},
    overlaying: 'y',
    side: 'right',
    range: [0, 100],
    showgrid: false,
    zeroline: true, zerolinecolor: '#cccccc', zerolinewidth: 1,
    tickfont: {{size: 10, color: '#44546A'}},
    titlefont: {{color: '#44546A'}},
    fixedrange: true,
  }},
  xaxis2: {{
    domain: [0.57, 1.0],
    title: {{ text: 'Time (fs)', font: {{size: 12}} }},
    showgrid: true, gridcolor: '#eeeeee',
    zeroline: true, zerolinecolor: '#cccccc', zerolinewidth: 1,
  }},
  yaxis3: {{
    anchor: 'x2',
    title: {{ text: 'Intensity (norm. to TL)', font: {{size: 11, color: '#8C1515'}} }},
    range: [-0.02, 1.12],
    showgrid: true, gridcolor: '#eeeeee',
    tickfont: {{size: 10, color: '#8C1515'}},
    titlefont: {{color: '#8C1515'}},
    fixedrange: true,
  }},
  showlegend: true,
  legend: {{
    x: 0.5, y: 1.20,
    xanchor: 'center',
    orientation: 'h',
    font: {{size: 11}},
    bgcolor: 'rgba(255,255,255,0.9)',
    bordercolor: '#dddddd',
    borderwidth: 1,
  }},
  margin: {{t: 80, b: 50, l: 65, r: 65}},
  paper_bgcolor: 'white',
  plot_bgcolor: 'white',
}};

const config = {{ responsive: true, displayModeBar: false }};
Plotly.newPlot('plot', traces, layout, config);

/* ── Slider update ─────────────────────────────────────────────────────────── */
function update() {{
  const ig = parseInt(document.getElementById('gdd-slider').value);
  const it = parseInt(document.getElementById('tod-slider').value);

  const newPhase    = D.phase[ig][it];
  const newTemporal = D.temporal[ig][it];

  Plotly.restyle('plot', {{ y: [newPhase]    }}, [1]);
  Plotly.restyle('plot', {{ y: [newTemporal] }}, [2]);

  const gddText = D.gdd_values[ig].toFixed(0) + ' fs\u00B2';
  const todText = D.tod_values[it].toFixed(0) + ' fs\u00B3';
  document.getElementById('gdd-label').textContent = gddText;
  document.getElementById('tod-label').textContent = todText;
}}

document.getElementById('gdd-slider').addEventListener('input', update);
document.getElementById('tod-slider').addEventListener('input', update);

// Initialise labels
update();
</script>
</body>
</html>
"""

# ── Write output ──────────────────────────────────────────────────────────────
out = "dispersion_interactive.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved → {out}")
print(f"  Grid: {N_GDD} GDD × {N_TOD} TOD = {N_GDD*N_TOD} combinations")
print(f"  Spectral pts: {len(omega_plot)},  Temporal pts: {len(t_plot)}")
print(f"  Data JSON size: {len(data_json)/1024:.0f} kB")
