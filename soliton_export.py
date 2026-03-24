"""
soliton_export.py — generate soliton_interactive.html

CSV layout (soliton_propagation_hz.csv):
  col 0       : frequency axis in Hz (uniform grid, 76–2535 THz)
  cols 1..201 : spectral intensity at each z step (arbitrary units)
  201 z-steps linearly spaced over 0–1 m

Two-panel layout:
  Left  — spectral intensity vs wavelength (nm), normalised per step
  Right — temporal intensity via IFFT of sqrt(S), normalised per step
"""
import numpy as np
import json

# ── Load ──────────────────────────────────────────────────────────────────────
data     = np.genfromtxt("soliton_propagation_hz.csv", delimiter=",")
freq_Hz  = data[:, 0]
spectra  = data[:, 1:]

freq_THz  = freq_Hz / 1e12
wl_nm     = (3e8 / freq_Hz) * 1e9   # wavelength in nm (decreasing order)
N_f      = len(freq_Hz)
N_z      = spectra.shape[1]
z_cm     = np.linspace(0, 100, N_z)

# ── Temporal reconstruction ───────────────────────────────────────────────────
# Frequency grid is perfectly uniform → direct IFFT of sqrt(S) gives
# the transform-limited temporal envelope.
df  = freq_Hz[1] - freq_Hz[0]          # 1.689 THz — verified uniform
dt  = 1.0 / (N_f * df)                 # 0.406 fs per sample
t_fs = (np.arange(N_f) - N_f // 2) * dt * 1e15   # centred time axis in fs

T_LIM = 30.0    # fs display half-width
t_mask   = np.abs(t_fs) <= T_LIM
t_disp   = t_fs[t_mask]

def temporal(spec):
    A  = np.sqrt(np.maximum(spec, 0.0))
    Et = np.fft.fftshift(np.fft.ifft(A))
    It = np.abs(Et) ** 2
    return It

# ── Pre-compute and normalise ─────────────────────────────────────────────────
spec_data = []
temp_data = []

for i in range(N_z):
    s = spectra[:, i]

    # Spectral: normalise to own peak
    s_norm = s / s.max()
    spec_data.append([round(float(v), 6) for v in s_norm])

    # Temporal: normalise to own peak
    It = temporal(s)
    It_disp = It[t_mask] / It.max()
    temp_data.append([round(float(v), 6) for v in It_disp])

# ── Bundle ────────────────────────────────────────────────────────────────────
bundle = {
    "wl":      [round(float(v), 3) for v in wl_nm],   # nm, decreasing
    "t":       [round(float(v), 4) for v in t_disp],
    "spectra": spec_data,
    "temporal":temp_data,
    "z_cm":    [round(float(v), 2) for v in z_cm],
    "N_z":     N_z,
}
data_json = json.dumps(bundle, separators=(",", ":"))

# ── HTML ──────────────────────────────────────────────────────────────────────
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
  #plot {{ width: 100%; height: 460px; }}
  .controls {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 16px;
    padding: 8px 0 2px;
  }}
  .slider-label {{
    font-size: 13px;
    font-weight: 600;
    color: #44546A;
    white-space: nowrap;
  }}
  input[type=range] {{
    width: 400px;
    accent-color: #8C1515;
    cursor: pointer;
  }}
  .slider-value {{
    font-size: 13px;
    color: #333;
    min-width: 72px;
    font-variant-numeric: tabular-nums;
  }}
</style>
</head>
<body>
<div id="plot"></div>
<div class="controls">
  <span class="slider-label">Propagation distance</span>
  <input type="range" id="z-slider" min="0" max="{N_z - 1}" value="0" step="1"/>
  <span class="slider-value" id="z-label">0.00 cm</span>
</div>

<script>
const D = {data_json};

const traces = [
  /* 0 — spectral intensity (left panel) */
  {{
    x: D.wl, y: D.spectra[0],
    type: 'scatter', mode: 'lines',
    line: {{ color: '#8C1515', width: 1.8 }},
    fill: 'tozeroy', fillcolor: 'rgba(140,21,21,0.08)',
    xaxis: 'x', yaxis: 'y',
    hovertemplate: '%{{x:.1f}} nm  I=%{{y:.4f}}<extra></extra>',
  }},
  /* 1 — temporal intensity (right panel) */
  {{
    x: D.t, y: D.temporal[0],
    type: 'scatter', mode: 'lines',
    line: {{ color: '#8C1515', width: 1.8 }},
    fill: 'tozeroy', fillcolor: 'rgba(140,21,21,0.08)',
    xaxis: 'x2', yaxis: 'y2',
    hovertemplate: '%{{x:.1f}} fs  I=%{{y:.4f}}<extra></extra>',
  }},
];

const layout = {{
  annotations: [
    {{
      text: 'Spectral Domain',
      x: 0.5, y: 1.06, xref: 'x domain', yref: 'paper',
      showarrow: false,
      font: {{size: 13, color: '#44546A', family: 'Lato, sans-serif'}},
    }},
    {{
      text: 'Temporal Domain',
      x: 0.5, y: 1.06, xref: 'x2 domain', yref: 'paper',
      showarrow: false,
      font: {{size: 13, color: '#44546A', family: 'Lato, sans-serif'}},
    }},
  ],
  xaxis: {{
    domain: [0, 0.46],
    title: {{ text: 'Wavelength (nm)', font: {{size: 12}} }},
    showgrid: true, gridcolor: '#eeeeee',
    range: [200, 1400],
    zeroline: false,
  }},
  yaxis: {{
    title: {{ text: 'Intensity (norm.)', font: {{size: 11, color: '#8C1515'}} }},
    range: [-0.03, 1.15],
    showgrid: true, gridcolor: '#eeeeee',
    tickfont: {{size: 10, color: '#8C1515'}},
    fixedrange: true,
  }},
  xaxis2: {{
    domain: [0.54, 1.0],
    title: {{ text: 'Time (fs)', font: {{size: 12}} }},
    range: [-25, 25],
    showgrid: true, gridcolor: '#eeeeee',
    zeroline: true, zerolinecolor: '#cccccc',
  }},
  yaxis2: {{
    anchor: 'x2',
    title: {{ text: 'Intensity (norm.)', font: {{size: 11, color: '#8C1515'}} }},
    range: [-0.03, 1.15],
    showgrid: true, gridcolor: '#eeeeee',
    tickfont: {{size: 10, color: '#8C1515'}},
    fixedrange: true,
  }},
  margin: {{t: 55, b: 50, l: 65, r: 30}},
  paper_bgcolor: 'white',
  plot_bgcolor: 'white',
  showlegend: false,
}};

const config = {{ responsive: true, displayModeBar: false }};
Plotly.newPlot('plot', traces, layout, config);

function update() {{
  const iz = parseInt(document.getElementById('z-slider').value);
  Plotly.restyle('plot', {{ y: [D.spectra[iz]]  }}, [0]);
  Plotly.restyle('plot', {{ y: [D.temporal[iz]] }}, [1]);
  document.getElementById('z-label').textContent = D.z_cm[iz].toFixed(2) + ' cm';
}}

document.getElementById('z-slider').addEventListener('input', update);
</script>
</body>
</html>
"""

out = "soliton_interactive.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved → {out}")
print(f"  {N_z} z-steps")
print(f"  Spectral: {N_f} pts, {freq_THz.min():.1f}–{freq_THz.max():.1f} THz")
print(f"  Temporal: {t_mask.sum()} pts, ±{T_LIM:.0f} fs window, {dt*1e15:.3f} fs resolution")
print(f"  JSON: {len(data_json)/1024:.0f} kB")
