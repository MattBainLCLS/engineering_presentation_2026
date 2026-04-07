"""
spm_export.py
Generates spm_interactive.html — a Plotly interactive showing SPM spectral
broadening. X-axis: wavelength in nm centred at 1030 nm.
Slider: normalised peak intensity 0 → 1 in 0.02 steps, ticks at 0.2 intervals.
Uses a custom HTML range input to avoid Plotly's built-in slider limitations.
"""
import json
import numpy as np

# ── Physical constants & pulse parameters ─────────────────────────────────────
c_nm_THz = 2.99792458e5    # nm·THz  (speed of light)
lambda0  = 1030.0           # nm  (Yb laser)
nu0_THz  = c_nm_THz / lambda0  # THz ≈ 291 THz

tau_fwhm = 223.0            # fs  (pulse FWHM → 7 nm bandwidth at 1030 nm)
tau      = tau_fwhm / (2 * np.sqrt(np.log(2)))  # 1/e half-width in fs

B_max    = 5 * np.pi        # max B-integral (normalised intensity = 1)

# ── Time & frequency arrays ───────────────────────────────────────────────────
N      = 8192
t      = np.linspace(-1200.0, 1200.0, N)
dt     = t[1] - t[0]

E0 = np.exp(-t**2 / (2 * tau**2))
I0 = E0**2

freq_rel_THz = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 1e3
nu_THz       = nu0_THz + freq_rel_THz
lam_nm       = c_nm_THz / nu_THz

lam_min, lam_max = 900.0, 1200.0
mask     = (lam_nm >= lam_min) & (lam_nm <= lam_max)
lam_disp = np.sort(lam_nm[mask])
sort_idx = np.argsort(lam_nm[mask])


def compute_spectrum(I_norm: float) -> np.ndarray:
    B  = I_norm * B_max
    E  = E0 * np.exp(1j * B * I0)
    S  = np.abs(np.fft.fftshift(np.fft.fft(E))) ** 2
    S /= S.max()
    return S[mask][sort_idx]


# ── Pre-compute spectra ───────────────────────────────────────────────────────
I_values = np.round(np.arange(0, 1.001, 0.02), 10)   # 0.00, 0.02, …, 1.00
spectra  = [compute_spectrum(v).tolist() for v in I_values]
x_data   = lam_disp.tolist()

# ── Build HTML ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script charset="utf-8"
  src="https://cdn.plot.ly/plotly-3.4.0.min.js"
  integrity="sha256-KEmPoupLpFyGMyGAiOsiNDbKDKAvxXAn/W+oQa0ZAfk="
  crossorigin="anonymous"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: sans-serif; }}
  #plot {{ width: 100%; height: calc(100% - 80px); min-height: 300px; }}
  #slider-wrap {{
    padding: 6px 48px 4px;
    box-sizing: border-box;
  }}
  #current-value {{
    text-align: center;
    font-size: 13px;
    color: #444;
    margin-bottom: 4px;
  }}
  #intensity-slider {{
    width: 100%;
    cursor: pointer;
  }}
  .tick-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #666;
    margin-top: 2px;
    padding: 0 1px;
  }}
</style>
</head>
<body>
  <div id="plot"></div>
  <div id="slider-wrap">
    <div id="current-value">Normalised intensity = 0.00</div>
    <input type="range" id="intensity-slider" min="0" max="{len(I_values)-1}" step="1" value="0">
    <div class="tick-labels">
      <span>0.0</span><span>0.2</span><span>0.4</span><span>0.6</span><span>0.8</span><span>1.0</span>
    </div>
  </div>

<script>
var xData   = {json.dumps(x_data)};
var spectra = {json.dumps(spectra)};
var iValues = {json.dumps([round(float(v), 2) for v in I_values])};

var layout = {{
  xaxis: {{
    title: "Wavelength (nm)",
    range: [900, 1200],
    showgrid: true,
    gridcolor: "#eeeeee"
  }},
  yaxis: {{
    title: "Spectral Intensity (normalised)",
    range: [-0.03, 1.15],
    showgrid: true,
    gridcolor: "#eeeeee"
  }},
  title: {{
    text: "Self-Phase Modulation \u2014 Spectral Broadening",
    x: 0.5,
    font: {{ size: 16 }}
  }},
  showlegend: false,
  template: "simple_white",
  margin: {{ t: 50, b: 50, l: 60, r: 20 }}
}};

Plotly.newPlot("plot", [{{
  x: xData,
  y: spectra[0],
  mode: "lines",
  line: {{ color: "#4298B5", width: 2.5 }}
}}], layout, {{responsive: true}});

var slider  = document.getElementById("intensity-slider");
var display = document.getElementById("current-value");

slider.addEventListener("input", function() {{
  var idx    = parseInt(this.value);
  var I_norm = iValues[idx];
  display.textContent = "Normalised intensity = " + I_norm.toFixed(2);
  Plotly.restyle("plot", {{ y: [spectra[idx]] }}, [0]);
}});
</script>
</body>
</html>"""

out = "interactives/spm_interactive.html"
with open(out, "w") as f:
    f.write(html)
print(f"Saved → {out}")
