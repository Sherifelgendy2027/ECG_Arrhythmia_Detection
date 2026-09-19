"""
ECG Arrhythmia Detection — FastAPI Backend
===========================================
Serves two prediction pipelines:
  1. Classical ML (SVM/LightGBM) on 11 patient-normalised features
  2. Deep Learning (1D-CNN / CNN-BiLSTM) on raw 216-sample waveforms

Models are loaded once at startup via the lifespan context manager.
If model files are absent (e.g. during Docker build / CI), lightweight
mock predictors are substituted so the app still starts cleanly.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent / "models"
CLASSICAL_MODEL_PATH = MODEL_DIR / "classical_model.pkl"
DL_MODEL_PATH = MODEL_DIR / "dl_model.pt"

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ClassicalFeaturesInput(BaseModel):
    """11 patient-normalised features expected by the classical pipeline."""

    mean_amplitude_norm: float
    std_amplitude_norm: float
    max_amplitude_norm: float
    min_amplitude_norm: float
    peak_to_peak_norm: float
    rr_interval_relative: float
    beat_skewness: float
    beat_kurtosis: float
    zero_crossing_rate: float
    qrs_energy_norm: float
    beat_energy_norm: float

    def to_array(self) -> np.ndarray:
        """Return features as a (1, 11) numpy array in field-declaration order."""
        return np.array(
            [[
                self.mean_amplitude_norm,
                self.std_amplitude_norm,
                self.max_amplitude_norm,
                self.min_amplitude_norm,
                self.peak_to_peak_norm,
                self.rr_interval_relative,
                self.beat_skewness,
                self.beat_kurtosis,
                self.zero_crossing_rate,
                self.qrs_energy_norm,
                self.beat_energy_norm,
            ]],
            dtype=np.float64,
        )


class RawWaveformInput(BaseModel):
    """Raw ECG signal of exactly 216 samples."""

    waveform: List[float]

    @field_validator("waveform")
    @classmethod
    def validate_length(cls, v: List[float]) -> List[float]:
        if len(v) != 216:
            raise ValueError(
                f"Waveform must contain exactly 216 samples, got {len(v)}"
            )
        return v


class PredictionResponse(BaseModel):
    """Unified response schema for both pipelines."""

    prediction: int
    label: str
    probability: float
    model_type: str

# ---------------------------------------------------------------------------
# Deep Learning Model Architectures (Required for unpickling dl_model.pt)
# ---------------------------------------------------------------------------

class ECG1DCNN(nn.Module):
    """1D-CNN: learns beat-morphology filters from raw waveform."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout1d(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout1d(0.2), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout1d(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1),
        )

    def forward(self, x_wave, x_tab=None):
        return self.head(self.conv(x_wave)).squeeze(-1)


class ECGCNNBiLSTM(nn.Module):
    """CNN front-end + BiLSTM."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout1d(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout1d(0.2), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x_wave, x_tab=None):
        x = self.conv(x_wave).transpose(1, 2)  # (batch, seq, 64)
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(h_cat).squeeze(-1)


class SelfAttention1D(nn.Module):
    """Self-Attention Pooling over temporal dimension."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(), nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):  # (batch, seq, feature_dim)
        w = torch.softmax(self.attn(x), dim=1)  # (batch, seq, 1)
        return torch.sum(x * w, dim=1)         # (batch, feature_dim)


class ECGCNNBiLSTMAttention(nn.Module):
    """CNN-BiLSTM with Temporal Self-Attention Pooling."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout1d(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout1d(0.2), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.attn = SelfAttention1D(128)
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x_wave, x_tab=None):
        x = self.conv(x_wave).transpose(1, 2)  # (batch, seq, 64)
        out, _ = self.lstm(x)                  # (batch, seq, 128)
        ctx = self.attn(out)                   # (batch, 128) via attention
        return self.head(ctx).squeeze(-1)

# ---------------------------------------------------------------------------
# Mock Predictors (used when model files are absent)
# ---------------------------------------------------------------------------

class _MockClassicalModel:
    """Returns Normal (0) with probability 0.5 for any input."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class _MockDLModel(nn.Module):
    """Returns Normal (0) with probability 0.5 for any input."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Linear(1, 1)  # keeps PyTorch happy

    def forward(self, x_wave: torch.Tensor, x_tab=None) -> torch.Tensor:
        batch = x_wave.size(0)
        return torch.zeros(batch)  # logit 0.0 -> sigmoid(0.0) = 0.5

# ---------------------------------------------------------------------------
# Global Model Registry (populated in lifespan)
# ---------------------------------------------------------------------------
models: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models into memory before the first request is served."""

    # ── Classical model ──────────────────────────────────────────────────
    if CLASSICAL_MODEL_PATH.exists():
        models["classical"] = joblib.load(CLASSICAL_MODEL_PATH)
        models["classical_is_mock"] = False
        logger.info("Loaded classical model from %s", CLASSICAL_MODEL_PATH)
    else:
        models["classical"] = _MockClassicalModel()
        models["classical_is_mock"] = True
        logger.warning(
            "Classical model not found at %s — using mock predictor",
            CLASSICAL_MODEL_PATH,
        )

    # ── Deep learning model ──────────────────────────────────────────────
    if DL_MODEL_PATH.exists():
        try:
            # PyTorch models saved in Jupyter Notebooks are bound to __main__.
            # When running via uvicorn, __main__ is uvicorn, so torch.load fails.
            # We must inject our classes into __main__ before loading.
            import __main__
            __main__.ECG1DCNN = ECG1DCNN
            __main__.ECGCNNBiLSTM = ECGCNNBiLSTM
            __main__.SelfAttention1D = SelfAttention1D
            __main__.ECGCNNBiLSTMAttention = ECGCNNBiLSTMAttention
            
            # Map storage to CPU in case the model was saved on a GPU/MPS device
            models["dl"] = torch.load(
                DL_MODEL_PATH,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            models["dl"].eval()
            models["dl_is_mock"] = False
            logger.info("Loaded DL model (full) from %s", DL_MODEL_PATH)
        except Exception as e:
            # If that fails, try loading as a state_dict — but we need the
            # model class for that, which is user-defined.  For now, fall
            # back to mock so the app stays up.
            models["dl"] = _MockDLModel().eval()
            models["dl_is_mock"] = True
            logger.warning(
                f"DL model file found but could not be loaded — using mock predictor: {e}"
            )
    else:
        models["dl"] = _MockDLModel().eval()
        models["dl_is_mock"] = True
        logger.warning(
            "DL model not found at %s — using mock predictor",
            DL_MODEL_PATH,
        )

    logger.info("Startup complete — models loaded into memory")
    yield
    logger.info("Shutting down — releasing model memory")
    models.clear()

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description=(
        "Serves classical ML and deep-learning predictions for "
        "ECG arrhythmia classification (Normal vs Abnormal)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

LABEL_MAP = {0: "Normal", 1: "Abnormal"}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Liveness probe for Render / load-balancer health checks."""
    return {
        "status": "healthy",
        "classical_model_loaded": not models.get("classical_is_mock", True),
        "dl_model_loaded": not models.get("dl_is_mock", True),
    }


@app.post("/predict/classical", response_model=PredictionResponse)
async def predict_classical(features: ClassicalFeaturesInput):
    """
    Predict arrhythmia from 11 patient-normalised features
    using the classical ML model (SVM / LightGBM).
    """
    model = models["classical"]
    X = features.to_array()

    try:
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        probability = float(probabilities[1])  # P(Abnormal)
    except Exception as exc:
        logger.exception("Classical prediction failed")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        label=LABEL_MAP.get(prediction, "Unknown"),
        probability=round(probability, 4),
        model_type="classical (mock)" if models["classical_is_mock"] else "classical",
    )


@app.post("/predict/deeplearning", response_model=PredictionResponse)
async def predict_deeplearning(payload: RawWaveformInput):
    """
    Predict arrhythmia from a raw 216-sample ECG waveform
    using the deep learning model (1D-CNN / CNN-BiLSTM).
    """
    model = models["dl"]

    try:
        # Shape: (1, 1, 216) — (batch, channels, seq_len)
        tensor = torch.tensor(
            payload.waveform, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
    except Exception as exc:
        logger.exception("Tensor conversion failed")
        raise HTTPException(
            status_code=400,
            detail=f"Could not convert waveform to tensor: {exc}",
        ) from exc

    try:
        with torch.no_grad():
            logits = model(tensor)
            # Binary classification: 1 logit -> sigmoid
            probability = float(torch.sigmoid(logits).item())
            # We use 0.5 as the default threshold unless a calibrated one is loaded
            prediction = 1 if probability >= 0.5 else 0
    except Exception as exc:
        logger.exception("Deep learning inference failed")
        raise HTTPException(
            status_code=400,
            detail=f"Model inference failed: {exc}",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        label=LABEL_MAP.get(prediction, "Unknown"),
        probability=round(probability, 4),
        model_type="deep_learning (mock)" if models["dl_is_mock"] else "deep_learning",
    )


# ---------------------------------------------------------------------------
# HTML Frontend — browser-testable forms for both endpoints
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ECG Arrhythmia Detection</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --accent: #6c63ff; --accent-hover: #5a52e0;
    --text: #e4e4e7; --text-muted: #9ca3af;
    --success: #22c55e; --danger: #ef4444;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--font); background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
    padding: 2rem 1rem;
  }
  h1 {
    font-size: 1.8rem; font-weight: 700; margin-bottom: .25rem;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { color: var(--text-muted); margin-bottom: 2rem; font-size: .9rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; width: 100%; max-width: 960px; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem; display: flex; flex-direction: column; gap: .75rem;
  }
  .card h2 { font-size: 1.1rem; font-weight: 600; color: var(--accent); }
  .card p { font-size: .8rem; color: var(--text-muted); line-height: 1.4; }
  label { font-size: .78rem; color: var(--text-muted); }
  input, textarea {
    width: 100%; padding: .55rem .75rem; border-radius: 8px; font-size: .85rem;
    border: 1px solid var(--border); background: var(--bg); color: var(--text);
    outline: none; transition: border .2s;
  }
  input:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 80px; font-family: monospace; }
  button {
    padding: .65rem 1.2rem; border: none; border-radius: 8px; cursor: pointer;
    font-size: .85rem; font-weight: 600; background: var(--accent); color: #fff;
    transition: background .2s, transform .1s; align-self: flex-start;
  }
  button:hover { background: var(--accent-hover); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  .result {
    font-size: .82rem; padding: .75rem; border-radius: 8px;
    background: var(--bg); border: 1px solid var(--border);
    white-space: pre-wrap; word-break: break-word; min-height: 2.5rem;
    font-family: monospace; line-height: 1.5;
  }
  .badge {
    display: inline-block; padding: .15rem .5rem; border-radius: 6px;
    font-size: .72rem; font-weight: 600; margin-left: .5rem;
  }
  .badge-normal { background: rgba(34,197,94,.15); color: var(--success); }
  .badge-abnormal { background: rgba(239,68,68,.15); color: var(--danger); }
</style>
</head>
<body>
<h1>&#9829; ECG Arrhythmia Detection</h1>
<p class="subtitle">Test the classical ML and deep learning prediction endpoints</p>
<div class="grid">

  <!-- Classical Model Card -->
  <div class="card" id="classical-card">
    <h2>Classical ML Model</h2>
    <p>Enter the 11 patient-normalised features (comma-separated):<br>
    <em>mean_amp, std_amp, max_amp, min_amp, p2p, rr_rel, skew, kurt,
    zcr, qrs_e, beat_e</em></p>
    <textarea id="classical-input"
      placeholder="0.12, 0.45, 1.02, -0.87, 1.89, 0.95, 0.34, 2.10, 0.08, 0.55, 0.43"></textarea>
    <button onclick="predictClassical()">Predict</button>
    <div class="result" id="classical-result">Awaiting input…</div>
  </div>

  <!-- Deep Learning Model Card -->
  <div class="card" id="dl-card">
    <h2>Deep Learning Model</h2>
    <p>Paste a raw ECG waveform as 216 comma-separated float values.</p>
    <textarea id="dl-input" rows="5"
      placeholder="0.12, -0.04, 0.31, … (216 values total)"></textarea>
    <button onclick="predictDL()">Predict</button>
    <div class="result" id="dl-result">Awaiting input…</div>
  </div>
</div>

<script>
const FEATURE_NAMES = [
  "mean_amplitude_norm","std_amplitude_norm","max_amplitude_norm",
  "min_amplitude_norm","peak_to_peak_norm","rr_interval_relative",
  "beat_skewness","beat_kurtosis","zero_crossing_rate",
  "qrs_energy_norm","beat_energy_norm"
];

function badge(label) {
  const cls = label === "Normal" ? "badge-normal" : "badge-abnormal";
  return `<span class="badge ${cls}">${label}</span>`;
}

async function predictClassical() {
  const el = document.getElementById("classical-result");
  const raw = document.getElementById("classical-input").value.trim();
  if (!raw) { el.textContent = "Please enter feature values."; return; }

  const vals = raw.split(",").map(Number);
  if (vals.length !== 11 || vals.some(isNaN)) {
    el.textContent = "Error: need exactly 11 numeric values."; return;
  }

  const body = {};
  FEATURE_NAMES.forEach((k, i) => body[k] = vals[i]);
  el.textContent = "Running…";

  try {
    const res = await fetch("/predict/classical", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) { el.textContent = "Error: " + (data.detail || res.statusText); return; }
    el.innerHTML =
      `Prediction: ${data.prediction} ${badge(data.label)}\\n` +
      `Probability (Abnormal): ${(data.probability * 100).toFixed(2)}%\\n` +
      `Model: ${data.model_type}`;
  } catch (e) { el.textContent = "Request failed: " + e.message; }
}

async function predictDL() {
  const el = document.getElementById("dl-result");
  const raw = document.getElementById("dl-input").value.trim();
  if (!raw) { el.textContent = "Please paste waveform values."; return; }

  const waveform = raw.split(",").map(Number);
  if (waveform.length !== 216 || waveform.some(isNaN)) {
    el.textContent = "Error: need exactly 216 numeric values."; return;
  }
  el.textContent = "Running…";

  try {
    const res = await fetch("/predict/deeplearning", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({waveform}),
    });
    const data = await res.json();
    if (!res.ok) { el.textContent = "Error: " + (data.detail || res.statusText); return; }
    el.innerHTML =
      `Prediction: ${data.prediction} ${badge(data.label)}\\n` +
      `Probability (Abnormal): ${(data.probability * 100).toFixed(2)}%\\n` +
      `Model: ${data.model_type}`;
  } catch (e) { el.textContent = "Request failed: " + e.message; }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the browser-testable HTML frontend."""
    return HTML_PAGE
