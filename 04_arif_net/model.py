"""
ARIF-Net v1 — arsitektur (docs/ARIFNET_SPEC.md §2).

  Encoder temporal (GRU; ablation E-1: LSTM/TCN) atas sequence harga + proyeksi fitur harga & kalender
  (+ modality eksternal bergerbang, ablation G-5) → z
  Kepala KEJADIAN   (G-1): logit P(r ≠ 0 | z)
  Kepala BESARAN    (G-2): kuantil bersyarat monoton Q̃_τ(z) untuk r ≠ 0
  Komponen DRIFT    (G-3): Q_τ = λ_{c,h}·d(c,h,t)/σ_{c,h} + Q̃_τ, λ = sigmoid(θ_{c,h}) diinisialisasi ≈ 0
  Prediksi titik  = median campuran (1−p)·δ₀ + p·G  (optimal untuk galat absolut, selaras relMAE)

Semua keluaran dalam satuan r/σ_{c,h} (target terstandardisasi, adaptasi A-3).
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import torch
from torch import nn

_data = import_module("03_training.data")
_neural = import_module("03_training.models.neural")
COMMODITIES, HORIZONS, SEQ = _data.COMMODITIES, _data.HORIZONS, _data.SEQ_PRICE_CHANNELS

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
BASE_GROUPS = ("PRICE", "CAL", "CALF")             # input inti: harga + kalender origin/target
EXTERNAL = ("NEWS", "CLIM", "BBM")                  # modality eksternal (ablation G-5)


class _TCNEncoder(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.tcn = _neural.TCN(d=d, dropout=dropout)

    def forward(self, seq):
        x = seq.transpose(1, 2)
        for b in self.tcn.blocks:
            x = torch.relu(b["net"](x) + b["res"](x))
        return x[..., -1]


class ARIFNet(nn.Module):
    def __init__(self, dims: dict, d: int = 64, dropout: float = 0.2, emb: int = 8, encoder: str = "gru",
                 hurdle: bool = True, quantile: bool = True, drift: bool = True, point_loss: str | None = None,
                 modalities: list | None = None, drift_window: int = 365, drift_alpha: float = 0.5, **_):
        super().__init__()
        self.hurdle, self.drift = hurdle, drift
        self.point = point_loss is not None                          # C-4: regresi titik (MSE) tanpa hurdle/kuantil
        self.taus = TAUS if (quantile and not self.point) else (0.5,)
        self.drift_window, self.drift_alpha = int(drift_window), float(drift_alpha)
        self.base = [g for g in BASE_GROUPS if dims.get(g)]
        bad = [g for g in (modalities or []) if g not in EXTERNAL or not dims.get(g)]
        if bad or isinstance(modalities, str):
            raise ValueError(f"modality tidak valid {modalities!r}; pilihan {EXTERNAL} (list)")
        self.ext = list(modalities or [])
        self.needs = ("seq",) + tuple(f"x_{g}" for g in self.base + self.ext) + (("x_DRIFT_RAW",) if drift else ())

        self.e_c, self.e_h = nn.Embedding(len(COMMODITIES), emb), nn.Embedding(len(HORIZONS), emb)
        if encoder == "tcn":
            self.enc = _TCNEncoder(d, dropout)
        else:
            self.enc = (nn.LSTM if encoder == "lstm" else nn.GRU)(SEQ, d, batch_first=True)
        self.encoder = encoder
        self.tab = nn.Linear(sum(dims[g] for g in self.base), d)
        self.ext_proj = nn.ModuleDict({g: nn.Linear(dims[g], d) for g in self.ext})
        self.ext_gate = nn.ModuleDict({g: nn.Linear(2 * d, 1) for g in self.ext})
        self.trunk = nn.Sequential(nn.Linear(2 * d + 2 * emb, d), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d, d), nn.GELU())
        self.event = nn.Linear(d, 1)
        self.q_head = nn.Linear(d, len(self.taus))
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)
        self.lam = nn.Parameter(torch.full((len(COMMODITIES), len(HORIZONS)), -4.0))   # λ ≈ 0,018 di awal

    # ── komponen
    def encode(self, B):
        if self.encoder == "tcn":
            return self.enc(B["seq"])
        out, _ = self.enc(B["seq"])
        return out[:, -1]

    def representation(self, B):
        h = self.encode(B)
        t = nn.functional.gelu(self.tab(torch.cat([B[f"x_{g}"] for g in self.base], 1)))
        for g in self.ext:                                           # G-5: kontribusi eksternal bergerbang
            v = nn.functional.gelu(self.ext_proj[g](B[f"x_{g}"]))
            t = t + torch.sigmoid(self.ext_gate[g](torch.cat([h, v], 1))) * v
        return self.trunk(torch.cat([h, t, self.e_c(B["c"]), self.e_h(B["hi"])], 1))

    def drift_term(self, B, sig_ch: torch.Tensor) -> torch.Tensor:
        """d(c,h,t)/σ_{c,h}; kolom DRIFT_RAW = [slope_365, slope_730, season] (urutan data.feature_columns)."""
        raw = B["x_DRIFT_RAW"]
        slope = raw[:, 0] if self.drift_window == 365 else raw[:, 1]
        h = torch.as_tensor(np.array(HORIZONS), dtype=torch.float32)[B["hi"]]
        d = self.drift_alpha * slope * h + (1 - self.drift_alpha) * raw[:, 2]
        return torch.sigmoid(self.lam[B["c"], B["hi"]]) * d / sig_ch

    def forward(self, B, sig_ch: torch.Tensor) -> dict:
        z = self.representation(B)
        raw = self.q_head(z)
        if len(self.taus) > 1:                                       # kuantil monoton di sekitar median
            mid = len(self.taus) // 2
            up = torch.cumsum(nn.functional.softplus(raw[:, mid + 1:]), 1)
            dn = torch.cumsum(nn.functional.softplus(raw[:, :mid].flip(1)), 1).flip(1)
            q = torch.cat([raw[:, mid:mid + 1] - dn, raw[:, mid:mid + 1], raw[:, mid:mid + 1] + up], 1)
        else:
            q = raw
        if self.drift:
            q = q + self.drift_term(B, sig_ch)[:, None]
        return {"logit": self.event(z).squeeze(1), "q": q}


def mixture_median(p: np.ndarray, q: np.ndarray, taus=TAUS) -> np.ndarray:
    """Median campuran F(x) = (1−p)·1[x ≥ 0] + p·G(x) (SPEC §2.5).

    p: P(r ≠ 0), bentuk (n,). q: kuantil bersyarat G, bentuk (n, k) menaik, pada `taus`.
    G⁻¹ = interpolasi linear antar kuantil, datar di luar [τ_min, τ_max]. Satu kuantil (k = 1) = massa titik.
    """
    p = np.clip(np.asarray(p, float), 1e-9, 1.0)
    q = np.asarray(q, float)
    taus = np.asarray(taus, float)
    n = len(p)
    out = np.zeros(n)

    def G_inv(level, i):
        return q[i, 0] if len(taus) == 1 else float(np.interp(level, taus, q[i]))

    def G_cdf_below0(i):                                            # G(0⁻)
        if len(taus) == 1:
            return 1.0 if q[i, 0] < 0 else 0.0
        if q[i, -1] < 0:
            return 1.0
        if q[i, 0] >= 0:
            return 0.0
        return float(np.interp(0.0, q[i], taus))

    for i in range(n):
        a = p[i] * G_cdf_below0(i)
        if a >= 0.5:
            out[i] = G_inv(0.5 / p[i], i)
        elif a + (1 - p[i]) >= 0.5:
            out[i] = 0.0
        else:
            out[i] = G_inv((p[i] - 0.5) / p[i], i)
    return out


def point_prediction(model: ARIFNet, outs: dict) -> np.ndarray:
    """Prediksi titik (satuan r/σ) sesuai varian."""
    q = outs["q"].detach().numpy()
    if model.point or not model.hurdle:
        return q[:, len(model.taus) // 2]                           # C-1 / C-4: median (atau titik) langsung
    p = torch.sigmoid(outs["logit"]).detach().numpy()
    return mixture_median(p, q, model.taus)
