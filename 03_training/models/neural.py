"""
ARIF-Net — Model neural penelitian terdahulu & baseline deep, diimplementasikan ulang pada data, split, dan target
yang sama (r(t,h), direct multi-horizon). Arsitektur mengikuti paper; adaptasi yang wajib dicatat di
docs/MODEL_ADAPTATIONS.md (target r(t,h) pooled 3 komoditas × 8 horizon lewat embedding c & h, lapis keluaran
diinisialisasi nol agar langkah 0 = persistence, panjang window disesuaikan data harian).

  RNN     LSTM / GRU univariat atas sequence harga        — Manogna et al. (2025): 1 layer 128 unit, dropout 0,2
  BiLSTM  BiLSTM harga + fitur tabular multimodal (concat) — baseline deep
  DIALSTM Dual Input Attention LSTM                        — Gu et al. (2022): atensi fitur + atensi temporal atas
          input multivariat (harga + iklim), concat → LSTM 6 unit → dropout 0,2 → FC(10) → FC(1)
  TFT     Temporal Fusion Transformer                      — Amalia et al. (2026) / Lim et al. (2021): GRN, variable
          selection, LSTM encoder, interpretable attention; static = (c, h), known future = kalender tanggal target
  TCN     TCN dilated causal + WeightNorm                  — Zhao et al. (2025): 6 layer, kernel 3, dilasi 1…32,
          dropout 0,3 (tahap 1 TCN-XGBoost)

Kontrak: forward(B) → r̂ terstandardisasi σ_{c,h}; embedding(B) → representasi sebelum kepala keluaran.
Batch B: c, hi, seq (harga), seq_clim (iklim), x_<kelompok> (tabular), x_CALF (kalender tanggal target).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ..data import COMMODITIES, HORIZONS, MULTIMODAL, SEQ_PRICE_CHANNELS


def zero_out(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _Base(nn.Module):
    needs: tuple = ()

    def __init__(self, emb: int):
        super().__init__()
        self.e_c = nn.Embedding(len(COMMODITIES), emb)
        self.e_h = nn.Embedding(len(HORIZONS), emb)

    def ch(self, B):
        return [self.e_c(B["c"]), self.e_h(B["hi"])]

    def forward(self, B):
        return self.head(self.embedding(B)).squeeze(1)


class RNN(_Base):
    """Manogna et al. (2025): sequence harga → LSTM/GRU (1 layer) → Dropout → Dense(1)."""
    needs = ("seq",)

    def __init__(self, cell: str = "lstm", d: int = 128, layers: int = 1, dropout: float = 0.2, emb: int = 8, **_):
        super().__init__(emb)
        Cell = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = Cell(SEQ_PRICE_CHANNELS, d, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = zero_out(nn.Linear(d + 2 * emb, 1))

    def embedding(self, B):
        out, _ = self.rnn(B["seq"])
        return torch.cat([self.drop(out[:, -1])] + self.ch(B), 1)


class BiLSTM(_Base):
    """Baseline deep: BiLSTM harga + seluruh fitur tabular multimodal digabung (concat, tanpa gerbang)."""

    def __init__(self, dims: dict, d: int = 64, dropout: float = 0.2, emb: int = 8, **_):
        super().__init__(emb)
        self.groups = [g for g in MULTIMODAL if dims.get(g)]
        self.needs = ("seq",) + tuple(f"x_{g}" for g in self.groups)
        self.rnn = nn.LSTM(SEQ_PRICE_CHANNELS, d, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(sum(dims[g] for g in self.groups), d)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * d + 2 * emb, d)
        self.head = zero_out(nn.Linear(d, 1))

    def embedding(self, B):
        out, _ = self.rnn(B["seq"])
        tab = nn.functional.gelu(self.proj(torch.cat([B[f"x_{g}"] for g in self.groups], 1)))
        z = self.drop(torch.cat([out[:, -1], tab] + self.ch(B), 1))
        return nn.functional.gelu(self.fc(z))


class DIALSTM(_Base):
    """Gu et al. (2022) DIA-LSTM (Gambar 2–3b, Tabel 2).

    X ∈ R^{T×n} = harga (4 kanal) + iklim eksogen (n_clim + mask) per langkah waktu.
      atensi fitur   : a = softmax(Linear_n(X)) atas sumbu fitur, X_f = a ⊙ X
      atensi temporal: b = softmax(Linear_T(Xᵀ)) atas sumbu waktu, X_t = (b ⊙ Xᵀ)ᵀ
      [X_f ; X_t] → LSTM(units) → Dropout → Flatten → FC(fc) → FC(1)
    """
    needs = ("seq", "seq_clim")

    def __init__(self, L: int, n_clim: int, units: int = 6, fc: int = 10, dropout: float = 0.2, emb: int = 8, **_):
        super().__init__(emb)
        n = SEQ_PRICE_CHANNELS + n_clim
        self.att_f = nn.Linear(n, n)
        self.att_t = nn.Linear(L, L)
        self.lstm = nn.LSTM(2 * n, units, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(L * units + 2 * emb, fc)           # FC(10), aktivasi None (Tabel 2)
        self.head = zero_out(nn.Linear(fc, 1))

    def embedding(self, B):
        X = torch.cat([B["seq"], B["seq_clim"]], 2)             # (B, T, n)
        xf = torch.softmax(self.att_f(X), dim=2) * X
        xt = (torch.softmax(self.att_t(X.transpose(1, 2)), dim=2) * X.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(torch.cat([xf, xt], 2))
        z = torch.cat([self.drop(out).flatten(1)] + self.ch(B), 1)
        return self.fc(z)


class GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021)."""

    def __init__(self, n_in: int, d: int, n_out: int | None = None, ctx: int = 0, dropout: float = 0.1):
        super().__init__()
        n_out = n_out or d
        self.skip = nn.Linear(n_in, n_out) if n_in != n_out else nn.Identity()
        self.fc1 = nn.Linear(n_in + ctx, d)
        self.fc2 = nn.Linear(d, d)
        self.glu = nn.Linear(d, 2 * n_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_out)

    def forward(self, x, c=None):
        h = self.fc1(torch.cat([x, c], -1) if c is not None else x)
        h = self.drop(self.fc2(nn.functional.elu(h)))
        a, b = self.glu(h).chunk(2, -1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))


class InterpretableMHA(nn.Module):
    """Interpretable multi-head attention (Lim et al., 2021 §4.4): value dibagi antar head, bobot dirata-rata."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dk = heads, max(d // heads, 1)
        self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(heads)])
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k):
        v = self.v(k)
        att = torch.stack([torch.softmax(self.q[i](q) @ self.k[i](k).transpose(1, 2) / self.dk ** 0.5, -1)
                           for i in range(self.heads)]).mean(0)
        return self.out(self.drop(att) @ v), att


class TFT(_Base):
    """Temporal Fusion Transformer (Lim et al., 2021), dipakai Amalia et al. (2026) pada harga pangan Indonesia."""
    needs = ("seq", "x_CALF")

    def __init__(self, n_future: int, d: int = 32, heads: int = 4, dropout: float = 0.1, emb: int = 8, **_):
        super().__init__(emb)
        nv = SEQ_PRICE_CHANNELS
        self.static = GRN(2 * emb, d, dropout=dropout)
        self.var_emb = nn.ModuleList([nn.Linear(1, d) for _ in range(nv)])
        self.var_sel = GRN(nv * d, d, n_out=nv, ctx=d, dropout=dropout)
        self.var_grn = nn.ModuleList([GRN(d, d, dropout=dropout) for _ in range(nv)])
        self.h0, self.c0 = nn.Linear(d, d), nn.Linear(d, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate_lstm = GRN(d, d, dropout=dropout)
        self.enrich = GRN(d, d, ctx=d, dropout=dropout)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.post = GRN(d, d, dropout=dropout)
        self.future = GRN(max(n_future, 1), d, ctx=d, dropout=dropout)
        self.fuse = GRN(2 * d, d, dropout=dropout)
        self.head = zero_out(nn.Linear(d, 1))
        self.nv, self.n_future = nv, n_future

    def embedding(self, B):
        seq = B["seq"]
        s = self.static(torch.cat(self.ch(B), 1))
        T = seq.shape[1]
        v = [self.var_emb[i](seq[..., i:i + 1]) for i in range(self.nv)]
        w = torch.softmax(self.var_sel(torch.cat(v, -1), s.unsqueeze(1).expand(-1, T, -1)), -1)
        x = sum(w[..., i:i + 1] * self.var_grn[i](v[i]) for i in range(self.nv))
        out, _ = self.lstm(x, (self.h0(s).unsqueeze(0).contiguous(), self.c0(s).unsqueeze(0).contiguous()))
        out = self.gate_lstm(out + x)
        enr = self.enrich(out, s.unsqueeze(1).expand(-1, T, -1))
        a, _ = self.attn(enr[:, -1:], enr)
        z = self.post(a[:, 0] + enr[:, -1])
        xf = B["x_CALF"] if self.n_future else torch.zeros(len(seq), 1)
        fut = self.future(xf, s)
        return self.fuse(torch.cat([z, fut], 1))


class Chomp(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x[..., :-self.k] if self.k else x


class TCN(_Base):
    """Zhao et al. (2025) Gambar 3: blok residual [Dilated Causal Conv → WeightNorm → ReLU → Dropout] × 2."""
    needs = ("seq",)

    def __init__(self, d: int = 32, kernel: int = 3, levels: int = 6, dropout: float = 0.3, emb: int = 8, **_):
        super().__init__(emb)
        blocks, ch = [], SEQ_PRICE_CHANNELS
        for i in range(levels):
            dil, pad = 2 ** i, (kernel - 1) * 2 ** i
            blocks.append(nn.ModuleDict({
                "net": nn.Sequential(
                    weight_norm(nn.Conv1d(ch, d, kernel, padding=pad, dilation=dil)), Chomp(pad), nn.ReLU(),
                    nn.Dropout(dropout),
                    weight_norm(nn.Conv1d(d, d, kernel, padding=pad, dilation=dil)), Chomp(pad), nn.ReLU(),
                    nn.Dropout(dropout)),
                "res": nn.Conv1d(ch, d, 1) if ch != d else nn.Identity()}))
            ch = d
        self.blocks = nn.ModuleList(blocks)
        self.head = zero_out(nn.Linear(d + 2 * emb, 1))

    def embedding(self, B):
        x = B["seq"].transpose(1, 2)
        for b in self.blocks:
            x = torch.relu(b["net"](x) + b["res"](x))
        return torch.cat([x[..., -1]] + self.ch(B), 1)


def build_neural(name: str, cfg: dict, fd) -> nn.Module:
    m = {k: v for k, v in cfg["model"].items() if k not in ("type", "name")}
    if name in ("lstm", "gru"):
        return RNN(cell=name, **m)
    if name == "bilstm":
        return BiLSTM(dims=fd.dims, **m)
    if name == "dialstm":
        return DIALSTM(L=fd.L, n_clim=fd.dims["CLIM_SEQ"], **m)
    if name == "tft":
        return TFT(n_future=fd.dims.get("CALF", 0), **m)
    if name in ("tcn", "tcn_xgb"):
        return TCN(**m)
    raise KeyError(f"model neural tidak dikenal: {name}")
