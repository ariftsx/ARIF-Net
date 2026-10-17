"""
ARIF-Net — Model neural dari penelitian terdahulu:

  LSTM/GRU    — Manogna et al. (2025): univariat sequence harga
  BiLSTM      — Baseline deep: BiLSTM harga + fitur tabular multimodal (concat)
  DIA-LSTM    — Gu et al. (2022): input attention + temporal attention
  TFT         — Amalia et al. (2026): Temporal Fusion Transformer
  TCN         — Zhao et al. (2025): Temporal Convolutional Network (tahap 1 TCN-XGBoost)

Semua diimplementasikan dengan aturan pelatihan yang sama (training.py).
"""
from __future__ import annotations

import torch
from torch import nn


COMMODITIES = ("com_14", "com_11", "com_3")
HORIZONS = (1, 3, 7, 14, 30, 90, 180, 365)


def mlp(n_in: int, n_out: int, d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_out))


def zero_out(head: nn.Sequential):
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)


class RNN(nn.Module):
    """Manogna et al. (2025): LSTM/GRU pada riwayat harga saja."""

    def __init__(self, cell: str = "lstm", d: int = 32, layers: int = 1,
                 dropout: float = 0.1, emb: int = 8, seq_features: int = 4, **_):
        super().__init__()
        self.e_c = nn.Embedding(len(COMMODITIES), emb)
        self.e_h = nn.Embedding(len(HORIZONS), emb)
        Cell = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = Cell(seq_features, d, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.head = mlp(d + 2 * emb, 1, d, dropout)
        zero_out(self.head)
        self.name = cell

    def forward(self, seq, c_idx, h_idx):
        out, _ = self.rnn(seq)
        z = torch.cat([out[:, -1], self.e_c(c_idx), self.e_h(h_idx)], 1)
        return self.head(z).squeeze(1)


class BiLSTM(nn.Module):
    """Baseline deep: BiLSTM harga + fitur tabular multimodal (concat)."""

    def __init__(self, n_tabular: int, d: int = 32, dropout: float = 0.1,
                 emb: int = 8, seq_features: int = 4, **_):
        super().__init__()
        self.e_c = nn.Embedding(len(COMMODITIES), emb)
        self.e_h = nn.Embedding(len(HORIZONS), emb)
        self.rnn = nn.LSTM(seq_features, d, batch_first=True, bidirectional=True)
        self.tab_proj = nn.Linear(n_tabular, d)
        self.head = mlp(2 * d + d + 2 * emb, 1, 2 * d, dropout)
        self.drop = nn.Dropout(dropout)
        zero_out(self.head)
        self.name = "bilstm"

    def forward(self, seq, x_tab, c_idx, h_idx):
        out, _ = self.rnn(seq)
        tab = nn.functional.gelu(self.tab_proj(x_tab))
        z = self.drop(torch.cat([out[:, -1], tab, self.e_c(c_idx), self.e_h(h_idx)], 1))
        return self.head(z).squeeze(1)


class DIALSTM(nn.Module):
    """Gu et al. (2022) DIA-LSTM: input attention + temporal attention atas variabel iklim."""

    def __init__(self, n_climate: int, d: int = 32, dropout: float = 0.1,
                 emb: int = 8, seq_features: int = 4, **_):
        super().__init__()
        self.e_c = nn.Embedding(len(COMMODITIES), emb)
        self.e_h = nn.Embedding(len(HORIZONS), emb)
        self.lstm = nn.LSTM(seq_features, d, batch_first=True)
        self.att_in = nn.Sequential(nn.Linear(d + emb, d), nn.Tanh(), nn.Linear(d, n_climate))
        self.clim_proj = nn.Linear(n_climate, d)
        self.att_t = nn.Linear(d, d, bias=False)
        self.head = mlp(3 * d + 2 * emb, 1, d, dropout)
        zero_out(self.head)
        self.name = "dialstm"

    def forward(self, seq, x_clim, c_idx, h_idx):
        out, _ = self.lstm(seq)
        h_last = out[:, -1]
        ec = self.e_c(c_idx)
        # Input attention
        alpha = torch.softmax(self.att_in(torch.cat([h_last, ec], 1)), 1)
        x_att = nn.functional.gelu(self.clim_proj(alpha * x_clim))
        # Temporal attention
        score = torch.softmax(torch.einsum("btd,bd->bt", out, self.att_t(h_last)), 1)
        ctx = torch.einsum("bt,btd->bd", score, out)
        z = torch.cat([h_last, ctx, x_att, ec, self.e_h(h_idx)], 1)
        return self.head(z).squeeze(1)


class GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021) — dipakai TFT."""

    def __init__(self, n_in: int, d: int, n_out: int | None = None,
                 ctx: int = 0, dropout: float = 0.1):
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


class TFT(nn.Module):
    """Amalia et al. (2026): Temporal Fusion Transformer."""

    def __init__(self, n_calendar: int, d: int = 32, heads: int = 4,
                 dropout: float = 0.1, emb: int = 8, seq_features: int = 4, **_):
        super().__init__()
        self.e_c = nn.Embedding(len(COMMODITIES), emb)
        self.e_h = nn.Embedding(len(HORIZONS), emb)
        self.static = GRN(2 * emb, d, dropout=dropout)
        self.var_emb = nn.ModuleList([nn.Linear(1, d) for _ in range(seq_features)])
        self.var_sel = GRN(seq_features * d, d, n_out=seq_features, ctx=d, dropout=dropout)
        self.var_grn = nn.ModuleList([GRN(d, d, dropout=dropout) for _ in range(seq_features)])
        self.h0 = nn.Linear(d, d)
        self.c0 = nn.Linear(d, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate_lstm = GRN(d, d, dropout=dropout)
        self.enrich = GRN(d, d, ctx=d, dropout=dropout)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.post = GRN(d, d, dropout=dropout)
        self.future = nn.Linear(n_calendar, d)
        self.head = mlp(2 * d, 1, d, dropout)
        zero_out(self.head)
        self.name = "tft"
        self.n_seq = seq_features

    def forward(self, seq, x_cal, c_idx, h_idx):
        s = self.static(torch.cat([self.e_c(c_idx), self.e_h(h_idx)], 1))
        T = seq.shape[1]
        v = [self.var_emb[i](seq[..., i:i + 1]) for i in range(self.n_seq)]
        flat = torch.cat(v, -1)
        w = torch.softmax(self.var_sel(flat, s.unsqueeze(1).expand(-1, T, -1)), -1)
        x = sum(w[..., i:i + 1] * self.var_grn[i](v[i]) for i in range(self.n_seq))
        out, _ = self.lstm(x, (self.h0(s).unsqueeze(0).contiguous(),
                               self.c0(s).unsqueeze(0).contiguous()))
        out = self.gate_lstm(out + x)
        enr = self.enrich(out, s.unsqueeze(1).expand(-1, T, -1))
        a, _ = self.attn(enr[:, -1:], enr, enr)
        z = self.post(a[:, 0] + enr[:, -1])
        fut = nn.functional.gelu(self.future(x_cal))
        return self.head(torch.cat([z, fut], 1)).squeeze(1)


class Chomp(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x[..., :-self.k] if self.k else x


class TCN(nn.Module):
    """Zhao et al. (2025): TCN dilated causal atas sequence harga."""

    def __init__(self, d: int = 32, kernel: int = 3, levels: int = 5,
                 dropout: float = 0.1, emb: int = 8, seq_features: int = 4, **_):
        super().__init__()
        self.e_c = nn.Embedding(len(COMMODITIES), emb)
        self.e_h = nn.Embedding(len(HORIZONS), emb)
        blocks, ch = [], seq_features
        for i in range(levels):
            dil = 2 ** i
            pad = (kernel - 1) * dil
            blocks.append(nn.ModuleDict({
                "net": nn.Sequential(
                    nn.Conv1d(ch, d, kernel, padding=pad, dilation=dil), Chomp(pad), nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d, d, kernel, padding=pad, dilation=dil), Chomp(pad), nn.ReLU(),
                    nn.Dropout(dropout)),
                "res": nn.Conv1d(ch, d, 1) if ch != d else nn.Identity()}))
            ch = d
        self.blocks = nn.ModuleList(blocks)
        self.head = mlp(d + 2 * emb, 1, d, dropout)
        zero_out(self.head)
        self.name = "tcn"
        self.embed_dim = d

    def embedding(self, seq, c_idx, h_idx):
        x = seq.transpose(1, 2)
        for b in self.blocks:
            x = torch.relu(b["net"](x) + b["res"](x))
        return torch.cat([x[..., -1], self.e_c(c_idx), self.e_h(h_idx)], 1)

    def forward(self, seq, c_idx, h_idx):
        z = self.embedding(seq, c_idx, h_idx)
        return self.head(z).squeeze(1)


def build_neural(name: str, cfg: dict, n_tabular: int = 0,
                 n_climate: int = 0, n_calendar: int = 0) -> nn.Module:
    """Factory: buat model neural berdasarkan nama dan config."""
    m = cfg.get("model", cfg)
    if name in ("lstm", "gru"):
        return RNN(cell=name, **m)
    if name == "bilstm":
        return BiLSTM(n_tabular=n_tabular, **m)
    if name == "dialstm":
        return DIALSTM(n_climate=n_climate, **m)
    if name == "tft":
        return TFT(n_calendar=n_calendar, **m)
    if name in ("tcn", "tcn_xgb"):
        return TCN(**m)
    raise KeyError(f"Unknown neural model: {name}")
