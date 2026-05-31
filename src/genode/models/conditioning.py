from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from genode.models.config import OTFlowConfig
from genode.models.modules import MLP


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freq = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freq", freq, persistent=False)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        ang = t * self.freq[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class CondEmbedder(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.t_emb = FourierEmbedding(hidden_dim)
        self.t_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, dropout=0.0)
        if cfg.model.cond_dim > 0:
            self.cond_mlp = MLP(cfg.model.cond_dim, hidden_dim, hidden_dim, dropout=0.0)
        else:
            self.cond_mlp = None

    def embed_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.t_mlp(self.t_emb(t))

    def embed_cond(self, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.cond_mlp is None or cond is None:
            return None
        return self.cond_mlp(cond)


class LSTMContextEncoder(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        self.rnn = nn.LSTM(cfg.context_dim, cfg.model.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, (h, _) = self.rnn(ctx)
        return out, h[-1]


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * weights[:, :, None], dim=1)
        return self.out(pooled)


class TransformerContextEncoder(nn.Module):
    def __init__(self, cfg: OTFlowConfig, *, causal: Optional[bool] = None):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.use_causal_mask = bool(cfg.model.ctx_causal if causal is None else causal)
        self.in_proj = nn.Linear(cfg.context_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.data.history_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=cfg.model.ctx_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=cfg.model.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.model.ctx_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPool(hidden_dim)

    def _mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_causal_mask:
            return None
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(ctx)
        if x.shape[1] != self.pos.shape[1]:
            pos = F.interpolate(self.pos.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos
        x = x + pos[:, -x.shape[1] :, :]
        mask = self._mask(x.shape[1], x.device)
        h = self.out_norm(self.enc(x, mask=mask))
        pooled = 0.5 * (self.pool(h) + h[:, -1, :])
        return h, pooled


class HybridContextEncoder(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        kernel = max(1, int(cfg.model.ctx_local_kernel))
        scales = tuple(sorted({int(scale) for scale in cfg.model.ctx_pool_scales if int(scale) > 1}))

        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.scales = scales
        self.in_proj = nn.Linear(cfg.context_dim, hidden_dim)
        self.local_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel,
            groups=hidden_dim,
            padding=max(0, kernel - 1),
        )
        self.local_mix = nn.Linear(hidden_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.data.history_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=cfg.model.ctx_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=cfg.model.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.model.ctx_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.summary_pool = AttentionPool(hidden_dim)
        self.summary_fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.scale_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in self.scales])

    def _mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _position(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.pos.shape[1]:
            return F.interpolate(
                self.pos.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.pos

    def _append_pooled_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.scales:
            return tokens
        bsz, total_steps, hidden_dim = tokens.shape
        pieces = [tokens]
        for proj, scale in zip(self.scale_projs, self.scales):
            if total_steps < scale:
                continue
            remainder = total_steps % scale
            cropped = tokens[:, remainder:, :] if remainder != 0 else tokens
            pooled = cropped.reshape(bsz, -1, scale, hidden_dim).mean(dim=2)
            pieces.append(proj(pooled))
        return torch.cat(pieces, dim=1)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(ctx)
        local = self.local_conv(x.transpose(1, 2))
        local = local[:, :, : x.shape[1]].transpose(1, 2)
        x = x + self.local_mix(F.silu(local))
        x = x + self._position(x)[:, -x.shape[1] :, :]
        h = self.out_norm(self.enc(x, mask=self._mask(x.shape[1], x.device)))
        summary = self.summary_fuse(torch.cat([self.summary_pool(h), h[:, -1, :]], dim=-1))
        return self._append_pooled_tokens(h), summary


class MultiScaleContextEncoder(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        self.base_encoder = TransformerContextEncoder(cfg)
        self.scales = (1, 5, 10)
        self.pool_projs = nn.ModuleList(
            [nn.Linear(cfg.context_dim, cfg.model.hidden_dim) for _ in self.scales[1:]]
        )

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_base, pooled_base = self.base_encoder(ctx)
        bsz, total_steps, state_dim = ctx.shape
        multi_scale_tokens = [h_base]
        for proj, scale in zip(self.pool_projs, self.scales[1:]):
            if total_steps < scale:
                continue
            remainder = total_steps % scale
            ctx_crop = ctx[:, remainder:, :] if remainder != 0 else ctx
            pooled_ctx = ctx_crop.view(bsz, -1, scale, state_dim).mean(dim=2)
            multi_scale_tokens.append(proj(pooled_ctx))
        return torch.cat(multi_scale_tokens, dim=1), pooled_base

def build_context_encoder(cfg: OTFlowConfig) -> nn.Module:
    name = cfg.model.ctx_encoder.lower()
    if name == "lstm":
        return LSTMContextEncoder(cfg)
    if name == "transformer":
        return TransformerContextEncoder(cfg)
    if name == "hybrid":
        return HybridContextEncoder(cfg)
    if name == "multiscale":
        return MultiScaleContextEncoder(cfg)
    raise ValueError(f"Unknown ctx_encoder={cfg.model.ctx_encoder}")


class CrossAttentionConditioner(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=cfg.model.ctx_heads, batch_first=True, dropout=cfg.model.dropout)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))

    def forward(self, q: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        qh = self.q_proj(q)[:, None, :]
        out, _ = self.attn(qh, ctx_tokens, ctx_tokens, need_weights=False)
        return self.out(out[:, 0, :])


@dataclass
class ConditioningState:
    ctx: torch.Tensor
    ctx_summary: torch.Tensor
    t_emb: torch.Tensor
    cond_emb: Optional[torch.Tensor]
    ctx_tokens: torch.Tensor


@dataclass
class ConditioningCache:
    ctx_tokens: torch.Tensor
    ctx_summary: torch.Tensor
    summary: torch.Tensor
    cond_emb: Optional[torch.Tensor] = None


class SharedConditioningBackbone(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = build_context_encoder(cfg)
        self.conditioner = CondEmbedder(cfg)
        self.cross = CrossAttentionConditioner(cfg)
        self.x_proj = nn.Linear(cfg.sample_state_dim, cfg.model.hidden_dim)
        hidden_dim = cfg.model.hidden_dim
        self.summary_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def build_conditioning(
        self,
        hist: torch.Tensor,
        x_ref: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        force_zero_t: bool = False,
        cache: Optional[ConditioningCache] = None,
    ) -> ConditioningState:
        if cache is None:
            cache = self.precompute(hist, cond=cond)
        ctx_tokens = cache.ctx_tokens
        if t is None or force_zero_t:
            t = torch.zeros(x_ref.shape[0], 1, device=x_ref.device)
        t_emb = self.conditioner.embed_t(t)
        cond_emb = cache.cond_emb if (cache.cond_emb is not None or cond is None) else self.conditioner.embed_cond(cond)
        query = self.x_proj(x_ref) + t_emb
        if cond_emb is not None:
            query = query + cond_emb
        local_ctx = self.cross(query, ctx_tokens)
        summary = cache.summary
        gate = torch.sigmoid(self.fusion_gate(torch.cat([query, local_ctx, summary], dim=-1)))
        ctx = gate * local_ctx + (1.0 - gate) * summary
        return ConditioningState(
            ctx=ctx,
            ctx_summary=summary,
            t_emb=t_emb,
            cond_emb=cond_emb,
            ctx_tokens=ctx_tokens,
        )

    def precompute(
        self,
        hist: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
    ) -> ConditioningCache:
        ctx_tokens, ctx_summary = self.context_encoder(hist)
        summary = self.summary_proj(ctx_summary)
        cond_emb = self.conditioner.embed_cond(cond)
        return ConditioningCache(
            ctx_tokens=ctx_tokens,
            ctx_summary=ctx_summary,
            summary=summary,
            cond_emb=cond_emb,
        )
