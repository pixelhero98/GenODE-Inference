from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from genode.models.config import OTFlowConfig


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class ResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(self.out_proj(h))



def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, use_res: bool = True) -> nn.Module:
    if use_res:
        return ResMLP(in_dim, hidden_dim, out_dim, n_blocks=2, dropout=dropout)
    return MLP(in_dim, hidden_dim, out_dim, dropout=dropout)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:hidden_dim] = 1.0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)
        if x.dim() == 3 and params.dim() == 2:
            params = params[:, None, :]
        scale, shift = params.chunk(2, dim=-1)
        return scale * self.norm(x) + shift



def compute_rope_freqs(seq_len: int, dim: int, base: float = 10000.0, device: Optional[torch.device] = None) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    return torch.outer(t, inv_freq)



def apply_rotary_pos_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    return torch.stack([out_real, out_imag], dim=-1).flatten(-2)


class RoPEAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, channels)
        return self.out_proj(out)


class TransformerFUBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.adaln_sa = AdaLN(hidden_dim, hidden_dim)
        self.self_attn = RoPEAttention(hidden_dim, n_heads, dropout=dropout)
        self.adaln_ca = AdaLN(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.adaln_ff = AdaLN(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        ctx_tokens: torch.Tensor,
        adaln_cond: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.adaln_sa(x, adaln_cond)
        if freqs is None:
            freqs = compute_rope_freqs(x.shape[1], self.self_attn.head_dim, device=x.device)
        x = x + self.self_attn(h, freqs)
        h = self.adaln_ca(x, adaln_cond)
        h, _ = self.cross_attn(h, ctx_tokens, ctx_tokens, need_weights=False)
        x = x + h
        h = self.adaln_ff(x, adaln_cond)
        return x + self.ffn(h)


class TransformerFUNet(nn.Module):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.levels = cfg.data.levels
        self.future_block_len = int(max(1, cfg.prediction_horizon))
        self.token_dim = int(getattr(cfg.data, "token_dim", 4))
        if self.token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {self.token_dim}")
        sample_state_dim = int(cfg.sample_state_dim)
        if sample_state_dim % self.token_dim != 0:
            raise ValueError(
                f"sample_state_dim={sample_state_dim} must be divisible by token_dim={self.token_dim}"
            )
        self.seq_len = sample_state_dim // self.token_dim
        self.n_heads = cfg.model.fu_net_heads
        self.in_proj = nn.Linear(self.token_dim, hidden_dim)
        head_dim = hidden_dim // self.n_heads
        self.register_buffer("rope_freqs", compute_rope_freqs(self.seq_len, head_dim), persistent=False)
        self.blocks = nn.ModuleList(
            [TransformerFUBlock(hidden_dim, self.n_heads, cfg.model.dropout) for _ in range(cfg.model.fu_net_layers)]
        )
        self.out_adaln = AdaLN(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.token_dim)

    def _get_rope_freqs(self, device: torch.device) -> torch.Tensor:
        if self.rope_freqs.device != device or self.rope_freqs.shape[0] != self.seq_len:
            head_dim = self.in_proj.out_features // self.n_heads
            self.rope_freqs = compute_rope_freqs(self.seq_len, head_dim, device=device)
        return self.rope_freqs

    def forward(self, x: torch.Tensor, ctx_tokens: torch.Tensor, adaln_cond: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        tokens = x.view(bsz, self.seq_len, self.token_dim)
        h = self.in_proj(tokens)
        freqs = self._get_rope_freqs(h.device)
        for block in self.blocks:
            h = block(h, ctx_tokens, adaln_cond, freqs)
        h = self.out_adaln(h, adaln_cond)
        return self.out_proj(h).reshape(bsz, -1)


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
