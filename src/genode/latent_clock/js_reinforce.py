"""Paper-based Dirichlet/James--Stein comparator (Yu et al., arXiv:2511.22177).

This is a distinct baseline, not a GICO student. See docs/js-reinforce.md for
the reconstruction choices, including the skipped interval and empirical baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

JS_PROTOCOL = "genode_js_reinforce_v1"
MIN_INTERVAL = 2.0**-20


@dataclass(frozen=True)
class JSArchitecture:
    noise_channels: int
    text_dim: int
    target_nfe: int
    pooled_dim: int = 0
    blocks: int = 4
    conv_depth: int = 2
    conv_width: int = 32
    attention_heads: int = 4
    hidden_dim: int = 256

    def __post_init__(self):
        if any(value <= 0 for key, value in asdict(self).items() if key != "pooled_dim"):
            raise ValueError("JS architecture dimensions must be positive.")
        if self.pooled_dim < 0 or self.conv_width % 8 or self.conv_width % self.attention_heads:
            raise ValueError("Invalid JS pooled dimension or convolution/attention widths.")
        if (self.target_nfe + 1) * MIN_INTERVAL >= 1:
            raise ValueError("NFE exceeds the representable JS interval budget.")


class _ImageTextBlock(nn.Module):
    def __init__(self, input_channels, index, config):
        super().__init__()
        layers = []
        for j in range(config.conv_depth):
            channels = config.conv_width * 2 ** min(4, index + j)
            layers.extend((nn.Conv2d(input_channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.SiLU()))
            input_channels = channels
        self.convolutions = nn.Sequential(*layers)
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(config.text_dim, channels)
        self.attention = nn.MultiheadAttention(
            channels,
            config.attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.normalization = nn.LayerNorm(channels)
        self.output_channels = channels

    def forward(self, noise, text, padding_mask):
        value = self.convolutions(noise)
        batch, channels, height, width = value.shape
        query = self.query(value.flatten(2).transpose(1, 2))
        key_value = self.key_value(text)
        attended, _ = self.attention(query, key_value, key_value, key_padding_mask=padding_mask, need_weights=False)
        fused = self.normalization(query + attended)
        return fused.transpose(1, 2).reshape(batch, channels, height, width), fused.mean(1)


class DirichletSchedulePolicy(nn.Module):
    """Noise CNN, native-text cross-attention, and joint Dirichlet head (Table 4)."""

    def __init__(self, architecture: JSArchitecture):
        super().__init__()
        self.architecture = architecture
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels, pooled_channels = architecture.noise_channels, architecture.pooled_dim
        for i in range(architecture.blocks):
            block = _ImageTextBlock(channels, i, architecture)
            self.blocks.append(block)
            channels = block.output_channels
            pooled_channels += channels
            if i + 1 < architecture.blocks:
                next_channels = architecture.conv_width * 2 ** min(4, (i + 1) * architecture.conv_depth - 1)
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(channels, next_channels, 3, stride=2, padding=1),
                        nn.GroupNorm(8, next_channels),
                        nn.SiLU(),
                    )
                )
                channels = next_channels
        self.head = nn.Sequential(
            nn.Linear(pooled_channels, architecture.hidden_dim),
            nn.SiLU(),
            nn.Linear(architecture.hidden_dim, architecture.target_nfe + 1),
        )

    def forward(self, noise, text, pooled=None, padding_mask=None):
        config = self.architecture
        if noise.ndim != 4 or noise.shape[1] != config.noise_channels:
            raise ValueError("JS needs image-shaped native generation noise.")
        if text.ndim != 3 or text.shape[0] != noise.shape[0] or text.shape[-1] != config.text_dim:
            raise ValueError("JS needs native text tokens matching the noise batch.")
        if padding_mask is not None:
            if padding_mask.shape != text.shape[:2] or padding_mask.dtype != torch.bool:
                raise ValueError("JS text padding mask must be boolean [batch, tokens].")
            if padding_mask.all(-1).any():
                raise ValueError("JS text must contain at least one unmasked token.")
        features, value = [], noise.detach().float()
        text = text.detach().float()
        for i, block in enumerate(self.blocks):
            value, summary = block(value, text, padding_mask)
            features.append(summary)
            if i < len(self.downsamples):
                value = self.downsamples[i](value)
        if config.pooled_dim:
            if pooled is None or pooled.shape != (noise.shape[0], config.pooled_dim):
                raise ValueError("JS requires the configured native pooled text features.")
            features.append(pooled.detach().float())
        elif pooled is not None:
            raise ValueError("This JS architecture does not use pooled text.")
        return F.softplus(self.head(torch.cat(features, dim=-1))) + 1e-3


def sample_intervals(concentration: torch.Tensor, *, seed: int, rollouts: int = 1) -> torch.Tensor:
    """Replayable clock draws without changing the generation-noise RNG state."""
    if concentration.ndim != 2 or rollouts < 1 or not torch.isfinite(concentration).all():
        raise ValueError("JS concentrations must be a finite [contexts, intervals] matrix.")
    if (concentration <= 0).any():
        raise ValueError("JS concentrations must be positive.")
    device = concentration.device
    devices = [] if device.type == "cpu" else [device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        # Double precision reduces underflow; the deployment map below still
        # explicitly protects the float32 solver grid, without rejection sampling.
        result = torch.distributions.Dirichlet(concentration.detach().double()).sample((rollouts,))
    return result.transpose(0, 1).contiguous()


@dataclass(frozen=True)
class JSSchedule:
    key: str
    target_nfe: int
    nodes: tuple[float, ...]
    raw_intervals: tuple[float, ...]

    def __post_init__(self):
        nodes = np.asarray(self.nodes, dtype=np.float64)
        if nodes.shape != (self.target_nfe + 1,) or not np.isfinite(nodes).all():
            raise ValueError("JS schedule requires L+1 finite solver nodes.")
        if not 0 < nodes[0] < 1 or nodes[-1] != 1 or np.any(np.diff(nodes) <= 0):
            raise ValueError("JS skips the initial interval and integrates monotonically to the terminal endpoint.")


def materialize_js(intervals, *, key: str = "js") -> JSSchedule:
    """Evaluate at t_1,...,t_L, followed by zero: exactly L Euler calls.

    The initial interval is skipped. The sampled Gaussian latent is unchanged.
    This explicit interpretation follows the L-timestep policy definition and
    Table 4's skipped interval; it is not a claim about unreleased author code.
    """
    raw = np.asarray(intervals, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2 or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("JS interval actions must be a positive finite vector.")
    if not np.isclose(raw.sum(), 1, rtol=0, atol=1e-6) or raw.size * MIN_INTERVAL >= 1:
        raise ValueError("JS actions must sum to one and fit the numerical interval budget.")
    mass = (1 - raw.size * MIN_INTERVAL) * (raw / raw.sum()) + MIN_INTERVAL
    nodes = np.cumsum(mass)
    nodes[-1] = 1.0
    sigmas = (1 - nodes).astype(np.float32)
    times = sigmas[:-1] * np.float32(1000)
    if np.any(np.diff(sigmas) >= 0) or np.any(np.diff(times) >= 0):
        raise ValueError("JS grid collapsed in native float32 precision.")
    return JSSchedule(str(key), raw.size - 1, tuple(nodes), tuple(raw))


def james_stein_baseline(rewards: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Literal empirical Eqs. (4)--(10), for a balanced B-by-K rollout batch.

    The batch-estimated shrinkage coefficient depends on the current rewards.
    Detachment is necessary for REINFORCE but does not establish unbiasedness
    of that empirical estimator. We reproduce it without making that claim.
    """
    value = rewards.detach().double()
    if value.ndim != 2 or value.shape[0] < 3 or value.shape[1] < 2 or not torch.isfinite(value).all():
        raise ValueError("JS shrinkage needs at least three contexts and two finite rewards per context.")
    batch, count = value.shape
    rloo = (value.sum(1, keepdim=True) - value) / (count - 1)
    cross = (value.sum() - value) / (value.numel() - 1)
    within = (value - rloo).square().sum() / (batch * (count - 1))
    between = ((value.mean(1) - cross.mean(1)).square().sum() / (batch - 1) - within / count).clamp_min(0)
    variance = within / (count - 1)
    denominator = variance + between
    shrinkage = torch.where(denominator > 0, variance / denominator.clamp_min(torch.finfo(value.dtype).tiny), 0.0)
    baseline = (1 - shrinkage) * rloo + shrinkage * cross
    return baseline, {
        "within_variance": within.item(),
        "between_variance": between.item(),
        "shrinkage": shrinkage.item(),
    }


def reinforce_loss(concentration, intervals, rewards):
    """Whole-clock REINFORCE; no PPO clipping, entropy bonus or KL term."""
    if intervals.ndim != 3 or intervals.shape[::2] != concentration.shape or rewards.shape != intervals.shape[:2]:
        raise ValueError("JS concentrations/actions/rewards must have shapes [B,D], [B,K,D], [B,K].")
    baseline, diagnostics = james_stein_baseline(rewards)
    distribution = torch.distributions.Dirichlet(concentration.double()[:, None, :])
    log_prob = distribution.log_prob(intervals.detach().double())
    advantage = rewards.detach().double() - baseline
    loss = -(advantage * log_prob).mean()
    if not torch.isfinite(loss):
        raise ValueError("JS REINFORCE loss is not finite.")
    return loss, diagnostics
