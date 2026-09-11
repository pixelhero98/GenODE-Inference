"""One shared Transformer design for each GICO role."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DENSITY_BINS = 64
DENSITY_MIXTURE = 1e-8
RATIO_COUNT = DENSITY_BINS - 1


def guarded_mass(mass: Tensor) -> Tensor:
    if mass.shape[-1] != DENSITY_BINS or not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError("Density must contain 64 finite nonnegative masses.")
    if not bool(torch.allclose(mass.sum(-1), torch.ones_like(mass.sum(-1)), atol=1e-6, rtol=1e-6)):
        raise ValueError("Density masses must sum to one.")
    return (1 - DENSITY_MIXTURE) * mass.double() + DENSITY_MIXTURE / DENSITY_BINS


@dataclass(frozen=True)
class ModelConfig:
    condition_dim: int
    metric_count: int
    width: int = 128
    layers: int = 2
    heads: int = 4
    feedforward: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.width not in (64, 128) or (self.layers, self.heads, self.feedforward) != (2, 4, 256):
            raise ValueError("Unified GICO requires width64 or width128, layers2/heads4/feedforward256.")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (self.condition_dim, self.metric_count)):
            raise ValueError("Condition width and metric count must be positive integers.")
        if not 0 <= self.dropout <= 0.1:
            raise ValueError("Dropout must be between zero and 0.1.")

    def to_payload(self) -> dict:
        return asdict(self)


class _Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.condition = nn.Sequential(
            nn.Linear(config.condition_dim, config.width), nn.GELU(), nn.Linear(config.width, config.width)
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    config.width,
                    config.heads,
                    config.feedforward,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.layers)
            ]
        )
        self.norm = nn.LayerNorm(config.width)

    def encode(self, condition: Tensor, tokens: Tensor, *, causal: bool = False) -> Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.config.condition_dim:
            raise ValueError("Condition tensor has the wrong width.")
        if condition.shape[0] != tokens.shape[0] or not bool(torch.isfinite(condition).all()):
            raise ValueError("Conditions must be finite and match the density batch.")
        projected = self.condition(condition)[:, None]
        mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).triu(1)
        for block in self.blocks:
            tokens = block(tokens + projected, src_mask=mask if causal else None)
        return self.norm(tokens)


def _geometry() -> Tensor:
    return torch.stack(
        ((torch.arange(DENSITY_BINS) + 0.5) / DENSITY_BINS, torch.full((DENSITY_BINS,), 1 / DENSITY_BINS)), dim=-1
    )


class DensityTeacher(_Transformer):
    def __init__(self, config: ModelConfig, density_mean=None, density_scale=None) -> None:
        super().__init__(config)
        self.register_buffer(
            "density_mean", torch.zeros(DENSITY_BINS) if density_mean is None else density_mean.float()
        )
        self.register_buffer(
            "density_scale", torch.ones(DENSITY_BINS) if density_scale is None else density_scale.float()
        )
        self.validate_density_normalization()

        self.register_buffer("geometry", _geometry(), persistent=False)
        self.input = nn.Linear(3, config.width)
        self.output = nn.Linear(config.width, config.metric_count)

    def validate_density_normalization(self):
        if self.density_mean.shape != (DENSITY_BINS,) or self.density_scale.shape != (DENSITY_BINS,):
            raise ValueError("Teacher density normalizers require 64 coordinates.")
        if not bool(torch.isfinite(self.density_mean).all() and torch.isfinite(self.density_scale).all()) or bool(
            (self.density_scale <= 0).any()
        ):
            raise ValueError("Teacher density normalizers must be finite with positive scales.")

    def forward(self, condition: Tensor, mass: Tensor) -> Tensor:
        # Do not use no_grad here: students need the teacher's density gradient.
        log_mass = guarded_mass(mass).log().to(condition.dtype)
        log_mass = (log_mass - self.density_mean) / self.density_scale
        features = torch.cat((log_mass[..., None], self.geometry[None].expand(len(mass), -1, -1)), dim=-1)
        return self.output(self.encode(condition, self.input(features)).mean(dim=1))


class DeterministicStudent(_Transformer):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.register_buffer("geometry", _geometry(), persistent=False)
        self.input = nn.Linear(2, config.width)
        self.output = nn.Linear(config.width, 1)

    def forward(self, condition: Tensor) -> Tensor:
        queries = self.input(self.geometry)[None].expand(len(condition), -1, -1)
        return self.output(self.encode(condition, queries)).squeeze(-1).double().softmax(-1)


class StochasticStudent(_Transformer):
    """Continuous causal Gaussian log ratios, with no finite-support decoder."""

    def __init__(
        self, config: ModelConfig, ratio_mean: Tensor | None = None, ratio_scale: Tensor | None = None
    ) -> None:
        super().__init__(config)
        self.register_buffer("ratio_mean", torch.zeros(RATIO_COUNT) if ratio_mean is None else ratio_mean.float())
        self.register_buffer("ratio_scale", torch.ones(RATIO_COUNT) if ratio_scale is None else ratio_scale.float())
        if self.ratio_mean.shape != (RATIO_COUNT,) or self.ratio_scale.shape != (RATIO_COUNT,):
            raise ValueError("Log-ratio normalizers must have 63 coordinates.")
        if not bool(torch.isfinite(self.ratio_mean).all() and torch.isfinite(self.ratio_scale).all()):
            raise ValueError("Log-ratio normalizers must be finite.")
        if bool((self.ratio_scale <= 0).any()):
            raise ValueError("Log-ratio scales must be positive.")
        self.input = nn.Linear(1, config.width)
        self.position = nn.Embedding(RATIO_COUNT, config.width)
        self.output = nn.Linear(config.width, 2)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([0.0, -3.6375862]))

    def ratios(self, mass: Tensor) -> Tensor:
        log_mass = guarded_mass(mass).log()
        return ((log_mass[..., :-1] - log_mass[..., -1:]) - self.ratio_mean) / self.ratio_scale

    def density(self, ratios: Tensor) -> Tensor:
        if ratios.shape[-1] != RATIO_COUNT or not bool(torch.isfinite(ratios).all()):
            raise ValueError("A complete stochastic density requires 63 finite log ratios.")
        values = ratios.double() * self.ratio_scale + self.ratio_mean
        return torch.cat((values, torch.zeros_like(values[..., :1])), dim=-1).softmax(-1)

    def _distribution_head(self, condition: Tensor, shifted: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(shifted.shape[1], device=condition.device)
        tokens = self.input(shifted.to(condition.dtype)[..., None]) + self.position(positions)[None]
        values = self.output(self.encode(condition, tokens, causal=True))
        return values[..., 0], 0.05 + 1.95 * values[..., 1].sigmoid()

    def conditional_parameters(self, condition: Tensor, ratios: Tensor) -> tuple[Tensor, Tensor]:
        if ratios.shape != (len(condition), RATIO_COUNT):
            raise ValueError("Teacher-forced ratios must have shape [batch,63].")
        shifted = torch.cat((torch.zeros_like(ratios[:, :1]), ratios[:, :-1]), dim=1)
        return self._distribution_head(condition, shifted)

    def nll(self, condition: Tensor, target_ratios: Tensor) -> Tensor:
        mean, std = self.conditional_parameters(condition, target_ratios)
        return -torch.distributions.Normal(mean, std).log_prob(target_ratios).mean(-1)

    def sample(
        self, condition: Tensor, *, generator: torch.Generator | None = None, innovations: Tensor | None = None
    ) -> Tensor:
        if innovations is None:
            if generator is None:
                raise ValueError("Stochastic sampling requires an explicit clock RNG or innovations.")
            innovations = torch.randn(
                len(condition), RATIO_COUNT, generator=generator, device=condition.device, dtype=condition.dtype
            )
        if innovations.shape != (len(condition), RATIO_COUNT) or not bool(torch.isfinite(innovations).all()):
            raise ValueError("Clock innovations must be finite with shape [batch,63].")
        prefix = condition.new_zeros((len(condition), 0))
        for index in range(RATIO_COUNT):
            shifted = torch.cat((condition.new_zeros((len(condition), 1)), prefix), dim=1)
            mean, std = self._distribution_head(condition, shifted)
            value = mean[:, -1] + std[:, -1] * innovations[:, index]
            prefix = torch.cat((prefix, value[:, None]), dim=1)
        return self.density(prefix)


def density_kl(target: Tensor, prediction: Tensor) -> Tensor:
    return F.kl_div(guarded_mass(prediction).log(), guarded_mass(target), reduction="none").sum(-1)
