"""One exact-count EDM Inception FID path for all image comparison methods."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm


class FeatureMoments:
    """Merge observed batches in float64; never allocate or count padded samples."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.scatter = None

    def update(self, features) -> None:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
            raise ValueError("Features must be a nonempty finite [samples, dimensions] matrix")
        count = len(values)
        mean = values.mean(0)
        centered = values - mean
        scatter = centered.T @ centered
        if self.count == 0:
            self.count, self.mean, self.scatter = count, mean, scatter
            return
        if mean.shape != self.mean.shape:
            raise ValueError("Feature dimension changed between batches")
        delta = mean - self.mean
        total = self.count + count
        self.scatter += scatter + np.outer(delta, delta) * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("FID covariance requires at least two observed samples")
        return self.mean.copy(), self.scatter / (self.count - 1)


def fid_distance(mean, covariance, reference_mean, reference_covariance) -> float:
    """EDM's Gaussian FID formula with explicit input/numerical validation."""
    mean, reference_mean = (np.asarray(value, dtype=np.float64) for value in (mean, reference_mean))
    covariance, reference_covariance = (
        np.asarray(value, dtype=np.float64) for value in (covariance, reference_covariance)
    )
    if (
        mean.ndim != 1
        or reference_mean.shape != mean.shape
        or covariance.shape != (len(mean), len(mean))
        or reference_covariance.shape != covariance.shape
        or not all(np.isfinite(value).all() for value in (mean, reference_mean, covariance, reference_covariance))
    ):
        raise ValueError("FID moments have incompatible shapes or non-finite values")
    for matrix in (covariance, reference_covariance):
        if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=1e-7):
            raise ValueError("FID covariance must be symmetric")
    root = sqrtm(covariance @ reference_covariance)
    if not np.isfinite(root).all() or (np.iscomplexobj(root) and np.max(np.abs(root.imag)) > 1e-3):
        raise ValueError("FID covariance square root is numerically invalid")
    result = float(np.square(mean - reference_mean).sum() + np.trace(covariance + reference_covariance - 2 * root.real))
    if result < -1e-6:
        raise ValueError("FID is negative beyond numerical tolerance")
    return max(result, 0.0)


def edm_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert centered network outputs exactly as upstream EDM FID extraction."""
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError("Expected finite centered RGB images [batch, 3, height, width]")
    return (255 * ((images + 1) / 2)).clamp(0, 255).to(torch.uint8)


def load_detector(path: str | Path, *, device: str):
    """Load the caller-verified official EDM Inception pickle and freeze it.

    The official StyleGAN ``torch_utils``/``dnnlib`` must be importable (they are
    included in the pinned BézierFlow checkout). Pickles execute Python code:
    callers must verify the downloaded artifact before invoking this loader.
    """
    with Path(path).open("rb") as stream:
        detector = pickle.load(stream)
    return detector.eval().requires_grad_(False).to(device)


def seeded_noise(seeds: Sequence[int], *, device: str, shape=(3, 32, 32)) -> torch.Tensor:
    """Per-image RNG makes image noise invariant to evaluation batch boundaries."""
    if not seeds:
        raise ValueError("At least one image seed is required")
    return torch.stack(
        [
            torch.randn(shape, generator=torch.Generator(device=device).manual_seed(int(seed)), device=device)
            for seed in seeds
        ]
    )


@torch.no_grad()
def evaluate_moments(
    sample: Callable[[torch.Tensor], torch.Tensor],
    detector,
    seeds: Sequence[int],
    *,
    batch_size: int,
    device: str,
) -> FeatureMoments:
    """Generate and count precisely the registered seeds, including the final partial batch."""
    if batch_size < 1 or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires a positive batch size and at least two unique image seeds")
    moments = FeatureMoments()
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        images = sample(seeded_noise(batch_seeds, device=device))
        if len(images) != len(batch_seeds):
            raise ValueError("Sampler returned a different number of images than requested")
        features = detector(edm_uint8(images), return_features=True)
        if features.ndim != 2 or len(features) != len(batch_seeds):
            raise ValueError("Detector returned incompatible feature rows")
        moments.update(features.double().cpu().numpy())
    if moments.count != len(seeds):
        raise RuntimeError("FID count does not match the registered image seed count")
    return moments
