"""Frozen pooled-context and continuous solver-budget features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genode.solver_protocol import normalize_solver_key, solver_macro_steps


@dataclass(frozen=True)
class EmbeddingNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or not self.mean.size or self.std.shape != self.mean.shape:
            raise ValueError("Normalizer vectors must be nonempty and have matching widths.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Normalizer requires finite means and positive finite scales.")

    @classmethod
    def fit(cls, embeddings, context_ids):
        ids = sorted(set(context_ids))
        if not ids:
            raise ValueError("Normalization requires training contexts.")
        values = np.array([embeddings[key] for key in ids], dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Context embeddings must be a finite rectangular matrix.")
        std = values.std(0)
        return cls(values.mean(0), np.where(std < 1e-6, 1, std))

    def transform_one(self, embedding):
        values = np.asarray(embedding, dtype=np.float64)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("Context embedding has invalid width or nonfinite entries.")
        return ((values - self.mean) / self.std).astype(np.float32)

    def transform_table(self, embeddings):
        return {key: self.transform_one(value) for key, value in embeddings.items()}

    def to_payload(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_payload(cls, payload):
        return cls(np.array(payload["mean"], dtype=float), np.array(payload["std"], dtype=float))


@dataclass(frozen=True)
class Conditioning:
    context: EmbeddingNormalizer
    settings: EmbeddingNormalizer
    solvers: tuple[str, ...]
    unconditional: bool = False

    @staticmethod
    def budget(solver: str, nfe: int) -> np.ndarray:
        return np.log1p([nfe, solver_macro_steps(solver, nfe)])

    @classmethod
    def fit(cls, rows: list[dict], contexts: dict, *, unconditional: bool = False):
        if not rows or any(r["split"] != "train" for r in rows):
            raise ValueError("Condition normalization requires training rows only.")
        solvers = tuple(sorted({normalize_solver_key(r["solver"]) for r in rows}))
        settings = {(r["solver"], r["nfe"]): cls.budget(r["solver"], r["nfe"]) for r in rows}
        context = EmbeddingNormalizer.fit(contexts, [r["context_id"] for r in rows])
        if unconditional and any(np.any(np.asarray(contexts[r["context_id"]]) != 0) for r in rows):
            raise ValueError("Unconditional training requires explicit zero contexts.")
        return cls(context, EmbeddingNormalizer.fit(settings, settings), solvers, unconditional)

    @property
    def width(self) -> int:
        return len(self.context.mean) + len(self.solvers) + 2

    def transform(self, context, solver: str, nfe: int) -> np.ndarray:
        solver = normalize_solver_key(solver)
        if solver not in self.solvers:
            raise ValueError("Policy was not trained for this solver.")
        if self.unconditional and np.any(np.asarray(context) != 0):
            raise ValueError("Unconditional policies reject labels/nonzero context.")
        return np.concatenate(
            (
                self.context.transform_one(context),
                np.array([float(s == solver) for s in self.solvers]),
                self.settings.transform_one(self.budget(solver, nfe)),
            )
        ).astype(np.float32)

    def to_payload(self):
        return {
            "context": self.context.to_payload(),
            "settings": self.settings.to_payload(),
            "solvers": list(self.solvers),
            "unconditional": self.unconditional,
        }

    @classmethod
    def from_payload(cls, p):
        return cls(
            EmbeddingNormalizer.from_payload(p["context"]),
            EmbeddingNormalizer.from_payload(p["settings"]),
            tuple(p["solvers"]),
            p["unconditional"],
        )
