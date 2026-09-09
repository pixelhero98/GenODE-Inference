from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FrozenContext:
    context_id: str
    embedding: np.ndarray
    backbone_revision: str

    def __post_init__(self) -> None:
        value = np.asarray(self.embedding)
        if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("Frozen context embedding must be a non-empty finite vector.")
        value = np.asarray(value, dtype=np.float32).copy()
        value.setflags(write=False)
        object.__setattr__(self, "embedding", value)
        if not self.context_id or not self.backbone_revision:
            raise ValueError("Frozen context requires non-empty context and backbone identities.")


@dataclass(frozen=True)
class ExecutionTrace:
    method_key: str
    clock_key: str
    solver_key: str
    requested_nfe: int
    realized_nfe: int
    backbone_forwards: int
    cfg_sample_equivalents: int
    elapsed_gpu_seconds: float

    def __post_init__(self) -> None:
        if self.requested_nfe <= 0 or self.realized_nfe != self.requested_nfe:
            raise ValueError("Execution trace must report an exact positive NFE.")
        if self.backbone_forwards <= 0 or self.cfg_sample_equivalents < self.backbone_forwards:
            raise ValueError("Execution trace contains invalid forward counts.")
        if not math.isfinite(self.elapsed_gpu_seconds) or self.elapsed_gpu_seconds < 0:
            raise ValueError("Execution trace contains invalid elapsed time.")


@dataclass
class BudgetLedger:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        trace: ExecutionTrace,
        *,
        prompt_id: str,
        noise_seed: int,
        completed: bool,
        image_reward_calls: int = 0,
        vqa_score_calls: int = 0,
        audit_calls: int = 0,
        physical_reuse_key: str | None = None,
        error: str | None = None,
    ) -> None:
        if completed == bool(error):
            raise ValueError("A ledger record must be either completed or failed with an error.")
        payload = asdict(trace)
        payload.update(
            prompt_id=str(prompt_id),
            noise_seed=int(noise_seed),
            completed=bool(completed),
            image_reward_calls=int(image_reward_calls),
            vqa_score_calls=int(vqa_score_calls),
            audit_calls=int(audit_calls),
            physical_reuse_key=physical_reuse_key,
            error=error,
            recorded_unix_seconds=time.time(),
        )
        self.records.append(payload)

    def summary(self) -> dict[str, Any]:
        completed = [row for row in self.records if row["completed"]]
        unique = {row["physical_reuse_key"] or f"row:{index}": row for index, row in enumerate(completed)}
        return {
            "logical_trajectories": len(completed),
            "failed_attempts": len(self.records) - len(completed),
            "physical_trajectories": len(unique),
            "realized_nfe": sum(row["realized_nfe"] for row in completed),
            "backbone_forwards": sum(row["backbone_forwards"] for row in completed),
            "cfg_sample_equivalents": sum(row["cfg_sample_equivalents"] for row in completed),
            "image_reward_calls": sum(row["image_reward_calls"] for row in completed),
            "vqa_score_calls": sum(row["vqa_score_calls"] for row in completed),
            "audit_calls": sum(row["audit_calls"] for row in completed),
            "gpu_seconds": sum(row["elapsed_gpu_seconds"] for row in completed),
            "physical_realized_nfe": sum(row["realized_nfe"] for row in unique.values()),
            "physical_backbone_forwards": sum(row["backbone_forwards"] for row in unique.values()),
            "physical_cfg_sample_equivalents": sum(row["cfg_sample_equivalents"] for row in unique.values()),
            "physical_gpu_seconds": sum(row["elapsed_gpu_seconds"] for row in unique.values()),
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"records": self.records, "summary": self.summary()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@runtime_checkable
class FrozenGeneratorAdapter(Protocol):
    backbone_revision: str
    solver_key: str

    def encode_context(self, prompt_id: str, prompt: str) -> FrozenContext: ...

    def sample(self, *, noise_seed: int, context: FrozenContext, clock: Any) -> tuple[Any, ExecutionTrace]: ...


@runtime_checkable
class ComponentScorer(Protocol):
    def score(self, prompt: str, image: Any) -> tuple[float, float]: ...
