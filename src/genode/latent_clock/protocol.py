from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

BUDGETS = {
    "25": {"prompts": 32, "seeds": 2, "trajectories": 1600},
    "50": {"prompts": 64, "seeds": 2, "trajectories": 3200},
    "100": {"prompts": 128, "seeds": 2, "trajectories": 6400},
}
SPLIT_SIZES = {"pilot": 32, "calibration": 128, "validation": 128, "locked_test": 512}
NOISE_SEEDS = (17011, 94153)
OPTIMIZER_SEEDS = (3109, 6833, 104729)
PILOT_CLOCKS = ("uniform", "late_p_3", "late_p_3_reversed")
PILOT_NFES = (4, 8)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    image_id: int
    caption_id: int
    prompt: str
    split: str


def _normalized_caption(value: str) -> str:
    return _SPACE.sub(" ", str(value).strip()).casefold()


def _hash_order(seed: int, image_id: int, caption_id: int) -> str:
    return hashlib.sha256(f"latent-clock-coco-v1|{seed}|{image_id}|{caption_id}".encode()).hexdigest()


def build_prompt_splits(annotation_json: str | Path, *, seed: int = 48271) -> dict[str, Any]:
    source = Path(annotation_json)
    payload = json.loads(source.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("COCO annotation JSON must contain an annotations list.")
    one_per_image: dict[int, tuple[int, str]] = {}
    for row in annotations:
        image_id, caption_id, caption = int(row["image_id"]), int(row["id"]), str(row["caption"]).strip()
        if not caption:
            continue
        previous = one_per_image.get(image_id)
        if previous is None or caption_id < previous[0]:
            one_per_image[image_id] = (caption_id, caption)
    ordered = sorted(
        ((image_id, caption_id, caption) for image_id, (caption_id, caption) in one_per_image.items()),
        key=lambda row: _hash_order(seed, row[0], row[1]),
    )
    unique: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for row in ordered:
        normalized = _normalized_caption(row[2])
        if normalized not in seen:
            seen.add(normalized)
            unique.append(row)
    needed = sum(SPLIT_SIZES.values())
    if len(unique) < needed:
        raise ValueError(f"COCO source has {len(unique)} eligible prompts; {needed} are required.")
    records: list[PromptRecord] = []
    offset = 0
    for split, size in SPLIT_SIZES.items():
        for image_id, caption_id, caption in unique[offset : offset + size]:
            records.append(PromptRecord(f"coco-{image_id}-{caption_id}", image_id, caption_id, caption, split))
        offset += size
    body = {
        "protocol": "genode_latent_t2i_coco_split_v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": int(seed),
        "noise_seeds": list(NOISE_SEEDS),
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "records": [asdict(row) for row in records],
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


@dataclass(frozen=True)
class RewardScales:
    preference: float
    alignment: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 1e-12:
                raise ValueError(f"Degenerate {name} reward scale: {value!r}.")

    def utility(
        self, preference: float, alignment: float, uniform_preference: float, uniform_alignment: float
    ) -> dict[str, float]:
        u_preference = (float(preference) - float(uniform_preference)) / self.preference
        u_alignment = (float(alignment) - float(uniform_alignment)) / self.alignment
        values = {
            "u_preference": u_preference,
            "u_alignment": u_alignment,
            "utility": 0.5 * u_preference + 0.5 * u_alignment,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Non-finite balanced reward utility.")
        return values


def estimate_reward_scales(rows: Sequence[Mapping[str, Any]]) -> RewardScales:
    expected = {(str(row["prompt_id"]), int(row["nfe"]), str(row["clock_key"]), int(row["noise_seed"])) for row in rows}
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    required = {
        (prompt, nfe, clock, seed)
        for prompt in prompts
        for nfe in PILOT_NFES
        for clock in PILOT_CLOCKS
        for seed in NOISE_SEEDS
    }
    if len(prompts) != SPLIT_SIZES["pilot"] or expected != required or len(rows) != len(required):
        raise ValueError("Reward-scale rows must be the complete 32-prompt, 2-seed, 3-clock, NFE-4/8 pilot.")
    lookup = {(str(r["prompt_id"]), int(r["nfe"]), str(r["clock_key"]), int(r["noise_seed"])): r for r in rows}
    advantages = [[], []]
    for prompt in prompts:
        for nfe in PILOT_NFES:
            baseline = [
                np.mean([float(lookup[(prompt, nfe, "uniform", seed)][key]) for seed in NOISE_SEEDS])
                for key in ("preference", "alignment")
            ]
            for clock in PILOT_CLOCKS[1:]:
                for index, key in enumerate(("preference", "alignment")):
                    candidate = np.mean([float(lookup[(prompt, nfe, clock, seed)][key]) for seed in NOISE_SEEDS])
                    advantages[index].append(candidate - baseline[index])
    return RewardScales(float(np.std(advantages[0], ddof=1)), float(np.std(advantages[1], ddof=1)))


def validate_budget(budget: str | int, rows: Iterable[Mapping[str, Any]]) -> None:
    key = str(budget)
    if key not in BUDGETS:
        raise ValueError(f"Unknown budget {budget!r}; expected one of {tuple(BUDGETS)}.")
    completed = [row for row in rows if bool(row.get("completed", False))]
    if len(completed) != BUDGETS[key]["trajectories"]:
        raise ValueError(
            f"Budget {key} requires {BUDGETS[key]['trajectories']} completed trajectories, got {len(completed)}."
        )
