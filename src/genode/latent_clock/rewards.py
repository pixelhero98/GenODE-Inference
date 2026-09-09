from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from genode.latent_clock.artifacts import canonical_sha256, read_jsonl, sha256_file, write_new_json, write_new_jsonl


def _weight_module(value: Any) -> Any:
    if callable(getattr(value, "state_dict", None)) and value.state_dict():
        return value
    for name in ("model", "rm", "scorer"):
        child = getattr(value, name, None)
        if child is not None and child is not value:
            return _weight_module(child)
    raise TypeError(f"Cannot locate non-empty model weights for {type(value).__name__}.")


def _module_fingerprint(value: Any) -> str:
    value = _weight_module(value)
    digest = hashlib.sha256()
    state = value.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        byte_view = tensor.reshape(-1).view(__import__("torch").uint8)
        chunk_bytes = 64 * 1024 * 1024
        for offset in range(0, byte_view.numel(), chunk_bytes):
            chunk = byte_view[offset : offset + chunk_bytes].cpu().contiguous().numpy()
            digest.update(memoryview(chunk))
    return digest.hexdigest()


class FrozenDualScorer:
    def __init__(self, *, device: str = "cuda", asset_manifest: str | None = None) -> None:
        import ImageReward as image_reward
        import t2v_metrics

        manifest_path = asset_manifest or os.environ.get("LATENT_CLOCK_ASSETS")
        if not manifest_path:
            raise ValueError("Scoring requires the frozen asset manifest through LATENT_CLOCK_ASSETS.")
        assets = json.loads(Path(manifest_path).read_text())
        self.image_reward = image_reward.load(
            "ImageReward-v1.0", device=device, download_root=assets["image_reward"]["snapshot"]
        )
        self.vqa_score = t2v_metrics.VQAScore(
            model="clip-flant5-xxl", device=device, cache_dir=str(Path(os.environ["HF_HOME"]) / "hub")
        )
        for model in (self.image_reward, self.vqa_score):
            _weight_module(model).eval().requires_grad_(False)
        self._fingerprints = (_module_fingerprint(self.image_reward), _module_fingerprint(self.vqa_score))
        self.versions = {
            "ImageReward": importlib.metadata.version("image-reward"),
            "t2v_metrics": importlib.metadata.version("t2v-metrics"),
            "vqa_model": "clip-flant5-xxl",
        }
        self.asset_manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()

    def verify_frozen(self) -> None:
        current = (_module_fingerprint(self.image_reward), _module_fingerprint(self.vqa_score))
        if current != self._fingerprints:
            raise RuntimeError("Reward-model weights changed during scoring.")

    def score(self, prompt: str, image_path: str | Path) -> tuple[float, float]:
        import torch

        path = str(Path(image_path).resolve(strict=True))
        with torch.inference_mode():
            preference = self.image_reward.score(str(prompt), path)
            alignment = self.vqa_score(images=[path], texts=[str(prompt)])
        preference_value = float(np.asarray(preference).reshape(-1)[0])
        alignment_value = float(alignment.detach().cpu().reshape(-1)[0].item())
        if not math.isfinite(preference_value) or not math.isfinite(alignment_value):
            raise ValueError("Reward model returned a non-finite score.")
        return preference_value, alignment_value


def score_image_manifest(source: str | Path, destination: str | Path, *, device: str = "cuda") -> None:
    rows = read_jsonl(source)
    if not rows:
        raise ValueError("Image manifest is empty.")
    manifest_path = os.environ["LATENT_CLOCK_ASSETS"]
    records = Path(str(destination) + ".records")
    records.mkdir(parents=True, exist_ok=True)
    identity = {"source_sha256": sha256_file(source), "asset_manifest_sha256": sha256_file(manifest_path)}
    identity_path = records / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Scoring resume directory belongs to different images or assets.")
    else:
        write_new_json(identity_path, identity)
    if Path(destination).exists() and Path(str(destination) + ".metadata.json").exists():
        return
    scorer = FrozenDualScorer(device=device)
    output = []
    for index, row in enumerate(rows):
        missing = {"prompt", "image_path"} - row.keys()
        if missing:
            raise ValueError(f"Image manifest row is missing {sorted(missing)}.")
        if sha256_file(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Image changed before scoring.")
        record_path = records / (canonical_sha256(row) + ".json")
        if record_path.exists():
            output.append(json.loads(record_path.read_text()))
            continue
        import torch

        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            preference, alignment = scorer.score(str(row["prompt"]), str(row["image_path"]))
        except BaseException as exc:
            with (records / "failed-attempts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "request_id": row["request_id"],
                            "error": repr(exc),
                            "wall_seconds": time.perf_counter() - started,
                        }
                    )
                    + "\n"
                )
            raise
        torch.cuda.synchronize()
        output.append(
            {
                **row,
                "preference": preference,
                "alignment": alignment,
                "image_reward_calls": 1,
                "vqa_score_calls": 1,
                "scorer_gpu_seconds": time.perf_counter() - started,
            }
        )
        write_new_json(record_path, output[-1])
        if (index + 1) % 25 == 0:
            print(json.dumps({"scored": index + 1, "total": len(rows)}), flush=True)
    scorer.verify_frozen()
    if not Path(destination).exists():
        write_new_jsonl(destination, output)

    write_new_json(
        str(destination) + ".metadata.json",
        {
            "scorer_versions": scorer.versions,
            "asset_manifest_sha256": scorer.asset_manifest_sha256,
            "weight_fingerprints": list(scorer._fingerprints),
            "rows": len(output),
        },
    )
