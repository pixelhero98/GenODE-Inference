"""Native text-to-image measurement preparation for common GICO training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from genode.gico.clocks import verify_measurement_clock
from genode.gico.collection import validate_collection
from genode.gico.policy import load_context_embedding_table, save_context_embedding_table
from genode.gico.train_gico import load_config, read_rows, run_config
from genode.latent_clock.artifacts import canonical_sha256, write_new_json, write_new_jsonl


def runtime_binding(runtime) -> dict:
    return {
        "task": runtime.metadata["backbone"],
        "backbone_revision": runtime.adapter.backbone_revision,
        "solver": runtime.adapter.solver_key,
        "runtime": runtime.metadata,
        "weights": runtime.fingerprints,
        "context_source": "pooled_native_text",
    }


def checkpoint_identity(path: str | None, method: str, *, student_kind: str = "GICO-det-policy") -> str | None:
    if path is None:
        return None
    if method == "gico":
        from genode.gico.policy import load_policy

        return load_policy(path, student_kind=student_kind).artifact_sha256
    from genode.latent_clock.artifacts import sha256_file

    return sha256_file(path)


def prepare_gico_rows(raw, *, manifest, task, nfes):
    """Validate complete collected prompt evidence without changing its split."""
    if task not in {"sana", "sd15"}:
        raise ValueError("Latent GICO requires a retained text-to-image task.")
    validate_collection(manifest, raw)
    if manifest["task"] != task or {r["nfe"] for r in raw} != set(nfes):
        raise ValueError("Collected task/NFE scope differs from preparation settings.")
    for row in raw:
        verify_measurement_clock(row)
    return raw


def prepare_gico(
    *,
    rows_paths: list[str],
    embeddings_paths: list[str],
    manifest_path: str,
    task: str,
    nfes: tuple[int, ...],
    output: str,
) -> dict:
    raw = [row for path in rows_paths for row in read_rows(path)]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if "collection_manifest" in manifest:
        manifest = manifest["collection_manifest"]
    rows = prepare_gico_rows(raw, manifest=manifest, task=task, nfes=nfes)
    contexts = {}
    for path in embeddings_paths:
        for key, value in load_context_embedding_table(path).items():
            if key in contexts and not np.array_equal(contexts[key], value):
                raise ValueError("Native context changed between collection phases.")
            contexts[key] = value.copy()
    required = {row["context_id"] for row in rows}
    if required - contexts.keys():
        raise ValueError("Native GICO evidence is missing pooled text embeddings.")
    if any(canonical_sha256(contexts[row["context_id"]].tolist()) != row["context_embedding_sha256"] for row in rows):
        raise ValueError("Pooled text context changed since native measurement collection.")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_jsonl(destination / "rows.jsonl", rows)
    write_new_json(destination / "collection.json", manifest)
    save_context_embedding_table(
        destination / "contexts.npz",
        {key: contexts[key] for key in required},
        metadata={"context_source": "pooled_native_text", "task": task},
    )
    config = {
        "rows": "rows.jsonl",
        "contexts": "contexts.npz",
        "collection_manifest": "collection.json",
        "output": "policy",
        "student_kind": "GICO-det-policy",
        "seed": 0,
        "device": "cuda",
        "purpose": "research",
    }
    write_new_json(destination / "train_config.json", config)
    return {"rows": len(rows), "complete_solves": manifest["complete_solves"], "contexts": len(required), "task": task}


def fit_gico(
    *,
    config_path: str,
    student_kind: str | None = None,
    teacher_score_weight: float | None = None,
    dry_run: bool = False,
) -> dict:
    config = load_config(config_path)
    if student_kind is not None:
        config["student_kind"] = student_kind
    if teacher_score_weight is not None:
        config["teacher_score_weight"] = teacher_score_weight
    return run_config(config, dry_run=dry_run)
