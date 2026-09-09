"""Native text-to-image measurement preparation for common GICO training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from genode.gico.clocks import REFERENCE_KEYS, verify_measurement_clock
from genode.gico.policy import save_context_embedding_table
from genode.gico.train_gico import load_config, run_config
from genode.latent_clock.artifacts import canonical_sha256, read_jsonl, write_new_json, write_new_jsonl
from genode.latent_clock.protocol import BUDGETS, NOISE_SEEDS, PILOT_CLOCKS, PILOT_NFES


def runtime_binding(runtime) -> dict:
    return {
        "task": runtime.metadata["backbone"],
        "backbone_revision": runtime.adapter.backbone_revision,
        "solver": runtime.adapter.solver_key,
        "runtime": runtime.metadata,
        "weights": runtime.fingerprints,
        "context_source": "pooled_native_text",
    }


def checkpoint_identity(path: str | None, method: str, *, student_kind: str = "deterministic") -> str | None:
    if path is None:
        return None
    if method == "gico":
        from genode.gico.policy import load_policy

        return load_policy(path, student_kind=student_kind).artifact_sha256
    from genode.latent_clock.artifacts import sha256_file

    return sha256_file(path)


def prepare_gico_rows(
    raw: list[dict], *, manifest: dict, task: str, nfes: tuple[int, ...], budget: str = "100"
) -> tuple[list[dict], list[dict]]:
    """Convert complete collected support and pilot panels without refitting metrics.

    COCO calibration prompts become training examples; separate validation
    prompts remain validation. Pilot prompts provide component scales only.
    """
    if task not in {"sana", "sd15"} or not nfes or any(type(nfe) is not int or nfe < 1 for nfe in nfes):
        raise ValueError("Latent GICO requires a retained task and positive target NFEs.")
    prompts = {row["prompt_id"]: row for row in manifest["records"]}
    if len(prompts) != len(manifest["records"]):
        raise ValueError("Prompt manifest has duplicate prompt identities.")
    phase_ids = {
        phase: [row["prompt_id"] for row in manifest["records"] if row["split"] == phase]
        for phase in ("pilot", "calibration", "validation")
    }
    phase_ids["calibration"] = phase_ids["calibration"][: BUDGETS[budget]["prompts"]]
    if not all(phase_ids.values()):
        raise ValueError("Native GICO needs separate pilot, training/calibration, and validation prompt panels.")
    source_phases = {}
    for prompt_id in sum(phase_ids.values(), []):
        prompt = prompts[prompt_id]
        source = (prompt["image_id"], " ".join(prompt["prompt"].split()).casefold())
        for identity in (("image", source[0]), ("caption", source[1])):
            if identity in source_phases and source_phases[identity] != prompt["split"]:
                raise ValueError("Prompt splits reuse an image or normalized caption.")
            source_phases[identity] = prompt["split"]
    selected = []
    for source in raw:
        phase = source["split"]
        if phase in {"locked_test", "test", "geneval"}:
            raise ValueError("Locked-test/audit rows cannot enter GICO fitting evidence.")
        if phase not in phase_ids or source["prompt_id"] not in phase_ids[phase]:
            continue
        if int(source["nfe"]) not in (PILOT_NFES if phase == "pilot" else nfes):
            continue
        keys = PILOT_CLOCKS if phase == "pilot" else REFERENCE_KEYS
        if source["clock_key"] not in keys:
            raise ValueError("Native teacher fitting requires reference-clock evidence only.")
        prompt = prompts[source["prompt_id"]]
        if any(source[field] != prompt[field] for field in ("prompt", "image_id", "caption_id", "split")):
            raise ValueError("Measured prompt does not match the frozen prompt manifest.")
        if source.get("completed") is not True or source.get("realized_nfe") != source["nfe"]:
            raise ValueError("Native evidence must record completed exact-NFE trajectories.")
        if source.get("density_mass") is None:
            raise ValueError("Historical direct-grid measurements lack 64-bin density provenance; recollect evidence.")
        binding = source["backbone_binding"]
        if (
            binding["task"] != task
            or binding["backbone_revision"] != source["checkpoint_id"]
            or binding["solver"] != source["solver_key"]
        ):
            raise ValueError("Measured latent backbone binding is inconsistent.")
        row = {
            "task": task,
            "backbone": source["checkpoint_id"],
            "solver": source["solver_key"],
            "nfe": int(source["nfe"]),
            "context_id": source["prompt_id"],
            "split": {"pilot": "calibration", "calibration": "train", "validation": "validation"}[phase],
            "seed": int(source["noise_seed"]),
            "ensemble_size": 1,
            "reference_id": canonical_sha256(
                {key: prompt[key] for key in ("prompt_id", "image_id", "caption_id", "prompt")}
            ),
            "measurement_protocol": source["measurement_protocol"],
            "schedule_key": source["clock_key"],
            "metrics": {"preference": float(source["preference"]), "alignment": float(source["alignment"])},
            "density_mass": list(source["density_mass"]),
            "time_grid": list(source["nodes"]),
            "backbone_binding": binding,
            "context_embedding_sha256": source["context_embedding_sha256"],
            "context_protocol": "pooled_native_text_prompt_holdout_v1",
        }
        if not np.isfinite(list(row["metrics"].values())).all():
            raise ValueError("Native preference/alignment scores must be finite.")
        verify_measurement_clock(row)
        selected.append(row)
    expected = {
        (prompt, nfe, key, seed)
        for phase, ids in phase_ids.items()
        for prompt in ids
        for nfe in (PILOT_NFES if phase == "pilot" else nfes)
        for key in (PILOT_CLOCKS if phase == "pilot" else REFERENCE_KEYS)
        for seed in NOISE_SEEDS
    }
    observed = {(row["context_id"], row["nfe"], row["schedule_key"], row["seed"]) for row in selected}
    if observed != expected or len(observed) != len(selected):
        raise ValueError("Native paired reference/pilot evidence is incomplete or duplicated.")
    return [row for row in selected if row["split"] != "calibration"], [
        row for row in selected if row["split"] == "calibration"
    ]


def prepare_gico(
    *,
    rows_paths: list[str],
    embeddings_paths: list[str],
    manifest_path: str,
    task: str,
    nfes: tuple[int, ...],
    budget: str,
    output: str,
) -> dict:
    raw = []
    for path in rows_paths:
        scoring = json.loads(Path(str(path) + ".metadata.json").read_text(encoding="utf-8"))
        score_identity = canonical_sha256(
            {key: scoring[key] for key in ("scorer_versions", "asset_manifest_sha256", "weight_fingerprints")}
        )
        for source in read_jsonl(path):
            raw.append(
                {**source, "measurement_protocol": canonical_sha256([source["measurement_protocol"], score_identity])}
            )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows, calibration = prepare_gico_rows(raw, manifest=manifest, task=task, nfes=nfes, budget=budget)
    contexts = {}
    for path in embeddings_paths:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError("Native pooled text embeddings must be finite vectors.")
                if key in contexts and not np.array_equal(contexts[key], value):
                    raise ValueError("Native context changed between collection phases.")
                contexts[key] = value.copy()
    required = {row["context_id"] for row in rows + calibration}
    if required - contexts.keys():
        raise ValueError("Native GICO evidence is missing pooled text embeddings.")
    if any(
        canonical_sha256(contexts[row["context_id"]].tolist()) != row["context_embedding_sha256"]
        for row in rows + calibration
    ):
        raise ValueError("Pooled text context changed since native measurement collection.")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_jsonl(destination / "rows.jsonl", rows)
    write_new_jsonl(destination / "calibration_rows.jsonl", calibration)
    save_context_embedding_table(
        destination / "contexts.npz",
        {key: contexts[key] for key in required},
        metadata={"context_source": "pooled_native_text", "task": task},
    )
    config = {
        "rows": "rows.jsonl",
        "contexts": "contexts.npz",
        "calibration_rows": "calibration_rows.jsonl",
        "output": "policy",
        "student_kind": "both",
        "teacher_score_weight": 0.01,
        "steps": 2000,
        "seed": 0,
        "device": "cuda",
        "purpose": "research",
    }
    write_new_json(destination / "train_config.json", config)
    return {"rows": len(rows), "pilot_rows": len(calibration), "contexts": len(required), "task": task}


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
