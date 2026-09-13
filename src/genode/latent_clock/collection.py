from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from genode.latent_clock.artifacts import canonical_sha256, read_jsonl, sha256_file, write_new_json, write_new_jsonl
from genode.latent_clock.clocks import Clock, reference_clocks
from genode.latent_clock.gico import checkpoint_identity, runtime_binding
from genode.latent_clock.protocol import BUDGETS, NOISE_SEEDS, PILOT_CLOCKS, PILOT_NFES
from genode.latent_clock.runtime import load_runtime

GENEVAL_SEEDS = (17011, 94153, 23003, 79999)


def prepare_geneval(
    *,
    source: str,
    output: str,
    method: str,
    nfe: int,
    checkpoint: str | None,
    freeze_path: str,
    student_kind: str = "deterministic",
    clock_seed: int = 0,
) -> None:
    frozen = json.loads(Path(freeze_path).read_text())
    identity = checkpoint_identity(checkpoint, method, student_kind=student_kind) if checkpoint else method
    if identity not in frozen["method_identities"]:
        raise ValueError("GenEval method was not frozen before audit generation.")
    prompts = []
    with Path(source).open() as stream:
        for index, line in enumerate(stream):
            metadata = json.loads(line)
            prompts.append(
                {
                    "prompt_id": f"geneval-{index:05d}",
                    "prompt": metadata["prompt"],
                    "metadata": metadata,
                    "split": "geneval",
                }
            )
    if len(prompts) != 553:
        raise ValueError("The official GenEval suite must contain all 553 prompts.")
    requests = []
    fixed_clock = None
    if method == "uniform":
        fixed_clock = reference_clocks(nfe)[0]
    elif method == "fixed":
        key = json.loads(Path(checkpoint).read_text())[str(nfe)]["clock_key"]
        fixed_clock = next(clock for clock in reference_clocks(nfe) if clock.key == key)
    elif method == "bo":
        selected = json.loads(Path(checkpoint).read_text())
        fixed_clock = Clock("bo", nfe, tuple(selected["nodes"]), "botorch_qlognei")
    for prompt in prompts:
        for noise_seed in GENEVAL_SEEDS:
            request = {
                **prompt,
                "nfe": int(nfe),
                "noise_seed": noise_seed,
                "method": method,
                "clock": asdict(fixed_clock) if fixed_clock else None,
            }
            request["request_id"] = canonical_sha256(request)
            requests.append(request)
    write_new_json(
        output,
        {
            "phase": "geneval",
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_identity(checkpoint, method, student_kind=student_kind),
            "student_kind": student_kind,
            "clock_seed": clock_seed,
            "freeze_sha256": sha256_file(freeze_path),
            "geneval_source_sha256": sha256_file(source),
            "requests": requests,
        },
    )


def export_geneval(*, images_path: str, output: str) -> None:
    rows = read_jsonl(images_path)
    groups = {}
    for row in rows:
        if row["split"] != "geneval":
            raise ValueError("GenEval export contains a non-GenEval trajectory.")
        groups.setdefault(row["prompt_id"], []).append(row)
    if len(groups) != 553 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("GenEval export requires four images for every official prompt.")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    import shutil

    for prompt_id, rows in sorted(groups.items()):
        index = int(prompt_id.rsplit("-", 1)[1])
        folder = root / f"{index:05d}"
        samples = folder / "samples"
        samples.mkdir(parents=True)
        (folder / "metadata.jsonl").write_text(json.dumps(rows[0]["metadata"]) + "\n")
        for sample_index, row in enumerate(sorted(rows, key=lambda r: int(r["noise_seed"]))):
            shutil.copyfile(row["image_path"], samples / f"{sample_index:04d}.png")


def prepare_collection(
    *,
    manifest_path: str,
    output: str,
    phase: str,
    nfes: list[int],
    budget: str = "100",
    method: str = "support",
    checkpoint: str | None = None,
    freeze_path: str | None = None,
    student_kind: str = "deterministic",
    clock_seed: int = 0,
) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    if phase not in ("pilot", "calibration", "validation", "locked_test"):
        raise ValueError("Invalid collection phase.")
    if phase == "pilot" and (tuple(nfes) != PILOT_NFES or method != "support"):
        raise ValueError("The pilot is exactly the three preregistered clocks at NFE 4/8.")
    if phase == "locked_test":
        if not freeze_path:
            raise ValueError("Locked evaluation requires the method-freeze manifest.")
        frozen = json.loads(Path(freeze_path).read_text())
        if frozen["prompt_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError("Method freeze and prompt manifest disagree.")
        identity = checkpoint_identity(checkpoint, method, student_kind=student_kind) if checkpoint else method
        if identity not in frozen["method_identities"]:
            raise ValueError("Requested method was not included in the method freeze.")
    prompts = [row for row in manifest["records"] if row["split"] == phase]
    if phase == "calibration":
        prompts = prompts[: BUDGETS[budget]["prompts"]]
    requests = []
    for prompt in prompts:
        for nfe in nfes:
            if method == "support":
                clocks = [c for c in reference_clocks(nfe) if phase != "pilot" or c.key in PILOT_CLOCKS]
            elif method in ("native", "gico", "pg", "ld3"):
                clocks = [None]
            elif method == "uniform":
                clocks = [reference_clocks(nfe)[0]]
            elif method == "fixed":
                fixed = json.loads(Path(checkpoint).read_text())
                clocks = [c for c in reference_clocks(nfe) if c.key == fixed[str(nfe)]["clock_key"]]
            elif method == "bo":
                selected = json.loads(Path(checkpoint).read_text())
                clocks = [Clock("bo", nfe, tuple(selected["nodes"]), "botorch_qlognei")]
            else:
                raise ValueError(f"Unsupported collection method {method!r}.")
            for clock in clocks:
                for noise_seed in NOISE_SEEDS:
                    request = {
                        **prompt,
                        "nfe": nfe,
                        "noise_seed": noise_seed,
                        "method": method,
                        "clock": None if clock is None else asdict(clock),
                    }
                    request["request_id"] = canonical_sha256(request)
                    requests.append(request)
    write_new_json(
        output,
        {
            "manifest_sha256": sha256_file(manifest_path),
            "phase": phase,
            "budget": budget,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_identity(checkpoint, method, student_kind=student_kind),
            "student_kind": student_kind,
            "clock_seed": clock_seed,
            "freeze_sha256": sha256_file(freeze_path) if freeze_path else None,
            "requests": requests,
        },
    )


def image_tensor_to_pil(value: Any) -> Any:
    import torch
    from PIL import Image

    if not isinstance(value, torch.Tensor) or value.shape != (1, 3, 512, 512):
        raise ValueError(f"Expected a decoded [1,3,512,512] image, got {getattr(value, 'shape', None)}.")
    array = ((value.detach().float().cpu()[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def collect(*, runtime_config: str, plan_path: str, output: str) -> None:
    import torch

    plan = json.loads(Path(plan_path).read_text())
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "collection_protocol": "shared_density64_complete_clock_v2",
        "plan_sha256": sha256_file(plan_path),
        "runtime_config_sha256": sha256_file(runtime_config),
    }
    identity_path = destination / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Resume directory belongs to a different plan or runtime.")
    else:
        write_new_json(identity_path, identity)
    if (
        plan["checkpoint"]
        and checkpoint_identity(plan["checkpoint"], plan["method"], student_kind=plan["student_kind"])
        != plan["checkpoint_sha256"]
    ):
        raise ValueError("Method checkpoint changed after collection planning.")
    runtime = load_runtime(runtime_config)
    model = None
    if plan["method"] == "gico":
        from genode.gico.policy import load_policy

        model = load_policy(
            plan["checkpoint"], student_kind=plan["student_kind"], expected_backbone=runtime.adapter.backbone_revision
        )
    elif plan["method"] == "pg":
        from genode.latent_clock.pg import ClockPolicy

        payload = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
        model = ClockPolicy(payload["context_dim"], payload["nfe"])
        model.load_state_dict(payload["state_dict"])
        model.eval().requires_grad_(False)
    elif plan["method"] == "ld3":
        from genode.latent_clock.ld3 import LD3Schedule

        if runtime.adapter.solver_key != "ipndm":
            raise ValueError("The official LD3 baseline requires explicit runtime solver='ipndm', not 'ipndm_v'.")
        model = LD3Schedule(**json.loads(Path(plan["checkpoint"]).read_text()))
        if (
            runtime.metadata["backbone"] != "sd15"
            or model.provenance["backbone_revision"] != runtime.adapter.backbone_revision
        ):
            raise ValueError("LD3 schedule does not match the current frozen SD1.5 checkpoint.")
    if plan["method"] == "gico" and (
        model.metadata["task"] != runtime.metadata["backbone"]
        or runtime.adapter.solver_key not in model.metadata["solvers"]
        or model.metadata["backbone_binding"] != runtime_binding(runtime)
    ):
        raise ValueError("GICO checkpoint and native generator/context/solver configuration differ.")
    images, embeddings, contexts = [], {}, {}
    for index, request in enumerate(plan["requests"]):
        prompt_id, request_id = request["prompt_id"], request["request_id"]
        record_path = destination / f"{request_id}.json"
        if prompt_id not in contexts:
            contexts[prompt_id] = runtime.adapter.encode_context(prompt_id, request["prompt"])
            embeddings[prompt_id] = contexts[prompt_id].embedding
        context = contexts[prompt_id]
        if record_path.exists():
            row = json.loads(record_path.read_text())
            if row["request_id"] != request_id or sha256_file(row["image_path"]) != row["image_sha256"]:
                raise ValueError("Completed trajectory record or image is corrupt.")
            images.append(row)
            continue
        if request["clock"]:
            clock = Clock(**request["clock"])
        elif request["method"] == "native":
            clock = runtime.native_clock(request["nfe"])
        elif request["method"] == "gico":
            from genode.gico.clocks import materialize

            mass = model.density(
                context.embedding,
                runtime.adapter.solver_key,
                request["nfe"],
                seed=plan["clock_seed"],
                request_id=request_id,
            )
            clock = Clock(
                f"gico_{plan['student_kind']}",
                request["nfe"],
                materialize(mass, runtime.adapter.solver_key, request["nfe"]),
                "unified_gico",
                tuple(float(value) for value in mass),
            )
        elif request["method"] == "pg":
            from genode.gico.conditioning import EmbeddingNormalizer
            from genode.latent_clock.pg import sample_policy_clock

            sample_seed = int(canonical_sha256([prompt_id, request["noise_seed"], payload["optimizer_seed"]])[:8], 16)
            embedding = EmbeddingNormalizer.from_payload(payload["embedding_normalizer"]).transform_one(
                context.embedding
            )
            clock = sample_policy_clock(model, embedding, seed=sample_seed)
        elif request["method"] == "ld3":
            if request["nfe"] != model.nfe:
                raise ValueError("LD3 schedule NFE differs from the requested evaluation NFE.")
            times = np.asarray(model.integration_times)
            clock = Clock("ld3", model.nfe, tuple((times[0] - times) / (times[0] - times[-1])), "ld3_dual_times")
        else:
            raise ValueError("Collection request does not materialize a clock.")
        if clock.target_nfe != request["nfe"]:
            raise ValueError("Materialized clock NFE differs from the requested comparison NFE.")
        started = time.perf_counter()
        try:
            if request["method"] == "ld3":
                value, trace = runtime.adapter.sample_times(
                    noise_seed=request["noise_seed"],
                    context=context,
                    integration_times=model.integration_times,
                    evaluation_times=model.evaluation_times,
                    clock_key="ld3",
                )
            else:
                value, trace = runtime.adapter.sample(noise_seed=request["noise_seed"], context=context, clock=clock)
            trace = replace(trace, method_key=plan["method"])
            image_path = destination / f"{request_id}.png"
            image_tensor_to_pil(value).save(image_path)
            row = {
                **request,
                "clock_key": clock.key,
                "nodes": list(clock.nodes),
                "density_mass": None if clock.density_mass is None else list(clock.density_mass),
                "clock_student_kind": plan["student_kind"] if plan["method"] == "gico" else None,
                "clock_seed": plan["clock_seed"] if plan["method"] == "gico" else None,
                "backbone_binding": runtime_binding(runtime),
                "context_embedding_sha256": canonical_sha256(context.embedding.tolist()),
                "measurement_protocol": canonical_sha256(
                    {
                        "clock_protocol": "shared_density64_v1",
                        "runtime": runtime_binding(runtime),
                        "postprocess": "decoded_clamp_round_uint8_v1",
                    }
                ),
                "completed": True,
                "image_path": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "trace": asdict(trace),
                "solver_key": runtime.adapter.solver_key,
                "checkpoint_id": runtime.adapter.backbone_revision,
                "realized_nfe": trace.realized_nfe,
                "runtime_nfe": clock.target_nfe,
                "macro_steps": clock.target_nfe,
                "solver_order": runtime.metadata.get("order", 1),
                "transferred_schedule": clock.key.startswith(("ays", "gits", "ots")),
            }
            if request["method"] == "ld3":
                row["integration_times"] = list(model.integration_times)
                row["evaluation_times"] = list(model.evaluation_times)
            if request["method"] == "pg":
                from genode.latent_clock.clocks import PG_CLOCK_PRECISION

                row["clock_precision"] = PG_CLOCK_PRECISION
                row["clock_precision_corrected"] = clock.source_kind == "pg_precision_corrected"
            write_new_json(record_path, row)
            images.append(row)
        except BaseException as exc:
            with (destination / "failed-attempts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "error": repr(exc),
                            "wall_seconds": time.perf_counter() - started,
                            "unix_time": time.time(),
                        }
                    )
                    + "\n"
                )
            raise
        if (index + 1) % 25 == 0:
            print(json.dumps({"completed": index + 1, "total": len(plan["requests"])}), flush=True)
    runtime.verify_frozen()
    np.savez(destination / "contexts.npz", **embeddings)
    write_new_jsonl(destination / "images.jsonl", images)
    write_new_json(
        destination / "complete.json",
        {**identity, "trajectories": len(images), "weights": runtime.fingerprints, "runtime": runtime.metadata},
    )


def sampler_parity(*, runtime_config: str, output: str) -> None:
    import torch

    runtime = load_runtime(runtime_config)
    context = runtime.adapter.encode_context("parity-000", "A red ceramic mug on a wooden table.")
    results = []
    for nfe in (4, 6, 8):
        native_clock = runtime.native_clock(nfe)
        native = runtime.native_sample(context, NOISE_SEEDS[0], nfe)
        repeated = runtime.native_sample(context, NOISE_SEEDS[0], nfe)
        repeatability = float((native - repeated).abs().max().item())
        custom, trace = runtime.adapter.sample(noise_seed=NOISE_SEEDS[0], context=context, clock=native_clock)
        error = float((native - custom).abs().max().item())
        tolerance = repeatability
        if error > tolerance:
            write_new_json(
                output,
                {
                    "status": "failed",
                    "runtime": runtime.metadata,
                    "nfe": nfe,
                    "clock_nodes": list(native_clock.nodes),
                    "native_repeatability_max_abs": repeatability,
                    "replay_max_abs": error,
                    "tolerance": tolerance,
                },
            )
            raise RuntimeError(f"Native parity failed at NFE {nfe}: {error} > {tolerance}.")
        for clock in reference_clocks(nfe):
            value, clock_trace = runtime.adapter.sample(noise_seed=NOISE_SEEDS[0], context=context, clock=clock)
            if not torch.isfinite(value).all() or clock_trace.realized_nfe != nfe:
                raise RuntimeError(f"Invalid runtime output for {clock.key} at NFE {nfe}.")
        results.append(
            {
                "nfe": nfe,
                "native_repeatability_max_abs": repeatability,
                "replay_max_abs": error,
                "tolerance": tolerance,
                "trace": asdict(trace),
                "reference_clocks_passed": 25,
                "audit_trajectories": 28,
            }
        )
    runtime.verify_frozen()
    write_new_json(output, {"runtime": runtime.metadata, "weights": runtime.fingerprints, "parity": results})
