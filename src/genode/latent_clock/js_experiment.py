"""SANA/COCO adapter for the paper-based JS comparator.

Run ``python -m genode.latent_clock.js_experiment --help``. Paths and execution
resources live in experiment configuration, never in the installed package.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from genode.gico.rewards import RewardCalibration
from genode.latent_clock.artifacts import (
    canonical_sha256,
    read_jsonl,
    sha256_file,
    write_new_json,
)
from genode.latent_clock.collection import image_tensor_to_pil
from genode.latent_clock.gico import runtime_binding
from genode.latent_clock.js_reinforce import DirichletSchedulePolicy, JSArchitecture, materialize_js, sample_intervals
from genode.latent_clock.js_training import JSFit, fit_js, load_js, save_js, stack_inputs
from genode.latent_clock.runtime import load_runtime
from genode.latent_clock.search import ScoreWorker


def training_panel(rows, manifest, *, nfe, noise_seeds):
    anchors = [r for r in rows if r["split"] == "train" and r["schedule_key"] == "uniform" and r["nfe"] == nfe]
    lookup = {(r["context_id"], r["seed"]): r for r in anchors}
    prompt_ids = sorted({key[0] for key in lookup})
    if (
        not prompt_ids
        or len(lookup) != len(anchors)
        or set(lookup) != {(p, s) for p in prompt_ids for s in noise_seeds}
    ):
        raise ValueError("JS anchors must form a complete, unique training prompt/noise panel.")
    prompts = {r["prompt_id"]: r for r in manifest["records"]}
    if len(prompts) != len(manifest["records"]) or any(prompts[p]["split"] != "calibration" for p in prompt_ids):
        raise ValueError("JS training prompts must belong to the designated calibration/training split.")
    for row in anchors:
        prompt = prompts[row["context_id"]]
        if row["reference_id"] != canonical_sha256(
            {key: prompt[key] for key in ("prompt_id", "image_id", "caption_id", "prompt")}
        ):
            raise ValueError("JS anchor refers to a different prompt or reference image.")
        if row["task"] != "sana" or row["solver"] != "euler" or row["ensemble_size"] != 1:
            raise ValueError("JS SANA anchors require the single-image Euler protocol.")
        if not np.allclose(row["time_grid"], np.linspace(0, 1, nfe + 1), rtol=0, atol=1e-12):
            raise ValueError("JS anchor is not the executed uniform Euler grid.")
    return lookup, prompts


def _same_generator(left, right, audited_implementations):
    # This comparison is solely for reusing previously measured uniform anchors.
    # Source hashes are both recorded. Encoder, sampler settings and weights must
    # match; the JS policy code necessarily changes the package-wide code hash.
    def generator(binding):
        result = {**binding, "runtime": dict(binding["runtime"])}
        result["runtime"].pop("implementation_sha256", None)
        return result

    return [
        left["runtime"]["implementation_sha256"],
        right["runtime"]["implementation_sha256"],
    ] == audited_implementations and generator(left) == generator(right)


def _scorer_identity(payload):
    return {
        "scorer_versions": payload["versions"],
        "weight_fingerprints": payload["fingerprints"],
        "asset_manifest_sha256": payload["asset_manifest_sha256"],
    }


def verify_anchor_scores(anchors, raw, scoring):
    score_hash = canonical_sha256(scoring)
    matched = {}
    for row in raw:
        key = (row["prompt_id"], row["noise_seed"])
        if key not in anchors or row["clock_key"] != "uniform" or row["nfe"] != anchors[key]["nfe"]:
            continue
        if key in matched or row.get("completed") is not True or row["realized_nfe"] != row["nfe"]:
            raise ValueError("Anchor scoring source has duplicated or incomplete uniform observations.")
        expected = anchors[key]
        if expected["measurement_protocol"] != canonical_sha256([row["measurement_protocol"], score_hash]):
            raise ValueError("Anchor evidence does not bind the declared scorer provenance.")
        if any(row[key] != expected["metrics"][key] for key in ("preference", "alignment")):
            raise ValueError("Anchor metrics differ from the declared scored observations.")
        matched[key] = row
    if set(matched) != set(anchors):
        raise ValueError("Anchor scoring source is missing paired training observations.")


def run(config_path, *, phase):
    config = json.loads(Path(config_path).read_text())
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or any(not value.strip() for value in devices):
        raise RuntimeError("JS generation/scoring requires two visible GPUs, one per frozen runtime process.")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0].strip()
    destination = Path(config["output"]) / phase
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(config["manifest"]).read_text())
    calibration = RewardCalibration.from_payload(json.loads(Path(config["reward_calibration"]).read_text()))
    anchor_scoring = json.loads(Path(config["anchor_scoring"]).read_text())
    expected_scorer = {
        key: anchor_scoring[key] for key in ("scorer_versions", "weight_fingerprints", "asset_manifest_sha256")
    }
    if calibration.task != "sana" or calibration.solver != "euler":
        raise ValueError("The JS runtime currently supports SANA Euler only.")
    fit_config = JSFit(**config["fit"])
    rows = read_jsonl(config["anchor_evidence"])
    anchors, prompts = training_panel(rows, manifest, nfe=config["nfe"], noise_seeds=config["noise_seeds"])
    verify_anchor_scores(anchors, read_jsonl(config["anchor_measurements"]), expected_scorer)
    fit_config.updates(len(anchors))
    train_ids = {key[0] for key in anchors}
    heldout_ids = {r["prompt_id"] for r in manifest["records"] if r["split"] in ("validation", "locked_test")}
    if train_ids & heldout_ids or set(calibration.calibration_contexts) & heldout_ids:
        raise ValueError("Reward calibration or training overlaps validation/locked-test prompts.")
    identity = {
        "phase": phase,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "inputs": {
            key: sha256_file(config[key])
            for key in (
                "manifest",
                "reward_calibration",
                "anchor_evidence",
                "runtime_config",
                "anchor_scoring",
                "anchor_measurements",
            )
        },
    }
    write_new_json(destination / "identity.json", identity)
    runtime = load_runtime(config["runtime_config"])
    binding = runtime_binding(runtime)
    if any(
        not _same_generator(row["backbone_binding"], binding, config["audited_implementations"])
        or row["backbone"] != calibration.backbone
        for row in anchors.values()
    ):
        raise ValueError("JS anchors/calibration and the frozen runtime differ.")
    indices = sorted(anchors)
    context_ids = (
        train_ids if phase == "fit" else {r["prompt_id"] for r in manifest["records"] if r["split"] == "validation"}
    )
    contexts = {p: runtime.adapter.encode_context(p, prompts[p]["prompt"]) for p in sorted(context_ids)}
    if phase == "fit":
        for (prompt_id, _), anchor in anchors.items():
            if canonical_sha256(contexts[prompt_id].embedding.tolist()) != anchor["context_embedding_sha256"]:
                raise ValueError("JS native prompt embedding differs from its paired anchor.")

    def inputs(index):
        prompt_id, seed = indices[index]
        return runtime.adapter.policy_inputs(context=contexts[prompt_id], noise_seed=seed)

    torch.manual_seed(fit_config.seed)
    if phase == "fit":
        example = inputs(0)
        architecture = JSArchitecture(
            noise_channels=example["noise"].shape[1],
            text_dim=example["text"].shape[-1],
            pooled_dim=example["pooled"].shape[-1],
            target_nfe=config["nfe"],
        )
        policy = DirichletSchedulePolicy(architecture).to("cuda")
    else:
        policy, metadata = load_js(Path(config["output"]) / "fit/policy/policy.pt", device="cuda")
        if metadata["runtime_binding"] != binding or metadata["input_hashes"] != identity["inputs"]:
            raise ValueError("JS validation runtime or registered inputs differ from fitting.")
        if metadata["config_sha256"] != identity["config_sha256"] or policy.architecture.target_nfe != config["nfe"]:
            raise ValueError("JS validation settings differ from the registered training experiment.")
    worker = ScoreWorker(config["scorer_python"], destination / "scorer.log", cuda_device=devices[1].strip())
    observations = []
    try:
        if _scorer_identity(worker.request({"identity": True})) != expected_scorer:
            raise ValueError("JS live scorers differ from the paired anchor scorers.")
        with (destination / "observations.jsonl").open("x") as journal:

            def evaluate(prompt_id, seed, clock, *, extra):
                path = destination / f"image-{len(observations):05d}.png"
                image, trace = runtime.adapter.sample(context=contexts[prompt_id], noise_seed=seed, clock=clock)
                image_tensor_to_pil(image).save(path)
                scores = worker.request({"prompt": prompts[prompt_id]["prompt"], "image_path": str(path)})
                row = {
                    "prompt_id": prompt_id,
                    "noise_seed": seed,
                    "nfe": config["nfe"],
                    "nodes": list(clock.nodes),
                    "raw_intervals": list(clock.raw_intervals),
                    "image_path": str(path),
                    "image_sha256": sha256_file(path),
                    "trace": asdict(trace),
                    **scores,
                    **extra,
                }
                if phase == "fit":
                    anchor = anchors[(prompt_id, seed)]
                    row["utility"] = float(
                        calibration.scalar({**anchor, "metrics": scores, "anchor_metrics": anchor["metrics"]})
                    )
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                observations.append(row)
                return row.get("utility")

            if phase == "fit":

                def terminal(index, clock, update, member):
                    prompt_id, seed = indices[index]
                    return evaluate(prompt_id, seed, clock, extra={"update": update, "rollout": member})

                history = fit_js(
                    policy,
                    context_count=len(indices),
                    inputs=inputs,
                    evaluate=terminal,
                    config=fit_config,
                    on_update=lambda row: print(json.dumps(row), flush=True),
                )
                if len(observations) + len(anchors) != fit_config.trajectories:
                    raise RuntimeError("JS realized training trajectory budget differs from registration.")
            else:
                with torch.inference_mode():
                    for prompt_id in sorted(contexts):
                        for seed in config["noise_seeds"]:
                            request_id = canonical_sha256(["js-validation", fit_config.seed, prompt_id, seed])
                            clock_seed = int(request_id[:15], 16)
                            concentration = policy(
                                **stack_inputs(
                                    [runtime.adapter.policy_inputs(context=contexts[prompt_id], noise_seed=seed)]
                                )
                            )
                            intervals = sample_intervals(concentration, seed=clock_seed)[0, 0].cpu().numpy()
                            evaluate(prompt_id, seed, materialize_js(intervals), extra={"clock_seed": clock_seed})
                history = []
        runtime.verify_frozen()
        scorer_identity = worker.close()
        if _scorer_identity(scorer_identity) != expected_scorer:
            raise ValueError("JS scorers changed during execution.")
    finally:
        if worker.process.poll() is None:
            worker.process.terminate()
            worker.process.wait(timeout=60)
        if not worker.log.closed:
            worker.log.close()
    metadata = {
        "config_sha256": identity["config_sha256"],
        "runtime_binding": binding,
        "reward_calibration": calibration.to_payload(),
        "input_hashes": identity["inputs"],
        "fit": asdict(fit_config),
        "noise_seeds": config["noise_seeds"],
        "training_contexts": sorted(train_ids),
        "anchor_binding": next(iter(anchors.values()))["backbone_binding"],
        "scorers": scorer_identity,
        "parameter_count": sum(p.numel() for p in policy.parameters()),
        "clock_rng": "independent_seeded_joint_dirichlet",
        "history": history,
    }
    if phase == "fit":
        save_js(destination / "policy", policy, metadata)
        restored, _ = load_js(destination / "policy/policy.pt", device="cuda")
        with torch.inference_mode():
            example = stack_inputs([inputs(0)])
            torch.testing.assert_close(policy(**example), restored(**example), rtol=0, atol=0)
    write_new_json(
        destination / "complete.json",
        {
            **metadata,
            "new_trajectories": len(observations),
            "charged_anchor_trajectories": len(anchors) if phase == "fit" else 0,
            "normalized_reward_mean": float(np.mean([r["utility"] for r in observations])) if phase == "fit" else None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "validation"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config, phase=args.phase)


if __name__ == "__main__":
    main()
