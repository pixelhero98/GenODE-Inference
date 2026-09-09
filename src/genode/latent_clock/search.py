from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from genode.gico.conditioning import EmbeddingNormalizer
from genode.latent_clock.artifacts import canonical_sha256, read_jsonl, sha256_file, write_new_json, write_new_jsonl
from genode.latent_clock.bo import fit_bo_clock
from genode.latent_clock.clocks import PG_CLOCK_PRECISION, Clock, bo_bounds, reference_clocks
from genode.latent_clock.collection import image_tensor_to_pil
from genode.latent_clock.pg import fit_pg_clock_policy
from genode.latent_clock.protocol import BUDGETS, NOISE_SEEDS, RewardScales
from genode.latent_clock.runtime import load_runtime


def serve_scores() -> None:
    from genode.latent_clock.rewards import FrozenDualScorer

    with contextlib.redirect_stdout(sys.stderr):
        scorer = FrozenDualScorer()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            if request.get("close"):
                scorer.verify_frozen()
                result = {
                    "closed": True,
                    "fingerprints": scorer._fingerprints,
                    "versions": scorer.versions,
                    "asset_manifest_sha256": scorer.asset_manifest_sha256,
                }
            else:
                import torch

                torch.cuda.synchronize()
                started = time.perf_counter()
                preference, alignment = scorer.score(request["prompt"], request["image_path"])
                torch.cuda.synchronize()
                result = {
                    "preference": preference,
                    "alignment": alignment,
                    "scorer_gpu_seconds": time.perf_counter() - started,
                }
        print(json.dumps(result), flush=True)
        if request.get("close"):
            return


class ScoreWorker:
    def __init__(self, python: str, log: Path, *, cuda_device: str):
        self.log = log.open("a")
        self.log.write(f"\nScorer startup for job {os.environ.get('GENODE_JOB_ID', 'unknown')}\n")
        self.log.flush()
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_device}
        self.process = subprocess.Popen(
            [python, "-m", "genode.latent_clock.cli", "serve-scores"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=environment,
        )
        if self._read() != {"ready": True}:
            raise RuntimeError("Scoring subprocess failed to initialize.")

    def _read(self) -> dict:
        value = self.process.stdout.readline()
        if not value:
            raise RuntimeError("Scoring subprocess stopped; see scorer.log.")
        return json.loads(value)

    def request(self, value: dict) -> dict:
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        return self._read()

    def close(self) -> dict:
        result = self.request({"close": True})
        self.process.wait(timeout=60)
        self.log.close()
        return result


def fit_search(
    *,
    method: str,
    runtime_config: str,
    manifest_path: str,
    scales_path: str,
    anchors_path: str,
    scorer_python: str,
    nfe: int,
    budget: str,
    seed: int,
    output: str,
) -> None:
    import torch

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or any(not device.strip() for device in devices):
        raise RuntimeError("BO/PG require two visible GPUs: one generator process and one legacy scorer process.")
    # Each worker owns one GPU and an independent CUDA context.
    # Set visibility before either process initializes CUDA; each sees local cuda:0.
    generator_device, scorer_device = (device.strip() for device in devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = generator_device
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "method": method,
        "nfe": nfe,
        "budget": budget,
        "seed": seed,
        "runtime_sha256": sha256_file(runtime_config),
        "manifest_sha256": sha256_file(manifest_path),
        "scales_sha256": sha256_file(scales_path),
        "anchors_sha256": sha256_file(anchors_path),
    }
    identity_path = destination / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Search resume directory has different inputs or settings.")
    else:
        write_new_json(identity_path, identity)
    if (destination / "complete.json").exists():
        return
    manifest = json.loads(Path(manifest_path).read_text())
    prompts = [row for row in manifest["records"] if row["split"] == "calibration"][: BUDGETS[budget]["prompts"]]
    prompt_lookup = {p["prompt_id"]: p for p in prompts}
    raw_anchors = [
        r
        for r in read_jsonl(anchors_path)
        if r["split"] == "calibration"
        and r["clock_key"] == "uniform"
        and int(r["nfe"]) == nfe
        and r["prompt_id"] in prompt_lookup
    ]
    anchors = {(r["prompt_id"], int(r["noise_seed"])): r for r in raw_anchors}
    if len(raw_anchors) != 2 * len(prompts) or set(anchors) != {(p, s) for p in prompt_lookup for s in NOISE_SEEDS}:
        raise ValueError("Uniform anchors are not the complete matched calibration panel.")
    scales_payload = json.loads(Path(scales_path).read_text())
    scales = RewardScales(scales_payload["preference"], scales_payload["alignment"])
    runtime = load_runtime(runtime_config)
    contexts = {p["prompt_id"]: runtime.adapter.encode_context(p["prompt_id"], p["prompt"]) for p in prompts}
    worker = ScoreWorker(scorer_python, destination / "scorer.log", cuda_device=scorer_device)
    journal_path = destination / "observations.jsonl"
    recorded = read_jsonl(journal_path) if journal_path.exists() else []
    records = []
    journal = journal_path.open("a")
    started = time.perf_counter()

    def evaluate(prompt_id: str, clock: Clock) -> float:
        utilities = []
        for noise_seed in NOISE_SEEDS:
            if len(records) < len(recorded):
                row = recorded[len(records)]
                if (
                    row["prompt_id"] != prompt_id
                    or row["noise_seed"] != noise_seed
                    or not np.allclose(row["nodes"], clock.nodes, rtol=0, atol=1e-10)
                ):
                    raise ValueError("Deterministic search replay diverged from recorded observations.")
                if sha256_file(row["image_path"]) != row["image_sha256"]:
                    raise ValueError("A cached search image changed.")
                records.append(row)
                utilities.append(row["utility"])
                continue
            anchor = anchors[(prompt_id, noise_seed)]
            if np.allclose(clock.nodes, np.linspace(0, 1, nfe + 1), rtol=0, atol=1e-12):
                row = {
                    **anchor,
                    "physical_reuse_key": anchor["image_sha256"],
                    "physical_generated": False,
                    "new_generator_gpu_seconds": 0.0,
                    "new_scorer_gpu_seconds": 0.0,
                }
            else:
                path = destination / f"image-{len(records):05d}.png"
                image, trace = runtime.adapter.sample(context=contexts[prompt_id], noise_seed=noise_seed, clock=clock)
                image_tensor_to_pil(image).save(path)
                scores = worker.request({"prompt": prompt_lookup[prompt_id]["prompt"], "image_path": str(path)})
                row = {
                    **prompt_lookup[prompt_id],
                    **scores,
                    "image_path": str(path),
                    "image_sha256": sha256_file(path),
                    "physical_generated": True,
                    "noise_seed": noise_seed,
                    "trace": asdict(trace),
                    "new_generator_gpu_seconds": trace.elapsed_gpu_seconds,
                    "new_scorer_gpu_seconds": scores["scorer_gpu_seconds"],
                }
            utility = scales.utility(row["preference"], row["alignment"], anchor["preference"], anchor["alignment"])
            row.update(
                nfe=nfe,
                method=method,
                optimizer_seed=seed,
                budget=budget,
                clock_key=clock.key,
                nodes=list(clock.nodes),
                clock_precision_corrected=clock.source_kind == "pg_precision_corrected",
                completed=True,
                **utility,
            )
            records.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            utilities.append(utility["utility"])
        return float(np.mean(utilities))

    try:
        if method == "bo":
            config_path = destination / "bo-configuration.json"
            if not config_path.exists():
                lower, upper = bo_bounds(nfe)
                write_new_json(
                    config_path,
                    {
                        "lower_bounds": lower.tolist(),
                        "upper_bounds": upper.tolist(),
                        "minimum_representable_interval": 2.0**-24,
                        "panel_queries": 25,
                        "initial_panels_including_uniform": max(8, 2 * (nfe - 1)),
                        "kernel": "Matern-5/2",
                        "acquisition": "qLogNoisyExpectedImprovement",
                        "gp": "SingleTaskGP",
                        "proposal_rejections_use_no_reward_observations": True,
                    },
                )

            def panel(clock: Clock) -> tuple[float, float]:
                values = np.asarray([evaluate(prompt["prompt_id"], clock) for prompt in prompts])
                print(
                    json.dumps(
                        {
                            "method": "bo",
                            "query": len(records) // (2 * len(prompts)),
                            "mean_utility": float(values.mean()),
                        }
                    ),
                    flush=True,
                )
                return float(values.mean()), float(values.var(ddof=1) / len(values))

            selected, observations = fit_bo_clock(nfe, panel, seed=seed)
            if not (destination / "clock.json").exists():
                write_new_json(destination / "clock.json", asdict(selected))
            if not (destination / "queries.jsonl").exists():
                write_new_jsonl(destination / "queries.jsonl", [asdict(row) for row in observations])
        elif method == "pg":
            for prompt in prompts:
                evaluate(prompt["prompt_id"], next(c for c in reference_clocks(nfe) if c.key == "uniform"))
            ordered_ids = sorted(contexts, key=lambda p: canonical_sha256(["holdout-v1", p]))
            fit_ids = ordered_ids[max(1, round(0.2 * len(ordered_ids))) :]
            embeddings = {key: value.embedding for key, value in contexts.items()}
            normalizer = EmbeddingNormalizer.fit(embeddings, fit_ids)
            transformed = normalizer.transform_table(embeddings)
            policy, rollouts = fit_pg_clock_policy(
                transformed, nfe, lambda prompt_id, clock, _: evaluate(prompt_id, clock), seed=seed
            )
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "context_dim": len(next(iter(transformed.values()))),
                    "nfe": nfe,
                    "optimizer_seed": seed,
                    "embedding_normalizer": normalizer.to_payload(),
                    "protocol": {
                        "name": "DDPO-inspired clock PG",
                        "rounds_after_uniform_anchor": 24,
                        "width": 256,
                        "hidden_layers": 2,
                        "learning_rate": 3e-4,
                        "ppo_clip": 0.2,
                        "update_epochs": 4,
                        "kl_coefficient": 0.01,
                        "entropy_coefficient": 0.001,
                        "clock_precision": PG_CLOCK_PRECISION,
                    },
                },
                destination / "policy.pt",
            )
            if not (destination / "rollouts.jsonl").exists():
                write_new_jsonl(destination / "rollouts.jsonl", [asdict(row) for row in rollouts])
        else:
            raise ValueError("Search method must be bo or pg.")
        if len(records) != BUDGETS[budget]["trajectories"]:
            raise RuntimeError("Search exceeded or failed to consume its preregistered data-access budget.")
        runtime.verify_frozen()
        scorer_state = worker.close()
    except BaseException as exc:
        with (destination / "failed-attempts.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "job_id": os.environ.get("GENODE_JOB_ID", "unknown"),
                        "error": repr(exc),
                        "recorded_observations": len(records),
                        "elapsed_seconds": time.perf_counter() - started,
                        "clock_precision": PG_CLOCK_PRECISION if method == "pg" else None,
                    }
                )
                + "\n"
            )
        raise
    finally:
        journal.close()
        if worker.process.poll() is None:
            worker.process.terminate()
            worker.process.wait(timeout=60)
            worker.log.close()
    write_new_json(
        destination / "complete.json",
        {
            "method": method,
            "budget": budget,
            "nfe": nfe,
            "optimizer_seed": seed,
            "logical_trajectories": len(records),
            "physical_trajectories": sum(r["physical_generated"] for r in records),
            "gpu_allocation": {"generator_process": 1, "scorer_process": 1, "total": 2},
            "clock_precision": PG_CLOCK_PRECISION if method == "pg" else None,
            "precision_corrected_trajectories": sum(r.get("clock_precision_corrected", False) for r in records),
            "elapsed_seconds": time.perf_counter() - started,
            "scorers": scorer_state,
            "weights": runtime.fingerprints,
            "manifest_sha256": sha256_file(manifest_path),
            "scales_sha256": sha256_file(scales_path),
            "anchors_sha256": sha256_file(anchors_path),
        },
    )
