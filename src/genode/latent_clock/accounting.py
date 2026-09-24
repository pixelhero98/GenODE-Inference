"""Report physical work once and each frozen method's access to calibration evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from genode.latent_clock.artifacts import read_jsonl, write_csv, write_new_json


def write_cost_ledger(root: str | Path, destination: Path, methods: list[dict]) -> None:
    root = Path(root)
    physical = defaultdict(lambda: defaultdict(float))
    seen = set()
    for source in sorted(root.rglob("images.jsonl")) + sorted((root / "fits").glob("*/observations.jsonl")):
        for row in read_jsonl(source):
            key = row["image_path"]
            if key in seen or row.get("physical_generated") is False:
                continue
            seen.add(key)
            phase = row["split"]
            trace = row["trace"]
            physical[phase]["completed_trajectories"] += 1
            for name in ("realized_nfe", "backbone_forwards", "cfg_sample_equivalents", "elapsed_gpu_seconds"):
                physical[phase][name] += trace[name]
    for source in sorted(root.rglob("*-scores.jsonl")):
        for row in read_jsonl(source):
            phase = row["split"]
            physical[phase]["image_reward_calls"] += row.get("image_reward_calls", 0)
            physical[phase]["vqa_score_calls"] += row.get("vqa_score_calls", 0)
            physical[phase]["scorer_gpu_seconds"] += row.get("scorer_gpu_seconds", 0)
    for source in sorted((root / "fits").glob("*/observations.jsonl")):
        for row in read_jsonl(source):
            if row.get("physical_generated"):
                physical["calibration"]["image_reward_calls"] += 1
                physical["calibration"]["vqa_score_calls"] += 1
                physical["calibration"]["scorer_gpu_seconds"] += row["new_scorer_gpu_seconds"]
    for source in sorted(root.rglob("visionreward/scores.jsonl")):
        for row in read_jsonl(source):
            physical["visionreward"]["scorer_calls"] += row["vision_reward_calls"]
            physical["visionreward"]["module_forwards"] += row["vision_module_forwards"]
            physical["visionreward"]["scorer_gpu_seconds"] += row["vision_gpu_seconds"]
    for source in sorted(root.rglob("geneval-results.jsonl.complete.json")):
        metadata = json.loads(source.read_text())
        physical["geneval"]["detector_calls"] += metadata["detector_calls"]
        physical["geneval"]["scorer_gpu_seconds"] += metadata["elapsed_gpu_seconds"]
        for name, counts in metadata["model_counts"].items():
            physical["geneval"][name + "_forwards"] += counts["forwards"]
            physical["geneval"][name + "_sample_equivalents"] += counts["sample_equivalents"]
    logical = []
    for method in methods:
        row = {
            "method": method["name"],
            "group": method["group"],
            "nfe": method["nfe"],
            "budget": method["budget"],
            "shared_pilot_trajectories_charged": 384,
            "shared_pilot_field_evaluations_charged": 2304,
            "calibration_trajectories_accessed": 0,
        }
        if method["method"] == "gico":
            from genode.gico.policy import load_policy

            metadata = load_policy(method["checkpoint"], policy_kind=method["policy_kind"]).metadata
            row["calibration_trajectories_accessed"] = sum(metadata["measurement_counts"].values())
            row["fit_wall_seconds"] = metadata["fitting_wall_seconds"]
        elif method["fit_report"]:
            fit = json.loads(Path(method["fit_report"]).read_text())
            row["calibration_trajectories_accessed"] = fit["logical_trajectories"]
            row["fit_gpu_seconds"] = fit.get("gpu_seconds")
            row["fit_elapsed_seconds"] = fit.get("elapsed_seconds")
        elif method["method"] == "ld3":
            provenance = json.loads(Path(method["checkpoint"]).read_text())["provenance"]
            row.update(
                {
                    key: provenance[key]
                    for key in ("teacher_trajectories", "teacher_gpu_seconds", "optimization_gpu_seconds")
                }
            )
            row["budget_matched_to_gico"] = False
        logical.append(row)
    failures = []
    for source in sorted(root.rglob("failed-attempts.jsonl")):
        failures.extend({"source": str(source), **row} for row in read_jsonl(source))
    parity = {str(path): json.loads(path.read_text()) for path in root.glob("*-parity.json")}
    runtime_audits = {str(path): json.loads(path.read_text()) for path in root.glob("*-runtime-check.json")}
    precision_audits = {
        str(path): json.loads(path.read_text()) for path in root.glob("pg-precision-check-*/complete.json")
    }
    memory_audits = {str(path): json.loads(path.read_text()) for path in root.glob("ld3-memory-check-*/complete.json")}
    write_new_json(
        destination / "cost-ledger.json",
        {
            "physical_by_phase": dict(physical),
            "logical_by_method": logical,
            "failed_attempts": failures,
            "sampler_audits": parity,
            "runtime_audits": runtime_audits,
            "precision_repair_audits": precision_audits,
            "ld3_memory_audits": memory_audits,
            "note": "Allocation accounting is external; instrumented computation and fitting wall time are reported separately.",
        },
    )
    write_csv(destination / "method-data-access.csv", logical)
    write_csv(
        destination / "physical-computation.csv", [{"phase": phase, **values} for phase, values in physical.items()]
    )
