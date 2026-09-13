"""Portable measured-utility evaluators using the package's frozen task runtimes."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np

from genode.gico.clocks import materialize
from genode.gico.evidence import content_hash
from genode.gico.policy import load_context_embedding_table
from genode.gico.rewards import measurement_metrics
from genode.gico.train_gico import read_rows


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def measurement_identity(row):
    """Address seed-specific targets independently from shared native contexts."""
    return content_hash([row["context_id"], row["seed"], row["reference_id"]])


def build_evaluator(config):
    """Build a callback from rows, contexts, runtime, cases, output and clock_seed.

    Paths are explicit. An optional ``python`` runs the supplied task environment
    in a separate process. Neither interface requires a private Python module.
    """
    required = {"rows", "contexts", "runtime", "cases", "output", "clock_seed"}
    if set(config) - (required | {"python", "clock_replicates"}) or not required <= config.keys():
        raise ValueError(f"Built-in selection requires {sorted(required)} and only documented options.")
    rows = read_rows(config["rows"])
    if not rows or any(r["split"] != "validation" for r in rows):
        raise ValueError("Selection panel must contain validation rows only.")
    anchors = [r for r in rows if r["schedule_key"] == "uniform"]
    if not anchors:
        raise ValueError("Selection requires measured uniform anchors.")
    contexts = load_context_embedding_table(config["contexts"])
    cases = json.loads(Path(config["cases"]).read_text(encoding="utf-8"))
    if any(r["context_id"] not in contexts or measurement_identity(r) not in cases for r in anchors):
        raise ValueError("Selection cases/native contexts do not cover every anchor.")
    runtime = json.loads(Path(config["runtime"]).read_text(encoding="utf-8"))
    if any(r["task"] != runtime["task"] for r in anchors):
        raise ValueError("Selection runtime task differs from its panel.")
    if type(config["clock_seed"]) is not int:
        raise ValueError("clock_seed must be an integer.")
    replicates = config.get("clock_replicates", 4)
    if type(replicates) is not int or replicates < 1:
        raise ValueError("clock_replicates must be positive.")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)

    runtime_instance = None

    def evaluate(candidate):
        nonlocal runtime_instance
        destination = output / candidate.checkpoint_id
        destination.mkdir()
        requests = []
        for anchor in anchors:
            count = replicates if candidate.student_kind == "GICO-sto-policy" else 1
            for replicate in range(count):
                clocks = []
                for member in range(anchor["ensemble_size"]):
                    identity = content_hash(
                        [
                            anchor["context_id"],
                            anchor["solver"],
                            anchor["nfe"],
                            anchor["seed"],
                            member,
                            replicate,
                        ]
                    )
                    mass = candidate.density(
                        contexts[anchor["context_id"]],
                        anchor["solver"],
                        anchor["nfe"],
                        seed=config["clock_seed"],
                        request_id=identity,
                    )
                    clocks.append(
                        {
                            "density_mass": mass.tolist(),
                            "time_grid": list(materialize(mass, anchor["solver"], anchor["nfe"])),
                            "clock_seed": config["clock_seed"],
                            "clock_request_id": identity,
                        }
                    )
                row = copy.deepcopy(anchor)
                row.update(
                    schedule_key="student",
                    selection_checkpoint_id=candidate.checkpoint_id,
                    clock_replicate=replicate,
                    student_kind=candidate.student_kind,
                )
                if candidate.student_kind == "GICO-sto-policy" and len(clocks) > 1:
                    row.pop("density_mass", None)
                    row.pop("time_grid", None)
                    row["sample_clocks"] = clocks
                else:
                    row.update(clocks[0])
                requests.append(
                    {
                        "anchor": anchor,
                        "student": row,
                        "clocks": clocks,
                        "context": contexts[anchor["context_id"]].tolist(),
                        "case": cases[measurement_identity(anchor)],
                    }
                )
        request = {"runtime": runtime, "requests": requests, "output": str(destination.resolve())}
        _write(destination / "request.json", request)
        if config.get("python"):
            subprocess.run(
                [
                    config["python"],
                    "-m",
                    "genode.gico.evaluators",
                    "--request",
                    str((destination / "request.json").resolve()),
                ],
                check=True,
            )
            return json.loads((destination / "measurements.json").read_text(encoding="utf-8"))
        if runtime_instance is None:
            from genode.gico.task_evaluators import load_task_evaluator

            runtime_instance = load_task_evaluator(runtime)
        return execute_request(request, runtime=runtime_instance)

    return evaluate


def execute_request(request, *, runtime=None):
    from genode.gico.task_evaluators import load_task_evaluator

    destination = Path(request["output"])
    runtime = load_task_evaluator(request["runtime"]) if runtime is None else runtime
    measured, uniform_cache = [], {}
    for index, item in enumerate(request["requests"]):
        anchor, student = item["anchor"], item["student"]
        runtime.verify_frozen()
        identity = content_hash(anchor)
        if identity not in uniform_cache:
            clock = {"density_mass": anchor["density_mass"], "time_grid": anchor["time_grid"]}
            actual = runtime.measure(
                anchor,
                [clock] * anchor["ensemble_size"],
                item["context"],
                item["case"],
                destination / f"anchor-{index}",
            )
            if any(
                not np.isclose(actual[k], anchor["metrics"][k], rtol=0, atol=1e-12) for k in measurement_metrics(anchor)
            ):
                raise ValueError("Executed uniform metrics differ from the frozen selection evidence; recollect it.")
            uniform_cache[identity] = actual
        student["metrics"] = runtime.measure(
            student, item["clocks"], item["context"], item["case"], destination / f"student-{index}"
        )
        runtime.verify_frozen()
        uniform = copy.deepcopy(anchor)
        uniform.update(metrics=uniform_cache[identity], clock_replicate=student["clock_replicate"])
        measured.extend((uniform, student))
    _write(destination / "measurements.json", measured)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    execute_request(json.loads(Path(args.request).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
