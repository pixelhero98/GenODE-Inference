"""Report paired terminal measurements using the fitted artifact's frozen calibration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from genode.gico.clocks import verify_measurement_clock
from genode.gico.evidence import content_hash
from genode.gico.policy import load_policy
from genode.gico.rewards import RewardCalibration, construct_rewards
from genode.gico.train_gico import read_rows


def summarize_measurements(rows: list[dict], policy, *, split: str = "test") -> dict:
    if not rows or any(r["split"] != split for r in rows):
        raise ValueError(f"Reporting requires {split} measurements exclusively.")
    forbidden = set(policy.metadata["split_contexts"]["train"])
    if split == "test":
        forbidden.update(policy.metadata["split_contexts"]["validation"])
    for calibration in policy.metadata["reward_calibrations"].values():
        forbidden.update(calibration["calibration_contexts"])
    if forbidden & {r["context_id"] for r in rows}:
        raise ValueError("Report contexts overlap fitting/calibration data.")
    for row in rows:
        if row["solver"] not in policy.metadata["solvers"]:
            raise ValueError("Report solver is absent from the frozen artifact.")
        verify_measurement_clock(row)
    groups = defaultdict(list)
    for solver, calibration in policy.metadata["reward_calibrations"].items():
        subset = [r for r in rows if r["solver"] == solver]
        if subset:
            for row in construct_rewards(subset, RewardCalibration.from_payload(calibration)):
                groups[(solver, row["nfe"], row["schedule_key"])].append(row)
    output = []
    for (solver, nfe, schedule), cells in sorted(groups.items()):
        rewards = np.array([r["reward"] for r in cells])
        sem = float(rewards.std(ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else None
        width = float(t.ppf(0.975, len(rewards) - 1) * sem) if sem is not None else None
        output.append(
            {
                "solver": solver,
                "nfe": nfe,
                "schedule": schedule,
                "paired_contexts": len(cells),
                "reward_mean": float(rewards.mean()),
                "reward_standard_error": sem,
                "reward_ci95": [float(rewards.mean() - width), float(rewards.mean() + width)]
                if width is not None
                else None,
                "raw_metrics": {k: float(np.mean([r["metrics"][k] for r in cells])) for k in cells[0]["metrics"]},
            }
        )
    return {
        "artifact_sha256": policy.artifact_sha256,
        "measurements_sha256": content_hash(rows),
        "split": split,
        "results": output,
        "selection_performed": False,
        "uncertainty_unit": "paired_context_mean_over_replicates",
    }


def report_main(*, default_split: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--student-kind", choices=("deterministic", "stochastic"), required=True)
    parser.add_argument("--rows", required=True, help="Paired JSONL terminal measurements, including uniform anchors.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize_measurements(
        read_rows(args.rows), load_policy(args.artifact, student_kind=args.student_kind), split=default_split
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main() -> None:
    report_main(default_split="test")


if __name__ == "__main__":
    main()
