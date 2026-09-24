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
from genode.gico.image_objective import validate_image_rows
from genode.gico.policy import load_context_embedding_table, load_policy
from genode.gico.rewards import RewardCalibration, construct_rewards, validate_terminal_metrics
from genode.gico.train_gico import read_rows


def collapse_clock_replicates(rows):
    """Average complete paired clock draws before nonlinear terminal rewards."""
    validate_image_rows(rows)
    for row in rows:
        validate_terminal_metrics(row)
    if not any("clock_replicate" in row for row in rows):
        return rows
    from genode.gico.rewards import measurement_metrics

    panels = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        replicate = row.get("clock_replicate")
        if type(replicate) is not int or replicate < 0:
            raise ValueError("Every replicated report row requires a nonnegative clock_replicate.")
        key = tuple(row[k] for k in ("task", "backbone", "split", "context_id", "solver", "nfe", "seed"))
        members = panels[key][row["schedule_key"]]
        if replicate in members:
            raise ValueError("Duplicate report clock replicate.")
        members[replicate] = row
    collapsed = []
    for schedules in panels.values():
        if "uniform" not in schedules:
            raise ValueError("Replicated reports require paired uniform anchors.")
        uniform = schedules["uniform"]
        expected = set(range(len(uniform)))
        if any(set(draws) != expected for draws in schedules.values()):
            raise ValueError("Report clock replicate panels are incomplete or unpaired.")
        metric_keys = measurement_metrics(uniform[0])
        for row in uniform.values():
            if any(row["metrics"][key] != uniform[0]["metrics"][key] for key in metric_keys):
                raise ValueError("Repeated uniform measurements changed across clock replicates.")
        for draws in schedules.values():
            first = draws[0]
            for row in draws.values():
                if any(
                    row.get(k) != first.get(k)
                    for k in (
                        "ensemble_size",
                        "reference_id",
                        "measurement_protocol",
                        "molecule_feature_map",
                        "sample_block",
                        "reference_block",
                        "target",
                    )
                ):
                    raise ValueError("Report clock replicates changed their paired measurement assets.")
            collapsed.append(
                {
                    **first,
                    "metrics": {
                        key: float(np.mean([row["metrics"][key] for row in draws.values()])) for key in metric_keys
                    },
                }
            )
    return collapsed


def summarize_measurements(rows: list[dict], policy, *, split: str = "test", contexts: dict | None = None) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Reports require validation or test data.")
    if not rows or any(r["split"] != split for r in rows):
        raise ValueError(f"Reporting requires {split} measurements exclusively.")
    forbidden = set(policy.metadata["split_contexts"]["train"])
    if split == "test":
        forbidden.update(policy.metadata["split_contexts"]["validation"])
    for calibration in policy.metadata["reward_calibrations"].values():
        forbidden.update(calibration["calibration_contexts"])
    if forbidden & {r["context_id"] for r in rows}:
        raise ValueError("Report contexts overlap fitting/calibration data.")
    clock_identities = set()
    for row in rows:
        if row["task"] in ("cifar10", "imagenet64"):
            if row.get("image_objective") != policy.metadata.get("image_objective"):
                raise ValueError("Report image-objective identity differs from the frozen artifact.")
            for phase in ("train", "calibration", "validation") if split == "test" else ("train", "calibration"):
                provenance = policy.metadata["image_split_identities"][phase]
                from genode.gico.image_objective import image_split_fields

                if any(
                    set(values).intersection(provenance.get(key, [])) for key, values in image_split_fields(row).items()
                ):
                    raise ValueError("Report image targets/noise overlap fitting/calibration data.")
        if row["solver"] not in policy.metadata["solvers"]:
            raise ValueError("Report solver is absent from the frozen artifact.")
        if row["measurement_protocol"] not in policy.metadata["measurement_protocols"]:
            raise ValueError("Report measurement protocol differs from the frozen calibration.")
        if row.get("backbone_binding") != policy.metadata.get("backbone_binding"):
            raise ValueError("Report native backbone binding differs from the frozen artifact.")
        if (
            row["task"].startswith("molecule_")
            and row.get("molecule_feature_map") not in policy.metadata.get("molecular_feature_maps", {}).values()
        ):
            raise ValueError("Report molecular feature map differs from the frozen training map.")
        from genode.gico.clocks import REFERENCE_KEYS
        from genode.gico.reporting import _check_clock

        learned = row["schedule_key"] not in REFERENCE_KEYS and row.get("measurement_role") != "baseline"
        if learned and (
            row.get("policy_sha256") != policy.artifact_sha256 or row.get("policy_kind") != policy.policy_kind
        ):
            raise ValueError("Report policy identity/policy kind is required for learned-policy measurements.")
        if "policy_sha256" in row and (
            row["policy_sha256"] != policy.artifact_sha256 or row.get("policy_kind") != policy.policy_kind
        ):
            raise ValueError("Report policy identity/policy kind differs from the selected artifact.")
        if "sample_clocks" in row:
            if "density_mass" in row or "time_grid" in row:
                raise ValueError("Declare either a fixed clock or sample_clocks, not both.")
            if len(row["sample_clocks"]) != row["ensemble_size"]:
                raise ValueError("Report requires exactly one clock per ensemble member.")
            for clock in row["sample_clocks"]:
                verify_measurement_clock(
                    {**row, "density_mass": clock["density_mass"], "time_grid": clock["time_grid"]}
                )
        else:
            verify_measurement_clock(row)
        if learned:
            if contexts is None or row["context_id"] not in contexts:
                raise ValueError("Learned-policy reports require native contexts for executed clock replay.")
            _check_clock({**row, "schedule_key": "policy"}, policy, contexts[row["context_id"]], clock_identities)
    raw_rows = rows
    rows = collapse_clock_replicates(raw_rows)
    measurements_sha256 = content_hash(raw_rows)
    groups = defaultdict(list)
    for solver, calibration in policy.metadata["reward_calibrations"].items():
        subset = [r for r in rows if r["solver"] == solver]
        if subset:
            for row in construct_rewards(subset, RewardCalibration.from_payload(calibration), varying_clocks=True):
                learned = row["schedule_key"] not in REFERENCE_KEYS and row.get("measurement_role") != "baseline"
                schedule = policy.policy_kind if learned else row["schedule_key"]
                groups[(solver, row["nfe"], schedule)].append(row)
    output = []
    for (solver, nfe, schedule), cells in sorted(groups.items()):
        units = cells
        if cells[0]["task"] == "imagenet64":
            # Classes in a paired block share sampling/reference uncertainty.
            # Average all classes equally first; uncertainty is across blocks.
            panels = defaultdict(list)
            for cell in cells:
                panels[cell["panel_id"]].append(cell)
            units = []
            for panel in panels.values():
                if len(panel) != 1000 or {r.get("class_id") for r in panel} != set(range(1000)):
                    raise ValueError("ImageNet reports require all 1000 classes exactly once per paired panel.")
                units.append(
                    {
                        "reward": float(np.mean([r["reward"] for r in panel])),
                        "metrics": {k: float(np.mean([r["metrics"][k] for r in panel])) for k in panel[0]["metrics"]},
                    }
                )
        rewards = np.array([r["reward"] for r in units])
        sem = float(rewards.std(ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else None
        width = float(t.ppf(0.975, len(rewards) - 1) * sem) if sem is not None else None
        output.append(
            {
                "solver": solver,
                "nfe": nfe,
                "schedule": schedule,
                "paired_contexts": len(cells),
                "independent_units": len(units),
                "reward_mean": float(rewards.mean()),
                "reward_standard_error": sem,
                "reward_ci95": [float(rewards.mean() - width), float(rewards.mean() + width)]
                if width is not None
                else None,
                "raw_metrics": {k: float(np.mean([r["metrics"][k] for r in units])) for k in units[0]["metrics"]},
            }
        )
    return {
        "artifact_sha256": policy.artifact_sha256,
        "policy_kind": policy.policy_kind,
        "measurements_sha256": measurements_sha256,
        "split": split,
        "results": output,
        "selection_performed": False,
        "uncertainty_unit": "paired_panel_mean_over_classes_and_replicates"
        if rows[0]["task"] == "imagenet64"
        else "paired_context_mean_over_replicates",
    }


def report_main(*, default_split: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--policy-kind", choices=("deterministic", "stochastic"), required=True)
    parser.add_argument("--rows", required=True, help="Paired JSONL terminal measurements, including uniform anchors.")
    parser.add_argument("--contexts", help="Native context NPZ required for learned-policy clock replay.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize_measurements(
        read_rows(args.rows),
        load_policy(args.artifact, policy_kind=args.policy_kind),
        split=default_split,
        contexts=load_context_embedding_table(args.contexts) if args.contexts else None,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main() -> None:
    report_main(default_split="test")


if __name__ == "__main__":
    main()
