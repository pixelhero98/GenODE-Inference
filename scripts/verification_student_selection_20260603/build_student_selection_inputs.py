from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from genode.gipo.policy import (
    attach_uniform_gipo_rewards,
    context_id_from_row,
    read_metric_rows_csv,
    sample_context_ids_stratified,
    split_rows_by_context_holdout,
    split_rows_by_series_holdout,
)
from genode.gipo.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


RUN_SPECS: tuple[dict[str, str], ...] = (
    {"source": "ess_bins", "run_id": "temp_fixed_005_b128"},
    {"source": "ess_bins", "run_id": "temp_ess20_b128"},
    {"source": "ess_bins", "run_id": "temp_ess25_b128"},
    {"source": "ess_bins", "run_id": "temp_ess30_b128"},
    {"source": "ess_bins", "run_id": "bin_ess25_b64"},
    {"source": "ess_bins", "run_id": "bin_ess25_b256"},
    {"source": "density_diversity", "run_id": "diverse_fixed005_b128"},
    {"source": "density_diversity", "run_id": "diverse_ess25_b128"},
    {"source": "density_diversity", "run_id": "diverse_ess40_b128"},
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _source_root(source: str, args: argparse.Namespace) -> Path:
    if source == "ess_bins":
        return Path(args.ess_root)
    if source == "density_diversity":
        return Path(args.diversity_root)
    raise ValueError(f"Unknown source {source!r}.")


def _calibration_manifest(source_root: Path) -> dict[str, Any]:
    matrix_path = source_root / "summary" / "verification_matrix.json"
    if matrix_path.exists():
        payload = _read_json(matrix_path)
        manifest = dict(payload.get("calibration_manifest", {}) or {})
        if manifest:
            return manifest
    cal_root = source_root / "calibration_inputs"
    return {
        "calibration_rows": str(cal_root / "context_calibration_rows.csv"),
        "calibration_embeddings": str(cal_root / "context_calibration_embeddings.npz"),
    }


def _copy_with_selection(rows: Iterable[Mapping[str, Any]], selection_split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["source_split_phase"] = str(row.get("split_phase", row.get("split", "")))
        copied["selection_split"] = str(selection_split)
        out.append(copied)
    return out


def _rows_for_context_ids(rows: Sequence[Mapping[str, Any]], context_ids: Sequence[str]) -> list[dict[str, Any]]:
    wanted = {str(context_id) for context_id in context_ids}
    return [dict(row) for row in rows if context_id_from_row(row) in wanted]


def _reconstruct_holdouts(
    rows: Sequence[Mapping[str, Any]],
    training_summary: Mapping[str, Any],
    *,
    seed: int,
    context_holdout_fraction: float,
    series_holdout_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    support_keys = tuple(str(key) for key in training_summary["support_schedule_keys"])
    support = set(support_keys)
    support_rows = [dict(row) for row in rows if str(row.get("scheduler_key")) in support]
    membership = dict(training_summary.get("split_membership", {}) or {})
    context_ids = list(dict(membership.get("context_disjoint", {}) or {}).get("context_ids", []) or [])
    series_ids = list(dict(membership.get("series_disjoint", {}) or {}).get("context_ids", []) or [])
    if context_ids or series_ids:
        return _rows_for_context_ids(support_rows, context_ids), _rows_for_context_ids(support_rows, series_ids), "training_summary_membership"

    rewarded_rows = attach_uniform_gipo_rewards(support_rows, support_schedule_keys=support_keys, pair_on_seed=True)
    sample_count = int(training_summary.get("sampled_context_count", 0) or 0)
    selected_context_ids = set(
        sample_context_ids_stratified(rewarded_rows, sample_count=sample_count, seed=int(seed))
    )
    sampled_rows = [row for row in rewarded_rows if context_id_from_row(row) in selected_context_ids]
    context_fit_pool_rows, context_holdout_rows = split_rows_by_context_holdout(
        sampled_rows,
        holdout_fraction=float(context_holdout_fraction),
        seed=int(seed),
    )
    _fit_rows, series_holdout_rows = split_rows_by_series_holdout(
        context_fit_pool_rows,
        holdout_fraction=float(series_holdout_fraction),
        seed=int(seed),
    )
    return context_holdout_rows, series_holdout_rows, "deterministic_reconstruction"


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.out_root)
    input_root = root / "selection_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    manifest_runs: list[dict[str, Any]] = []

    for spec in RUN_SPECS:
        source_root = _source_root(spec["source"], args)
        run_id = spec["run_id"]
        run_dir = source_root / "policy_runs" / run_id
        training_summary_path = run_dir / "gipo_training_summary.json"
        training_summary = _read_json(training_summary_path)
        calibration_manifest = _calibration_manifest(source_root)
        calibration_rows_path = Path(calibration_manifest["calibration_rows"])
        calibration_embeddings_path = Path(calibration_manifest["calibration_embeddings"])
        rows = read_metric_rows_csv(calibration_rows_path)
        context_rows, series_rows, split_source = _reconstruct_holdouts(
            rows,
            training_summary,
            seed=int(args.seed),
            context_holdout_fraction=float(args.context_holdout_fraction),
            series_holdout_fraction=float(args.series_holdout_fraction),
        )
        run_input_dir = input_root / run_id
        context_path = run_input_dir / "context_disjoint_rows.csv"
        series_path = run_input_dir / "series_disjoint_rows.csv"
        fixed_reference_path = run_input_dir / "fixed_reference_rows.csv"
        ser_reference_path = run_input_dir / "ser_reference_rows.csv"
        _write_rows(context_path, _copy_with_selection(context_rows, "context_disjoint"))
        _write_rows(series_path, _copy_with_selection(series_rows, "series_disjoint"))
        support_rows = [dict(row) for row in rows if str(row.get("scheduler_key")) in set(training_summary["support_schedule_keys"])]
        _write_rows(
            fixed_reference_path,
            [row for row in support_rows if str(row.get("scheduler_key")) in set(BASELINE_SCHEDULE_KEYS)],
        )
        _write_rows(
            ser_reference_path,
            [row for row in support_rows if str(row.get("scheduler_key")) == SER_PTG_SCHEDULE_KEY],
        )
        manifest_runs.append(
            {
                "source": spec["source"],
                "source_root": str(source_root),
                "run_id": run_id,
                "policy_run_dir": str(run_dir),
                "training_summary": str(training_summary_path),
                "student_checkpoint": str(run_dir / "gipo_student.pt"),
                "context_rows": str(context_path),
                "series_rows": str(series_path),
                "fixed_reference_rows": str(fixed_reference_path),
                "ser_reference_rows": str(ser_reference_path),
                "context_disjoint_row_count": int(len(context_rows)),
                "series_disjoint_row_count": int(len(series_rows)),
                "selection_input_source": split_source,
                "calibration_embeddings": str(calibration_embeddings_path),
                "density_bin_count": int(dict(training_summary.get("density_representation", {})).get("reference_bin_count", 0) or 0),
                "locked_test_used_for_selection": False,
            }
        )

    manifest = {
        "artifact": "genode_student_selection_inputs_manifest",
        "locked_test_used_for_selection": False,
        "run_count": int(len(manifest_runs)),
        "runs": manifest_runs,
    }
    (input_root / "student_selection_inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build calibration-only context holdout rows for student selection.")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--ess_root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602")
    parser.add_argument("--diversity_root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--series_holdout_fraction", type=float, default=0.20)
    return parser


def main() -> None:
    payload = build_inputs(build_argparser().parse_args())
    print(json.dumps({"run_count": payload["run_count"], "locked_test_used_for_selection": False}, sort_keys=True))


if __name__ == "__main__":
    main()
