from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.gipo.policy import (
    GIPO_PROTOCOL,
    GIPODensityStudentMLP,
    EmbeddingNormalizer,
    context_id_from_row,
    load_context_embedding_table,
    predict_gipo_density,
    read_metric_rows_csv,
    remapped_series_index,
)
from genode.gipo.evaluate_schedule_summary import build_comparison_summary
from genode.gipo.models import solver_macro_steps
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    project_paper_dataset_root,
    project_outputs_root,
    resolve_project_path,
)
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    LOCKED_TEST_PHASE,
    SOLVER_RUNTIME_NAMES,
    TRAIN_TUNING_PHASE,
    VALIDATION_PHASE,
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
)
from genode.runtime import ProgressBar, resolve_torch_device

GIPO_SCHEDULE_KEY = "gipo"
SELECTION_MODE_REPORTING = "reporting"
SELECTION_MODE_CALIBRATION = "calibration"
CONTEXT_DISJOINT_PHASE = "context_disjoint"
SERIES_DISJOINT_PHASE = "series_disjoint"
CALIBRATION_HOLDOUT_PHASES = (CONTEXT_DISJOINT_PHASE, SERIES_DISJOINT_PHASE)
SOURCE_SPLIT_PHASES = (TRAIN_TUNING_PHASE, VALIDATION_PHASE, LOCKED_TEST_PHASE)


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _read_csvs(paths_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path_text in _parse_csv(paths_text):
        rows.extend(read_metric_rows_csv(resolve_project_path(path_text)))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
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


def _source_split_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))).strip()


def _selection_split(row: Mapping[str, Any]) -> str:
    return str(row.get("selection_split") or row.get("report_split") or row.get("split_phase", row.get("split", ""))).strip()


def _validate_context_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_phase: str,
    selection_mode: str,
) -> None:
    if not rows:
        raise ValueError("context rows input contains no rows.")
    mode = str(selection_mode)
    requested = str(split_phase)
    if mode == SELECTION_MODE_CALIBRATION:
        locked = [
            row
            for row in rows
            if _source_split_phase(row) == LOCKED_TEST_PHASE or str(row.get("split_phase", row.get("split", ""))) == LOCKED_TEST_PHASE
        ]
        if locked:
            raise ValueError("Calibration GIPO selection refuses locked_test rows.")
    if requested in CALIBRATION_HOLDOUT_PHASES:
        bad_selection = sorted({_selection_split(row) for row in rows if _selection_split(row) != requested})
        if bad_selection:
            raise ValueError(f"GIPO calibration reporter expected selection_split={requested!r}; found {bad_selection}.")
        bad_source = sorted({_source_split_phase(row) for row in rows if _source_split_phase(row) not in (TRAIN_TUNING_PHASE, VALIDATION_PHASE)})
        if bad_source:
            raise ValueError(f"Calibration holdout rows require train/validation source split phases; found {bad_source}.")
        return
    bad = sorted({_source_split_phase(row) for row in rows if _source_split_phase(row) != requested})
    if bad:
        raise ValueError(f"GIPO reporter expected source split_phase={requested!r}; found {bad}.")


def _evaluation_seed_from_row(row: Mapping[str, Any]) -> int:
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return int(row["seed"])


def _row_group_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str, int, str]:
    return (
        _source_split_phase(row),
        str(row.get("dataset", row.get("dataset_key", ""))),
        _evaluation_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
    )


def _representative_context_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, str, int, str], Dict[str, Any]] = {}
    for row in rows:
        key = _row_group_key(row)
        if key not in grouped:
            copied = dict(row)
            copied["context_id"] = context_id_from_row(row)
            copied["evaluation_seed"] = _evaluation_seed_from_row(row)
            copied["source_split_phase"] = _source_split_phase(row)
            copied["selection_split"] = _selection_split(row)
            grouped[key] = copied
    return [grouped[key] for key in sorted(grouped)]


def _context_match_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str, int, str]:
    return (
        _source_split_phase(row),
        str(row.get("dataset", row.get("dataset_key", ""))),
        _evaluation_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
    )


def _filter_rows_to_contexts(
    rows: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    wanted = {_context_match_key(row) for row in representatives}
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        try:
            key = _context_match_key(row)
        except (KeyError, TypeError, ValueError):
            return [dict(item) for item in rows]
        if key in wanted:
            filtered.append(dict(row))
    return filtered


def _load_student_checkpoint(path: str | Path) -> Tuple[GIPODensityStudentMLP, Dict[str, int], EmbeddingNormalizer, Tuple[float, ...], Dict[str, Any]]:
    payload = torch.load(resolve_project_path(str(path)), map_location="cpu")
    if str(payload.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError(f"Unsupported GIPO student protocol {payload.get('protocol')!r}; expected {GIPO_PROTOCOL!r}.")
    if str(payload.get("student_policy_type", "")) != "continuous_density":
        raise ValueError("GIPO locked reporter only accepts continuous_density student checkpoints.")
    if bool(payload.get("locked_test_used_for_selection", False)):
        raise ValueError("GIPO student checkpoint indicates locked_test was used for selection.")
    density_meta = dict(payload.get("density_representation", {}))
    if str(density_meta.get("density_protocol", "")) != "density_mass_v1":
        raise ValueError("GIPO student checkpoint is missing density_mass_v1 metadata.")
    reference_time_grid = tuple(float(x) for x in density_meta["reference_time_grid"])
    series_index_map = {str(key): int(value) for key, value in dict(payload["series_index_map"]).items()}
    normalizer = EmbeddingNormalizer.from_payload(payload["embedding_normalizer"])
    student = GIPODensityStudentMLP(
        setting_dim=int(payload["setting_dim"]),
        density_dim=int(payload["density_dim"]),
        context_dim=int(payload["context_dim"]),
        num_series=len(series_index_map),
    )
    student.load_state_dict(payload["student_state"])
    student.eval()
    return student, series_index_map, normalizer, reference_time_grid, payload


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]], *, split_phase: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("dataset", row.get("dataset_key", ""))),
                int(row["seed"]),
                str(row["solver_key"]),
                int(row["target_nfe"]),
                str(row["scheduler_key"]),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, seed, solver, target_nfe, scheduler_key), group in sorted(grouped.items()):
        crps = np.asarray([float(row["crps"]) for row in group], dtype=np.float64)
        mase = np.asarray([float(row["mase"]) for row in group], dtype=np.float64)
        row: Dict[str, Any] = {
            "dataset": dataset,
            "split_phase": str(split_phase),
            "seed": int(seed),
            "solver_key": solver,
            "target_nfe": int(target_nfe),
            "scheduler_key": scheduler_key,
            "crps": float(np.mean(crps)),
            "mase": float(np.mean(mase)),
            "context_count": int(len(group)),
            "crps_std": float(np.std(crps)),
            "mase_std": float(np.std(mase)),
        }
        if all(row.get("mse") not in (None, "") for row in group):
            mse = np.asarray([float(row["mse"]) for row in group], dtype=np.float64)
            row["mse"] = float(np.mean(mse))
            row["mse_std"] = float(np.std(mse))
        out.append(row)
    return out


def _forecast_dataset_for_source_phase(splits: Mapping[str, Any], source_phase: str):
    phase = str(source_phase)
    if phase == TRAIN_TUNING_PHASE:
        return splits["train"]
    if phase == VALIDATION_PHASE:
        return splits["val"]
    if phase == LOCKED_TEST_PHASE:
        return splits["test"]
    raise ValueError(f"Unsupported source split_phase {source_phase!r}.")


def _output_prefix(split_phase: str, selection_mode: str, report_label: str = "") -> str:
    if str(split_phase) == LOCKED_TEST_PHASE and str(selection_mode) == SELECTION_MODE_REPORTING:
        return "locked_test_gipo"
    label = str(report_label).strip() or str(split_phase).strip() or "gipo"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label)
    return f"{safe}_gipo"


def report_gipo_locked_test(args: argparse.Namespace) -> Dict[str, Any]:
    student, series_index_map, normalizer, reference_time_grid, checkpoint_payload = _load_student_checkpoint(str(args.gipo_student_checkpoint))
    training_summary = json.loads(resolve_project_path(str(args.training_summary)).read_text(encoding="utf-8"))
    if str(training_summary.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError("training_summary protocol does not match continuous-density GIPO.")
    if bool(training_summary.get("locked_test_used_for_selection", False)):
        raise ValueError("training_summary indicates locked_test was used for selection.")

    selection_mode = str(getattr(args, "selection_mode", SELECTION_MODE_REPORTING))
    split_phase = str(getattr(args, "split_phase", LOCKED_TEST_PHASE))
    context_rows_arg = str(getattr(args, "context_rows", ""))
    embeddings_arg = str(getattr(args, "context_embeddings_npz", ""))
    if not context_rows_arg.strip():
        raise ValueError("GIPO reporter requires --context_rows.")
    if not embeddings_arg.strip():
        raise ValueError("GIPO reporter requires --context_embeddings_npz.")
    context_rows = _read_csvs(context_rows_arg)
    _validate_context_rows(context_rows, split_phase=split_phase, selection_mode=selection_mode)
    representatives = _representative_context_rows(context_rows)
    dataset = str(args.dataset)
    seeds = tuple(_parse_int_csv(str(args.seeds)))
    solvers = tuple(_parse_csv(str(args.solver_names)))
    target_nfes = tuple(_parse_int_csv(str(args.target_nfe_values)))
    expected_cells = {(dataset, seed, solver, target_nfe) for seed in seeds for solver in solvers for target_nfe in target_nfes}
    observed_cells = {
        (str(row.get("dataset", row.get("dataset_key", ""))), int(row["evaluation_seed"]), str(row["solver_key"]), int(row["target_nfe"]))
        for row in representatives
    }
    missing_cells = sorted(expected_cells - observed_cells)
    if split_phase == LOCKED_TEST_PHASE and missing_cells:
        raise ValueError(f"context rows are missing seed/solver/NFE cells: {missing_cells[:8]}")

    raw_embeddings = load_context_embedding_table(resolve_project_path(embeddings_arg))
    missing_context_embeddings = sorted({context_id_from_row(row) for row in representatives} - set(raw_embeddings))
    if missing_context_embeddings:
        raise KeyError(f"context embeddings NPZ is missing contexts: {missing_context_embeddings[:8]}")
    embeddings = normalizer.transform_table(raw_embeddings)

    device = resolve_torch_device(str(args.device))
    student.to(device)
    checkpoint = load_forecast_checkpoint_splits(
        cli_args=args,
        dataset_root=resolve_project_path(str(args.dataset_root)),
        shared_backbone_root=resolve_project_path(str(args.shared_backbone_root)),
        dataset=dataset,
        device=device,
    )
    model = checkpoint["model"]
    cfg = checkpoint["cfg"]
    splits = checkpoint["splits"]

    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = _output_prefix(split_phase, selection_mode, str(getattr(args, "report_label", "")))
    decision_rows: List[Dict[str, Any]] = []
    per_context_rows: List[Dict[str, Any]] = []
    with ProgressBar(len(representatives), f"GIPO {split_phase}") as progress:
        for row_idx, row in enumerate(representatives):
            context_id = context_id_from_row(row)
            source_phase = _source_split_phase(row)
            eval_ds = _forecast_dataset_for_source_phase(splits, source_phase)
            prediction = predict_gipo_density(
                student,
                row=row,
                context_embedding=embeddings[context_id],
                series_index_map=series_index_map,
                reference_time_grid=reference_time_grid,
                device=device,
            )
            example_idx = int(row.get("example_idx", row.get("example_index", 0)))
            solver = str(row["solver_key"])
            target_nfe = int(row["target_nfe"])
            eval_seed = int(row["evaluation_seed"]) + 10_000 * int(row_idx)
            metrics = evaluate_forecast_schedule(
                model,
                eval_ds,
                cfg,
                solver_name=str(SOLVER_RUNTIME_NAMES[solver]),
                runtime_nfe=int(solver_macro_steps(solver, target_nfe)),
                target_nfe=int(target_nfe),
                time_grid=prediction["time_grid"],
                num_eval_samples=int(args.num_eval_samples),
                seed=int(eval_seed),
                scheduler_key=GIPO_SCHEDULE_KEY,
                dataset_key=dataset,
                split_phase=source_phase,
                checkpoint_id=str(checkpoint["checkpoint_id"]),
                example_indices=[example_idx],
                batch_size=int(args.forecast_eval_batch_size),
                progress_label="",
                return_per_example_rows=False,
            )
            selected_row = {
                "dataset": dataset,
                "split_phase": str(split_phase),
                "source_split_phase": source_phase,
                "selection_mode": selection_mode,
                "selection_split": _selection_split(row),
                "seed": int(row["evaluation_seed"]),
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "scheduler_key": GIPO_SCHEDULE_KEY,
                "context_id": context_id,
                "example_idx": int(example_idx),
                "series_id": row.get("series_id", ""),
                "series_idx": row.get("series_idx", ""),
                "target_t": row.get("target_t", ""),
                "crps": float(metrics["crps"]),
                "mase": float(metrics["mase"]),
                "mse": metrics.get("mse", ""),
                "time_grid_json": json.dumps(prediction["time_grid"], separators=(",", ":")),
                "density_mass_hash": prediction["density_mass_hash"],
                "schedule_grid_hash": prediction["schedule_grid_hash"],
                "density_protocol": prediction["density_protocol"],
                "reference_grid_hash": prediction["reference_grid_hash"],
            }
            per_context_rows.append(selected_row)
            decision_rows.append(
                {
                    **selected_row,
                    "density_mass_json": json.dumps(prediction["density_mass"], separators=(",", ":")),
                    "macro_steps": int(prediction["macro_steps"]),
                    "policy_source": "frozen_gipo",
                    "locked_test_used_for_selection": False,
                }
            )
            progress.update()

    aggregate_rows = _aggregate_seed_rows(per_context_rows, split_phase=split_phase)
    _write_csv(out_dir / f"{output_prefix}_rows.csv", per_context_rows)
    _write_csv(out_dir / f"{output_prefix}_aggregate_rows.csv", aggregate_rows)
    _write_csv(out_dir / f"{output_prefix}_decisions.csv", decision_rows)

    baseline_rows = _read_csvs(str(args.baseline_rows)) if str(args.baseline_rows).strip() else []
    comparator_rows = _read_csvs(str(args.comparator_rows)) if str(args.comparator_rows).strip() else []
    if selection_mode == SELECTION_MODE_CALIBRATION:
        baseline_rows = _filter_rows_to_contexts(baseline_rows, representatives)
        comparator_rows = _filter_rows_to_contexts(comparator_rows, representatives)
    comparison = None
    if baseline_rows:
        comparison_student_rows = per_context_rows if selection_mode == SELECTION_MODE_CALIBRATION else aggregate_rows
        comparison = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
            student_rows=comparison_student_rows,
            dataset=dataset,
            split_phase=split_phase,
            seeds=tuple(sorted({int(row["seed"]) for row in comparison_student_rows})) or seeds,
            solver_names=tuple(sorted({str(row["solver_key"]) for row in comparison_student_rows})) or solvers,
            target_nfe_values=tuple(sorted({int(row["target_nfe"]) for row in comparison_student_rows})) or target_nfes,
        )
        (out_dir / f"{output_prefix}_comparison_summary.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    crps_values = [float(row["crps"]) for row in aggregate_rows]
    mase_values = [float(row["mase"]) for row in aggregate_rows]
    artifact_name = (
        "gipo_locked_test_report"
        if split_phase == LOCKED_TEST_PHASE and selection_mode == SELECTION_MODE_REPORTING
        else "gipo_student_calibration_report"
    )
    summary: Dict[str, Any] = {
        "artifact": artifact_name,
        "protocol": GIPO_PROTOCOL,
        "student_policy_type": "continuous_density",
        "scheduler_key": GIPO_SCHEDULE_KEY,
        "dataset": dataset,
        "split_phase": split_phase,
        "selection_mode": selection_mode,
        "source_split_phases": sorted({_source_split_phase(row) for row in representatives}),
        "context_row_count": int(len(per_context_rows)),
        "aggregate_row_count": int(len(aggregate_rows)),
        "missing_expected_cells": missing_cells,
        "mean_crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))) if crps_values else None,
        "mean_mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))) if mase_values else None,
        "density_representation": checkpoint_payload.get("density_representation", {}),
        "locked_test_used_for_selection": False,
        "comparison_summary_path": "" if comparison is None else str(out_dir / f"{output_prefix}_comparison_summary.json"),
    }
    (out_dir / f"{output_prefix}_policy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report locked-test performance for a frozen GIPO student.")
    parser.add_argument("--gipo_student_checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--context_rows", default="", help="Comma-separated context-row CSVs used to enumerate report contexts.")
    parser.add_argument("--context_embeddings_npz", default="", help="Context embedding table for --context_rows.")
    parser.add_argument("--split_phase", default=LOCKED_TEST_PHASE, help="Report split label: locked_test, validation_tuning, train_tuning, context_disjoint, or series_disjoint.")
    parser.add_argument("--selection_mode", choices=(SELECTION_MODE_REPORTING, SELECTION_MODE_CALIBRATION), default=SELECTION_MODE_REPORTING)
    parser.add_argument("--report_label", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--baseline_rows", default="")
    parser.add_argument("--comparator_rows", default="")
    parser.add_argument("--dataset", default="solar_energy_10m")
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(default_backbone_manifest_path()))
    parser.add_argument("--output_root", default=str(project_outputs_root()))
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = report_gipo_locked_test(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
