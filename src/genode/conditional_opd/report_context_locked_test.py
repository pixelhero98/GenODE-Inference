from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.conditional_opd.context_conditional import (
    CONTEXT_CONDITIONAL_PROTOCOL,
    ContextDensityStudentMLP,
    EmbeddingNormalizer,
    context_id_from_row,
    load_context_embedding_table,
    predict_context_density,
    read_metric_rows_csv,
    remapped_series_index,
)
from genode.conditional_opd.evaluate_schedule_summary import build_comparison_summary
from genode.conditional_opd.models import solver_macro_steps
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
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
)
from genode.runtime import ProgressBar, resolve_torch_device

CONTEXT_DENSITY_STUDENT_SCHEDULE_KEY = "context_density_student"


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


def _validate_locked_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("locked_context_rows contains no rows.")
    bad = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows if str(row.get("split_phase", row.get("split", ""))) != LOCKED_TEST_PHASE})
    if bad:
        raise ValueError(f"Context density locked reporter only accepts split_phase={LOCKED_TEST_PHASE!r}; found {bad}.")


def _evaluation_seed_from_row(row: Mapping[str, Any]) -> int:
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return int(row["seed"])


def _row_group_key(row: Mapping[str, Any]) -> Tuple[str, int, str, int, str]:
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        _evaluation_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
    )


def _representative_locked_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, int, str], Dict[str, Any]] = {}
    for row in rows:
        key = _row_group_key(row)
        if key not in grouped:
            copied = dict(row)
            copied["context_id"] = context_id_from_row(row)
            copied["evaluation_seed"] = _evaluation_seed_from_row(row)
            grouped[key] = copied
    return [grouped[key] for key in sorted(grouped)]


def _load_student_checkpoint(path: str | Path) -> Tuple[ContextDensityStudentMLP, Dict[str, int], EmbeddingNormalizer, Tuple[float, ...], Dict[str, Any]]:
    payload = torch.load(resolve_project_path(str(path)), map_location="cpu")
    if str(payload.get("protocol", "")) != CONTEXT_CONDITIONAL_PROTOCOL:
        raise ValueError(f"Unsupported context student protocol {payload.get('protocol')!r}; expected {CONTEXT_CONDITIONAL_PROTOCOL!r}.")
    if str(payload.get("student_policy_type", "")) != "continuous_density":
        raise ValueError("Context locked reporter only accepts continuous_density student checkpoints.")
    if bool(payload.get("locked_test_used_for_selection", False)):
        raise ValueError("Context student checkpoint indicates locked_test was used for selection.")
    density_meta = dict(payload.get("density_representation", {}))
    if str(density_meta.get("density_protocol", "")) != "density_mass_v1":
        raise ValueError("Context student checkpoint is missing density_mass_v1 metadata.")
    reference_time_grid = tuple(float(x) for x in density_meta["reference_time_grid"])
    series_index_map = {str(key): int(value) for key, value in dict(payload["series_index_map"]).items()}
    normalizer = EmbeddingNormalizer.from_payload(payload["embedding_normalizer"])
    student = ContextDensityStudentMLP(
        setting_dim=int(payload["setting_dim"]),
        density_dim=int(payload["density_dim"]),
        context_dim=int(payload["context_dim"]),
        num_series=len(series_index_map),
    )
    student.load_state_dict(payload["student_state"])
    student.eval()
    return student, series_index_map, normalizer, reference_time_grid, payload


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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
            "split_phase": LOCKED_TEST_PHASE,
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


def report_context_locked_test(args: argparse.Namespace) -> Dict[str, Any]:
    student, series_index_map, normalizer, reference_time_grid, checkpoint_payload = _load_student_checkpoint(str(args.context_student_checkpoint))
    training_summary = json.loads(resolve_project_path(str(args.training_summary)).read_text(encoding="utf-8"))
    if str(training_summary.get("protocol", "")) != CONTEXT_CONDITIONAL_PROTOCOL:
        raise ValueError("training_summary protocol does not match continuous density context OPD.")
    if bool(training_summary.get("locked_test_used_for_selection", False)):
        raise ValueError("training_summary indicates locked_test was used for selection.")

    locked_rows = _read_csvs(str(args.locked_context_rows))
    _validate_locked_rows(locked_rows)
    representatives = _representative_locked_rows(locked_rows)
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
    if missing_cells:
        raise ValueError(f"locked_context_rows are missing seed/solver/NFE cells: {missing_cells[:8]}")

    raw_embeddings = load_context_embedding_table(resolve_project_path(str(args.locked_context_embeddings_npz)))
    missing_context_embeddings = sorted({context_id_from_row(row) for row in representatives} - set(raw_embeddings))
    if missing_context_embeddings:
        raise KeyError(f"locked_context_embeddings_npz is missing contexts: {missing_context_embeddings[:8]}")
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
    eval_ds = checkpoint["splits"]["test"]

    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_rows: List[Dict[str, Any]] = []
    per_context_rows: List[Dict[str, Any]] = []
    with ProgressBar(len(representatives), "context density locked-test") as progress:
        for row_idx, row in enumerate(representatives):
            context_id = context_id_from_row(row)
            prediction = predict_context_density(
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
                scheduler_key=CONTEXT_DENSITY_STUDENT_SCHEDULE_KEY,
                dataset_key=dataset,
                split_phase=LOCKED_TEST_PHASE,
                checkpoint_id=str(checkpoint["checkpoint_id"]),
                example_indices=[example_idx],
                batch_size=int(args.forecast_eval_batch_size),
                progress_label="",
                return_per_example_rows=False,
            )
            selected_row = {
                "dataset": dataset,
                "split_phase": LOCKED_TEST_PHASE,
                "seed": int(row["evaluation_seed"]),
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "scheduler_key": CONTEXT_DENSITY_STUDENT_SCHEDULE_KEY,
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
                    "policy_source": "frozen_context_density_student",
                    "locked_test_used_for_selection": False,
                }
            )
            progress.update()

    aggregate_rows = _aggregate_seed_rows(per_context_rows)
    _write_csv(out_dir / "locked_test_context_density_rows.csv", per_context_rows)
    _write_csv(out_dir / "locked_test_context_density_aggregate_rows.csv", aggregate_rows)
    _write_csv(out_dir / "locked_test_context_density_decisions.csv", decision_rows)

    baseline_rows = _read_csvs(str(args.baseline_rows)) if str(args.baseline_rows).strip() else []
    comparator_rows = _read_csvs(str(args.comparator_rows)) if str(args.comparator_rows).strip() else []
    comparison = None
    if baseline_rows:
        comparison = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
            student_rows=aggregate_rows,
            dataset=dataset,
            split_phase=LOCKED_TEST_PHASE,
            seeds=seeds,
            solver_names=solvers,
            target_nfe_values=target_nfes,
        )
        (out_dir / "locked_test_context_density_comparison_summary.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    crps_values = [float(row["crps"]) for row in aggregate_rows]
    mase_values = [float(row["mase"]) for row in aggregate_rows]
    summary: Dict[str, Any] = {
        "artifact": "context_density_locked_test_report",
        "protocol": CONTEXT_CONDITIONAL_PROTOCOL,
        "student_policy_type": "continuous_density",
        "scheduler_key": CONTEXT_DENSITY_STUDENT_SCHEDULE_KEY,
        "dataset": dataset,
        "split_phase": LOCKED_TEST_PHASE,
        "context_row_count": int(len(per_context_rows)),
        "aggregate_row_count": int(len(aggregate_rows)),
        "mean_crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))) if crps_values else None,
        "mean_mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))) if mase_values else None,
        "density_representation": checkpoint_payload.get("density_representation", {}),
        "locked_test_used_for_selection": False,
        "comparison_summary_path": "" if comparison is None else str(out_dir / "locked_test_context_density_comparison_summary.json"),
    }
    (out_dir / "locked_test_context_density_policy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report locked-test performance for a frozen context-density OPD student.")
    parser.add_argument("--context_student_checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--locked_context_rows", required=True, help="Comma-separated locked-test context-row CSVs used only to enumerate contexts.")
    parser.add_argument("--locked_context_embeddings_npz", required=True)
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
    summary = report_context_locked_test(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
