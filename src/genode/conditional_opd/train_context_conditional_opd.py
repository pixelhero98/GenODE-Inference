from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from genode.conditional_opd.context_conditional import (
    DEFAULT_SUPPORT_SCHEDULE_KEYS,
    DEFAULT_SUPPORT_CHOICE_MARGIN,
    DEFAULT_TEACHER_SELECTION_MIN_PAIRWISE_ACCURACY,
    DEFAULT_TEACHER_SELECTION_MIN_SPEARMAN,
    ContextScheduleTeacherMLP,
    ContextSupportStudentMLP,
    EmbeddingNormalizer,
    attach_uniform_context_rewards,
    build_calibration_holdout_non_regression_guard,
    build_series_index_map,
    context_id_from_row,
    context_teacher_diagnostics,
    load_context_embedding_table,
    read_metric_rows_csv,
    recommended_context_calibration_count,
    sample_context_ids_stratified,
    split_rows_by_context_holdout,
    split_rows_by_series_holdout,
    series_key_from_row,
    train_context_support_student,
    train_context_teacher,
    validate_context_support_schedule_keys,
)
from genode.conditional_opd.models import setting_features, solver_macro_steps, validate_time_grid
from genode.data.otflow_paths import resolve_project_path
from genode.models.otflow_train_val import seed_all


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _load_schedule_summary_grids(paths: Sequence[str]) -> Dict[Tuple[str, str, int], Tuple[float, ...]]:
    grids: Dict[Tuple[str, str, int], Tuple[float, ...]] = {}
    for path_text in paths:
        path = resolve_project_path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Schedule summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        schedules = payload.get("schedules")
        if schedules:
            schedule_items = list(schedules)
        else:
            schedule_items = [
                {
                    "scheduler_key": str(payload.get("scheduler_key", payload.get("schedule_key", ""))),
                    "predictions": payload.get("predictions", []) or [],
                }
            ]
        for schedule in schedule_items:
            schedule_key = str(schedule.get("scheduler_key", schedule.get("schedule_key", ""))).strip()
            for item in list(schedule.get("predictions", []) or []):
                solver = str(item["solver_key"])
                target_nfe = int(item["target_nfe"])
                macro_steps = solver_macro_steps(solver, target_nfe)
                grids[(schedule_key, solver, target_nfe)] = validate_time_grid(item["time_grid"], macro_steps=macro_steps)
    return grids


def _observed_support(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    observed = tuple(sorted({str(row["scheduler_key"]) for row in rows}))
    canonical_order = tuple(key for key in DEFAULT_SUPPORT_SCHEDULE_KEYS if key in observed)
    extras = tuple(key for key in observed if key not in canonical_order)
    return validate_context_support_schedule_keys(canonical_order + extras)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train context-conditional OPD from per-example fixed/SER rows.")
    parser.add_argument("--rows_csv", required=True, help="Per-example fixed/SER metric rows CSV.")
    parser.add_argument("--context_embeddings_npz", required=True, help="Context embedding sidecar NPZ.")
    parser.add_argument("--schedule_summary_json", default="", help="Comma-separated schedule summaries for non-fixed support such as SER.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--support_schedule_keys", default="", help="Comma-separated fixed/SER support keys. Defaults to observed row keys.")
    parser.add_argument("--context_sample_count", type=int, default=0, help="0 uses the capped recommendation; pass an explicit count such as 288 for larger reusable calibration pools.")
    parser.add_argument("--context_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--series_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--holdout_fraction", type=float, default=None, help="Deprecated alias for --context_holdout_fraction.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_macro_steps", type=int, default=12)
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--teacher_checkpoint_every", type=int, default=100)
    parser.add_argument("--student_steps", type=int, default=500)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--support_choice_margin", type=float, default=DEFAULT_SUPPORT_CHOICE_MARGIN)
    parser.add_argument("--teacher_selection_min_pairwise_accuracy", type=float, default=DEFAULT_TEACHER_SELECTION_MIN_PAIRWISE_ACCURACY)
    parser.add_argument("--teacher_selection_min_spearman", type=float, default=DEFAULT_TEACHER_SELECTION_MIN_SPEARMAN)
    parser.add_argument("--series_unknown_dropout", type=float, default=0.10)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "row_count": int(len(rows)),
        "context_count": int(len({context_id_from_row(row) for row in rows})),
        "series_count": int(len({series_key_from_row(row) for row in rows})),
        "schedule_count": int(len({str(row["scheduler_key"]) for row in rows})),
    }


def _with_calibration_holdout_name(rows: Sequence[Mapping[str, Any]], holdout_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["calibration_holdout_name"] = str(holdout_name)
        out.append(copied)
    return out


def train_context_conditional_opd(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    rows = read_metric_rows_csv(resolve_project_path(str(args.rows_csv)))
    if not rows:
        raise ValueError("rows_csv contains no rows.")
    locked_rows = [row for row in rows if str(row.get("split_phase", row.get("split", ""))) == "locked_test"]
    if locked_rows:
        raise ValueError(f"Context training refuses locked_test rows in rows_csv; found {len(locked_rows)} locked-test rows.")
    support_keys = (
        validate_context_support_schedule_keys(_parse_csv(str(args.support_schedule_keys)))
        if str(args.support_schedule_keys).strip()
        else _observed_support(rows)
    )
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    missing_support_rows = sorted(set(support_keys) - observed_keys)
    if missing_support_rows:
        raise ValueError(f"Support schedules must have measured context rows; missing rows for {missing_support_rows}")
    rewarded_rows = attach_uniform_context_rewards(rows, support_schedule_keys=support_keys, pair_on_seed=True)
    available_context_ids = sorted({context_id_from_row(row) for row in rewarded_rows})
    sample_count = int(args.context_sample_count)
    if sample_count <= 0:
        sample_count = recommended_context_calibration_count(len(available_context_ids))
    selected_context_ids = set(sample_context_ids_stratified(rewarded_rows, sample_count=sample_count, seed=int(args.seed)))
    sampled_rows = [row for row in rewarded_rows if context_id_from_row(row) in selected_context_ids]
    context_holdout_fraction = float(args.context_holdout_fraction)
    if getattr(args, "holdout_fraction", None) is not None:
        context_holdout_fraction = float(args.holdout_fraction)
    context_fit_pool_rows, context_holdout_rows = split_rows_by_context_holdout(
        sampled_rows,
        holdout_fraction=context_holdout_fraction,
        seed=int(args.seed),
    )
    fit_rows, series_holdout_rows = split_rows_by_series_holdout(
        context_fit_pool_rows,
        holdout_fraction=float(args.series_holdout_fraction),
        seed=int(args.seed),
    )
    if not fit_rows:
        raise ValueError("Teacher fitting requires at least one row after context and series holdouts.")
    context_embeddings = load_context_embedding_table(resolve_project_path(str(args.context_embeddings_npz)))
    fit_context_ids = sorted({context_id_from_row(row) for row in fit_rows})
    normalizer = EmbeddingNormalizer.fit(context_embeddings, fit_context_ids)
    normalized_embeddings = normalizer.transform_table(context_embeddings)
    series_index_map = build_series_index_map(fit_rows)
    context_dim = int(next(iter(normalized_embeddings.values())).shape[0])
    setting_dim = int(setting_features("euler", 4).numel())
    teacher = ContextScheduleTeacherMLP(
        setting_dim=setting_dim,
        max_macro_steps=int(args.max_macro_steps),
        context_dim=context_dim,
        num_series=len(series_index_map),
    )
    student = ContextSupportStudentMLP(
        setting_dim=setting_dim,
        context_dim=context_dim,
        num_series=len(series_index_map),
        support_schedule_keys=support_keys,
    )
    schedule_grids = _load_schedule_summary_grids(_parse_csv(str(args.schedule_summary_json)))

    out_dir = resolve_project_path(str(args.out_dir))
    if not bool(args.dry_run):
        out_dir.mkdir(parents=True, exist_ok=True)
        teacher_training_raw = train_context_teacher(
            teacher,
            fit_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            max_macro_steps=int(args.max_macro_steps),
            steps=int(args.teacher_steps),
            lr=float(args.teacher_lr),
            pair_on_seed=True,
            diagnostic_splits={
                "context_disjoint": context_holdout_rows,
                "series_disjoint": series_holdout_rows,
            },
            teacher_checkpoint_every=int(args.teacher_checkpoint_every),
            teacher_selection_min_pairwise_accuracy=float(args.teacher_selection_min_pairwise_accuracy),
            teacher_selection_min_spearman=float(args.teacher_selection_min_spearman),
            series_unknown_probability=float(args.series_unknown_dropout),
            seed=int(args.seed),
        )
        selected_teacher_state = teacher_training_raw.pop("_selected_state_dict", None)
        if selected_teacher_state is not None:
            teacher.load_state_dict(selected_teacher_state)
        teacher_training = teacher_training_raw
        student_training = train_context_support_student(
            student,
            teacher,
            fit_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            support_schedule_keys=support_keys,
            schedule_grids=schedule_grids,
            max_macro_steps=int(args.max_macro_steps),
            steps=int(args.student_steps),
            lr=float(args.student_lr),
            support_choice_margin=float(args.support_choice_margin),
            series_unknown_probability=float(args.series_unknown_dropout),
            seed=int(args.seed),
        )
        selected_step = teacher_training.get("checkpoint_selection", {}).get("selected_step")
        guard_rows = _with_calibration_holdout_name(
            context_holdout_rows,
            "context_disjoint",
        ) + _with_calibration_holdout_name(
            series_holdout_rows,
            "series_disjoint",
        )
        calibration_guard = build_calibration_holdout_non_regression_guard(
            student,
            guard_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            support_schedule_keys=support_keys,
            margin=float(args.support_choice_margin),
            source_holdout_names=("context_disjoint", "series_disjoint"),
        )
        policy_payload = {
            "protocol": "context_conditional_opd_v1",
            "rows_csv": str(resolve_project_path(str(args.rows_csv))),
            "context_embeddings_npz": str(resolve_project_path(str(args.context_embeddings_npz))),
            "support_schedule_keys": list(support_keys),
            "seed": int(args.seed),
            "teacher_selected_step": selected_step,
            "student_objective": "teacher_guided_top1_top2_categorical_ce",
            "calibration_guard_id": calibration_guard.get("guard_id"),
            "context_holdout_fraction": float(context_holdout_fraction),
            "series_holdout_fraction": float(args.series_holdout_fraction),
            "series_unknown_dropout": float(args.series_unknown_dropout),
            "support_choice_margin": float(args.support_choice_margin),
        }
        policy_id = "ctx_policy_" + hashlib.sha256(json.dumps(policy_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        torch.save(
            {
                "policy_id": policy_id,
                "state_dict": teacher.state_dict(),
                "series_index_map": series_index_map,
                "context_dim": context_dim,
                "max_macro_steps": int(args.max_macro_steps),
                "support_schedule_keys": list(support_keys),
                "embedding_normalizer": {
                    "mean": [float(value) for value in normalizer.mean.tolist()],
                    "std": [float(value) for value in normalizer.std.tolist()],
                },
                "teacher_checkpoint_selection": teacher_training.get("checkpoint_selection", {}),
                "calibration_holdout_non_regression_guard": calibration_guard,
            },
            out_dir / "context_teacher.pt",
        )
        torch.save(
            {
                "policy_id": policy_id,
                "state_dict": student.state_dict(),
                "series_index_map": series_index_map,
                "context_dim": context_dim,
                "support_schedule_keys": list(support_keys),
                "embedding_normalizer": {
                    "mean": [float(value) for value in normalizer.mean.tolist()],
                    "std": [float(value) for value in normalizer.std.tolist()],
                },
                "teacher_checkpoint_selection": teacher_training.get("checkpoint_selection", {}),
                "calibration_holdout_non_regression_guard": calibration_guard,
            },
            out_dir / "context_student.pt",
        )
    else:
        policy_id = "dry_run"
        context_diag = context_teacher_diagnostics(
            teacher,
            context_holdout_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            max_macro_steps=int(args.max_macro_steps),
            split_name="context_disjoint",
            fit_context_ids=fit_context_ids,
            fit_series_keys=sorted(series_index_map),
        ) if context_holdout_rows else {}
        series_diag = context_teacher_diagnostics(
            teacher,
            series_holdout_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            max_macro_steps=int(args.max_macro_steps),
            split_name="series_disjoint",
            fit_context_ids=fit_context_ids,
            fit_series_keys=sorted(series_index_map),
        ) if series_holdout_rows else {}
        teacher_training = {
            "losses": [],
            "teacher_pair_count": 0,
            "dry_run": True,
            "checkpoint_selection": {
                "selection_protocol": "context_series_support_choice_teacher_checkpoint",
                "selection_split": "context_and_series_teacher_holdouts",
                "selected_step": None,
                "selection_metric": "context_series_support_top1_top2",
                "selection_constraints": {
                    "min_pairwise_accuracy": float(args.teacher_selection_min_pairwise_accuracy),
                    "min_spearman": float(args.teacher_selection_min_spearman),
                },
                "history": [],
                "uses_validation_labels": False,
            },
            "teacher_context_holdout_diagnostics": context_diag,
            "teacher_series_holdout_diagnostics": series_diag,
        }
        student_training = {"losses": [], "dry_run": True}
        calibration_guard = {
            "artifact": "calibration_holdout_non_regression_guard",
            "enabled": False,
            "dry_run": True,
            "locked_test_used_for_selection": False,
            "cell_decisions": [],
            "cell_decision_map": {},
        }

    teacher_checkpoint_selection = dict(teacher_training.get("checkpoint_selection", {}))
    selected_history = list(teacher_checkpoint_selection.get("history", []) or [])
    selected_step = teacher_checkpoint_selection.get("selected_step")
    selected_entry = next((entry for entry in selected_history if int(entry.get("step", -1)) == int(selected_step or -1)), None)
    selected_diagnostics = dict(selected_entry.get("diagnostics", {})) if selected_entry else {}
    context_holdout_diagnostics = selected_diagnostics.get("context_disjoint", teacher_training.get("teacher_context_holdout_diagnostics", {}))
    series_holdout_diagnostics = selected_diagnostics.get("series_disjoint", teacher_training.get("teacher_series_holdout_diagnostics", {}))
    fit_contexts = {context_id_from_row(row) for row in fit_rows}
    context_holdout_contexts = {context_id_from_row(row) for row in context_holdout_rows}
    series_holdout_contexts = {context_id_from_row(row) for row in series_holdout_rows}
    fit_series = {series_key_from_row(row) for row in fit_rows}
    context_holdout_series = {series_key_from_row(row) for row in context_holdout_rows}
    series_holdout_series = {series_key_from_row(row) for row in series_holdout_rows}
    summary = {
        "artifact": "context_conditional_opd_training_summary",
        "protocol": "context_conditional_opd_v1",
        "rows_csv": str(resolve_project_path(str(args.rows_csv))),
        "context_embeddings_npz": str(resolve_project_path(str(args.context_embeddings_npz))),
        "support_schedule_keys": list(support_keys),
        "available_context_count": int(len(available_context_ids)),
        "sampled_context_count": int(len(selected_context_ids)),
        "recommended_default_context_count": int(recommended_context_calibration_count(len(available_context_ids))),
        "context_holdout_fraction": float(context_holdout_fraction),
        "series_holdout_fraction": float(args.series_holdout_fraction),
        "fit_context_count": int(len(fit_contexts)),
        "holdout_context_count": int(len(context_holdout_contexts)),
        "fit_row_count": int(len(fit_rows)),
        "holdout_row_count": int(len(context_holdout_rows)),
        "teacher_fit": _split_counts(fit_rows),
        "teacher_context_holdout": _split_counts(context_holdout_rows),
        "teacher_series_holdout": _split_counts(series_holdout_rows),
        "teacher_context_holdout_context_overlap_count": int(len(fit_contexts & context_holdout_contexts)),
        "teacher_series_holdout_series_overlap_count": int(len(fit_series & series_holdout_series)),
        "teacher_context_holdout_series_overlap_count": int(len(fit_series & context_holdout_series)),
        "teacher_training": teacher_training,
        "teacher_selection_protocol": "context_and_series_disjoint_train_holdout_teacher_checkpoint",
        "policy_id": policy_id,
        "context_teacher_checkpoint": str(out_dir / "context_teacher.pt"),
        "context_student_checkpoint": str(out_dir / "context_student.pt"),
        "teacher_checkpoint_selection": teacher_checkpoint_selection,
        "teacher_context_holdout_diagnostics": context_holdout_diagnostics,
        "teacher_series_holdout_diagnostics": series_holdout_diagnostics,
        "student_training": student_training,
        "student_policy_type": "categorical_support_fixed_ser",
        "student_objective": "teacher_guided_top1_top2_categorical_ce",
        "reward_anchor_schedule_key": "uniform",
        "support_choice_margin": float(args.support_choice_margin),
        "calibration_holdout_non_regression_guard": calibration_guard,
        "series_unknown_dropout": float(args.series_unknown_dropout),
        "series_unknown_dropout_mode": "dynamic_per_step_for_student",
        "uses_bo": False,
        "locked_test_used_for_selection": False,
    }
    if not bool(args.dry_run):
        (out_dir / "context_conditional_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = train_context_conditional_opd(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
