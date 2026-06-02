from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from genode.conditional_opd.context_conditional import (
    CONTEXT_CONDITIONAL_PROTOCOL,
    DEFAULT_DENSITY_BIN_COUNT,
    DEFAULT_SUPPORT_SCHEDULE_KEYS,
    DEFAULT_TEACHER_TARGET_TEMPERATURE,
    ContextDensityStudentMLP,
    ContextScheduleTeacherMLP,
    DensityFeatureNormalizer,
    EmbeddingNormalizer,
    attach_uniform_context_rewards,
    build_series_index_map,
    context_id_from_row,
    density_mass_for_row,
    load_context_embedding_table,
    read_metric_rows_csv,
    recommended_context_calibration_count,
    sample_context_ids_stratified,
    split_rows_by_context_holdout,
    split_rows_by_series_holdout,
    train_context_density_student,
    train_context_teacher,
    validate_context_support_schedule_keys,
)
from genode.conditional_opd.density_representation import density_metadata, reference_grid_hash, uniform_reference_grid
from genode.conditional_opd.models import setting_features, solver_macro_steps, validate_time_grid
from genode.data.otflow_paths import resolve_project_path
from genode.models.otflow_train_val import seed_all
from genode.runtime import resolve_torch_device


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


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    from genode.conditional_opd.context_conditional import series_key_from_row

    return {
        "row_count": int(len(rows)),
        "context_count": int(len({context_id_from_row(row) for row in rows})),
        "series_count": int(len({series_key_from_row(row) for row in rows})),
        "schedule_count": int(len({str(row["scheduler_key"]) for row in rows})),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train context-conditional continuous-density OPD from per-example fixed/SER rows.")
    parser.add_argument("--rows_csv", required=True, help="Per-example fixed/SER metric rows CSV.")
    parser.add_argument("--context_embeddings_npz", required=True, help="Frozen context embedding sidecar NPZ.")
    parser.add_argument("--schedule_summary_json", default="", help="Comma-separated schedule summaries for non-fixed references such as SER.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--support_schedule_keys", default="", help="Comma-separated fixed/SER supervision keys. Defaults to observed row keys.")
    parser.add_argument("--context_sample_count", type=int, default=0)
    parser.add_argument("--context_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--series_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--density_bin_count", type=int, default=DEFAULT_DENSITY_BIN_COUNT)
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--teacher_checkpoint_every", type=int, default=100)
    parser.add_argument("--student_steps", type=int, default=500)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--teacher_temperature", type=float, default=DEFAULT_TEACHER_TARGET_TEMPERATURE)
    parser.add_argument("--teacher_rank_temperature", type=float, default=0.5)
    parser.add_argument("--teacher_regression_weight", type=float, default=0.25)
    parser.add_argument("--teacher_pair_margin", type=float, default=0.0)
    parser.add_argument("--series_unknown_dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def train_context_conditional_opd(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    rows = read_metric_rows_csv(resolve_project_path(str(args.rows_csv)))
    if not rows:
        raise ValueError("rows_csv contains no rows.")
    locked_rows = [row for row in rows if str(row.get("split_phase", row.get("split", ""))) == "locked_test"]
    if locked_rows:
        raise ValueError(f"Context density training refuses locked_test rows in rows_csv; found {len(locked_rows)} locked-test rows.")
    support_keys = (
        validate_context_support_schedule_keys(_parse_csv(str(args.support_schedule_keys)))
        if str(args.support_schedule_keys).strip()
        else _observed_support(rows)
    )
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    missing_support_rows = sorted(set(support_keys) - observed_keys)
    if missing_support_rows:
        raise ValueError(f"Supervision schedules must have measured context rows; missing rows for {missing_support_rows}")

    rewarded_rows = attach_uniform_context_rewards(rows, support_schedule_keys=support_keys, pair_on_seed=True)
    available_context_ids = sorted({context_id_from_row(row) for row in rewarded_rows})
    sample_count = int(args.context_sample_count)
    if sample_count <= 0:
        sample_count = recommended_context_calibration_count(len(available_context_ids))
    selected_context_ids = set(sample_context_ids_stratified(rewarded_rows, sample_count=sample_count, seed=int(args.seed)))
    sampled_rows = [row for row in rewarded_rows if context_id_from_row(row) in selected_context_ids]

    context_fit_pool_rows, context_holdout_rows = split_rows_by_context_holdout(
        sampled_rows,
        holdout_fraction=float(args.context_holdout_fraction),
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
    embedding_normalizer = EmbeddingNormalizer.fit(context_embeddings, fit_context_ids)
    normalized_embeddings = embedding_normalizer.transform_table(context_embeddings)
    missing_embeddings = sorted({context_id_from_row(row) for row in sampled_rows} - set(normalized_embeddings))
    if missing_embeddings:
        raise KeyError(f"Context embeddings are missing sampled contexts: {missing_embeddings[:8]}")

    series_index_map = build_series_index_map(fit_rows)
    context_dim = int(next(iter(normalized_embeddings.values())).shape[0])
    setting_dim = int(setting_features("euler", 4).numel())
    reference_time_grid = uniform_reference_grid(int(args.density_bin_count))
    schedule_grids = _load_schedule_summary_grids(_parse_csv(str(args.schedule_summary_json)))
    density_normalizer = DensityFeatureNormalizer.fit(
        (
            density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
            for row in fit_rows
        ),
        reference_time_grid=reference_time_grid,
    )

    density_meta = density_metadata(reference_time_grid)
    teacher = ContextScheduleTeacherMLP(
        setting_dim=setting_dim,
        density_dim=int(len(reference_time_grid) - 1),
        context_dim=context_dim,
        num_series=len(series_index_map),
    )
    student = ContextDensityStudentMLP(
        setting_dim=setting_dim,
        density_dim=int(len(reference_time_grid) - 1),
        context_dim=context_dim,
        num_series=len(series_index_map),
    )

    summary_base: Dict[str, Any] = {
        "artifact": "context_density_opd_training_summary",
        "protocol": CONTEXT_CONDITIONAL_PROTOCOL,
        "student_policy_type": "continuous_density",
        "student_objective": "teacher_weighted_density_mle_kl",
        "teacher_objective": "pairwise_rank_plus_huber_regression",
        "density_representation": density_meta,
        "support_schedule_keys": list(support_keys),
        "sampled_context_count": int(len(selected_context_ids)),
        "split_counts": {
            "fit": _split_counts(fit_rows),
            "context_disjoint": _split_counts(context_holdout_rows),
            "series_disjoint": _split_counts(series_holdout_rows),
        },
        "locked_test_used_for_selection": False,
    }

    out_dir = resolve_project_path(str(args.out_dir))
    if bool(args.dry_run):
        return {**summary_base, "status": "dry_run"}

    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_splits = {"context_disjoint": context_holdout_rows}
    if series_holdout_rows:
        diagnostic_splits["series_disjoint"] = series_holdout_rows
    device = resolve_torch_device(str(args.device))
    teacher_training = train_context_teacher(
        teacher,
        fit_rows,
        context_embeddings=normalized_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        steps=int(args.teacher_steps),
        lr=float(args.teacher_lr),
        rank_temperature=float(args.teacher_rank_temperature),
        regression_weight=float(args.teacher_regression_weight),
        pair_margin=float(args.teacher_pair_margin),
        diagnostic_splits=diagnostic_splits,
        teacher_checkpoint_every=int(args.teacher_checkpoint_every),
        series_unknown_probability=float(args.series_unknown_dropout),
        seed=int(args.seed),
        allowed_schedule_keys=support_keys,
        device=device,
    )
    student_training = train_context_density_student(
        student,
        teacher,
        fit_rows,
        context_embeddings=normalized_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        steps=int(args.student_steps),
        lr=float(args.student_lr),
        teacher_temperature=float(args.teacher_temperature),
        series_unknown_dropout=float(args.series_unknown_dropout),
        device=device,
    )

    teacher_path = out_dir / "context_density_teacher.pt"
    student_path = out_dir / "context_density_student.pt"
    torch.save(
        {
            "protocol": CONTEXT_CONDITIONAL_PROTOCOL,
            "teacher_state": teacher.state_dict(),
            "setting_dim": int(setting_dim),
            "density_dim": int(len(reference_time_grid) - 1),
            "context_dim": int(context_dim),
            "series_index_map": dict(series_index_map),
            "embedding_normalizer": embedding_normalizer.to_payload(),
            "density_feature_normalizer": density_normalizer.to_payload(),
            "density_representation": density_meta,
            "support_schedule_keys": list(support_keys),
            "teacher_training": teacher_training,
            "locked_test_used_for_selection": False,
        },
        teacher_path,
    )
    torch.save(
        {
            "protocol": CONTEXT_CONDITIONAL_PROTOCOL,
            "student_policy_type": "continuous_density",
            "student_objective": "teacher_weighted_density_mle_kl",
            "student_state": student.state_dict(),
            "setting_dim": int(setting_dim),
            "density_dim": int(len(reference_time_grid) - 1),
            "context_dim": int(context_dim),
            "series_index_map": dict(series_index_map),
            "embedding_normalizer": embedding_normalizer.to_payload(),
            "density_feature_normalizer": density_normalizer.to_payload(),
            "density_representation": density_meta,
            "support_schedule_keys": list(support_keys),
            "teacher_checkpoint": str(teacher_path),
            "teacher_training": teacher_training,
            "student_training": student_training,
            "locked_test_used_for_selection": False,
        },
        student_path,
    )

    policy_id_payload = {
        "protocol": CONTEXT_CONDITIONAL_PROTOCOL,
        "student_path": str(student_path),
        "reference_grid_hash": reference_grid_hash(reference_time_grid),
        "support_schedule_keys": list(support_keys),
        "teacher_selected_step": teacher_training.get("teacher_checkpoint_selection", {}).get("selected_step"),
    }
    policy_id = "ctx_density_" + hashlib.sha256(
        json.dumps(policy_id_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    summary = {
        **summary_base,
        "status": "completed",
        "policy_id": policy_id,
        "context_teacher_checkpoint": str(teacher_path),
        "context_student_checkpoint": str(student_path),
        "teacher_training": teacher_training,
        "student_training": student_training,
    }
    (out_dir / "context_conditional_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = train_context_conditional_opd(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
