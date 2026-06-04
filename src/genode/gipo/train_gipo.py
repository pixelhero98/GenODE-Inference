from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from genode.gipo.policy import (
    GIPO_PROTOCOL,
    DEFAULT_DENSITY_BIN_COUNT,
    DEFAULT_SUPPORT_SCHEDULE_KEYS,
    DEFAULT_TEACHER_MAX_TEMPERATURE,
    DEFAULT_TEACHER_MIN_TEMPERATURE,
    DEFAULT_TEACHER_HARD_MARGIN,
    DEFAULT_TEACHER_TARGET_ESS,
    DEFAULT_TEACHER_TARGET_TEMPERATURE,
    ARCHITECTURE_LIGHT_TRANSFORMER_V1,
    DEFAULT_TRANSFORMER_DROPOUT,
    DEFAULT_TRANSFORMER_HEADS,
    DEFAULT_TRANSFORMER_HIDDEN_DIM,
    DEFAULT_TRANSFORMER_LAYERS,
    STUDENT_TARGET_MODE_MARGIN_HARD_SOFT,
    STUDENT_TARGET_MODE_SOFT_MIXTURE,
    TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS,
    TEACHER_TEMPERATURE_MODE_FIXED,
    DensityFeatureNormalizer,
    EmbeddingNormalizer,
    attach_uniform_gipo_rewards,
    build_gipo_student_model,
    build_gipo_teacher_model,
    build_series_index_map,
    context_id_from_row,
    density_mass_for_row,
    load_context_embedding_table,
    read_metric_rows_csv,
    recommended_context_calibration_count,
    sample_context_ids_stratified,
    series_key_from_row,
    split_rows_by_context_holdout,
    split_rows_by_series_holdout,
    train_gipo_student,
    train_gipo_teacher,
    validate_gipo_support_schedule_keys,
    validate_teacher_objective_hyperparameters,
)
from genode.gipo.density_representation import density_metadata, reference_grid_hash, uniform_reference_grid
from genode.gipo.models import (
    SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config_for_rows,
    setting_feature_dim,
    solver_macro_steps,
    validate_setting_feature_mode,
    validate_time_grid,
)
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
    return validate_gipo_support_schedule_keys(canonical_order + extras)


def _stable_hash(values: Sequence[str]) -> str:
    payload = json.dumps([str(value) for value in values], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_membership_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    context_ids = sorted({context_id_from_row(row) for row in rows})
    series_keys = sorted({series_key_from_row(row) for row in rows})
    split_phases = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows})
    return {
        "row_count": int(len(rows)),
        "context_count": int(len(context_ids)),
        "series_count": int(len(series_keys)),
        "source_split_phases": split_phases,
        "context_ids": context_ids,
        "context_id_hash": _stable_hash(context_ids),
        "series_keys": series_keys,
        "series_key_hash": _stable_hash(series_keys),
    }


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    from genode.gipo.policy import series_key_from_row

    return {
        "row_count": int(len(rows)),
        "context_count": int(len({context_id_from_row(row) for row in rows})),
        "series_count": int(len({series_key_from_row(row) for row in rows})),
        "schedule_count": int(len({str(row["scheduler_key"]) for row in rows})),
    }


def _source_split_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))).strip()


def _raw_split_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("split_phase", row.get("split", ""))).strip()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GIPO continuous-density from per-example fixed/SER rows.")
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
    parser.add_argument("--teacher_architecture", choices=(ARCHITECTURE_LIGHT_TRANSFORMER_V1,), default=ARCHITECTURE_LIGHT_TRANSFORMER_V1)
    parser.add_argument("--student_architecture", choices=(ARCHITECTURE_LIGHT_TRANSFORMER_V1,), default=ARCHITECTURE_LIGHT_TRANSFORMER_V1)
    parser.add_argument("--transformer_hidden_dim", type=int, default=DEFAULT_TRANSFORMER_HIDDEN_DIM)
    parser.add_argument("--transformer_layers", type=int, default=DEFAULT_TRANSFORMER_LAYERS)
    parser.add_argument("--transformer_heads", type=int, default=DEFAULT_TRANSFORMER_HEADS)
    parser.add_argument("--transformer_dropout", type=float, default=DEFAULT_TRANSFORMER_DROPOUT)
    parser.add_argument(
        "--teacher_temperature_mode",
        choices=(TEACHER_TEMPERATURE_MODE_FIXED, TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS),
        default=TEACHER_TEMPERATURE_MODE_FIXED,
    )
    parser.add_argument("--teacher_temperature", type=float, default=DEFAULT_TEACHER_TARGET_TEMPERATURE)
    parser.add_argument("--teacher_target_ess", type=float, default=DEFAULT_TEACHER_TARGET_ESS)
    parser.add_argument("--teacher_min_temperature", type=float, default=DEFAULT_TEACHER_MIN_TEMPERATURE)
    parser.add_argument("--teacher_max_temperature", type=float, default=DEFAULT_TEACHER_MAX_TEMPERATURE)
    parser.add_argument(
        "--student_target_mode",
        choices=(STUDENT_TARGET_MODE_SOFT_MIXTURE, STUDENT_TARGET_MODE_MARGIN_HARD_SOFT),
        default=STUDENT_TARGET_MODE_SOFT_MIXTURE,
    )
    parser.add_argument("--teacher_hard_margin", type=float, default=DEFAULT_TEACHER_HARD_MARGIN)
    parser.add_argument(
        "--setting_encoder_mode",
        choices=(SETTING_ENCODER_MODE_CONTINUOUS_V3,),
        default=SETTING_ENCODER_MODE_CONTINUOUS_V3,
        help="Checkpoint-persisted setting encoder.",
    )
    parser.add_argument(
        "--setting_feature_mode",
        choices=(SETTING_ENCODER_MODE_CONTINUOUS_V3,),
        default=SETTING_ENCODER_MODE_CONTINUOUS_V3,
    )
    parser.add_argument("--teacher_utility_crps_weight", type=float, default=0.5)
    parser.add_argument("--teacher_utility_mase_weight", type=float, default=0.5)
    parser.add_argument("--teacher_rank_temperature", type=float, default=0.5)
    parser.add_argument("--teacher_regression_weight", type=float, default=0.25)
    parser.add_argument("--teacher_pair_margin", type=float, default=0.0)
    parser.add_argument("--series_unknown_dropout", type=float, default=0.10)
    parser.add_argument("--student_nfe_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--student_nfe_smoothness_mode", choices=("js", "logit_l2"), default="js")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def train_gipo(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    requested_setting_mode = str(args.setting_encoder_mode).strip() or str(args.setting_feature_mode)
    setting_feature_mode = validate_setting_feature_mode(requested_setting_mode)
    rows = read_metric_rows_csv(resolve_project_path(str(args.rows_csv)))
    if not rows:
        raise ValueError("rows_csv contains no rows.")
    locked_rows = [
        row
        for row in rows
        if _source_split_phase(row) == "locked_test" or _raw_split_phase(row) == "locked_test"
    ]
    if locked_rows:
        raise ValueError(f"GIPO training refuses locked_test rows in rows_csv; found {len(locked_rows)} locked-test rows.")
    validate_teacher_objective_hyperparameters(
        rank_temperature=float(args.teacher_rank_temperature),
        regression_weight=float(args.teacher_regression_weight),
        pair_margin=float(args.teacher_pair_margin),
    )
    support_keys = (
        validate_gipo_support_schedule_keys(_parse_csv(str(args.support_schedule_keys)))
        if str(args.support_schedule_keys).strip()
        else _observed_support(rows)
    )
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    missing_support_rows = sorted(set(support_keys) - observed_keys)
    if missing_support_rows:
        raise ValueError(f"Supervision schedules must have measured context rows; missing rows for {missing_support_rows}")

    rewarded_rows = attach_uniform_gipo_rewards(
        rows,
        support_schedule_keys=support_keys,
        utility_crps_weight=float(args.teacher_utility_crps_weight),
        utility_mase_weight=float(args.teacher_utility_mase_weight),
        pair_on_seed=True,
    )
    teacher_utility_weights = {
        "crps": float(rewarded_rows[0].get("u_comp_crps_weight", 0.5)),
        "mase": float(rewarded_rows[0].get("u_comp_mase_weight", 0.5)),
    }
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
    setting_encoder_config = setting_encoder_config_for_rows(fit_rows, mode=setting_feature_mode)

    context_embeddings = load_context_embedding_table(resolve_project_path(str(args.context_embeddings_npz)))
    fit_context_ids = sorted({context_id_from_row(row) for row in fit_rows})
    embedding_normalizer = EmbeddingNormalizer.fit(context_embeddings, fit_context_ids)
    normalized_embeddings = embedding_normalizer.transform_table(context_embeddings)
    missing_embeddings = sorted({context_id_from_row(row) for row in sampled_rows} - set(normalized_embeddings))
    if missing_embeddings:
        raise KeyError(f"Context embeddings are missing sampled contexts: {missing_embeddings[:8]}")

    series_index_map = build_series_index_map(fit_rows)
    context_dim = int(next(iter(normalized_embeddings.values())).shape[0])
    setting_dim = int(setting_feature_dim(setting_feature_mode, config=setting_encoder_config))
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
    transformer_model_config = {
        "hidden_dim": int(args.transformer_hidden_dim),
        "hidden_layers": int(args.transformer_layers),
        "attention_heads": int(args.transformer_heads),
        "dropout": float(args.transformer_dropout),
    }
    teacher = build_gipo_teacher_model(
        architecture=str(args.teacher_architecture),
        setting_dim=setting_dim,
        density_dim=int(len(reference_time_grid) - 1),
        context_dim=context_dim,
        num_series=len(series_index_map),
        model_config=transformer_model_config,
    )
    student = build_gipo_student_model(
        architecture=str(args.student_architecture),
        setting_dim=setting_dim,
        density_dim=int(len(reference_time_grid) - 1),
        context_dim=context_dim,
        num_series=len(series_index_map),
        model_config=transformer_model_config,
    )
    teacher_model_config = teacher.model_config()
    student_model_config = student.model_config()

    summary_base: Dict[str, Any] = {
        "artifact": "gipo_training_summary",
        "protocol": GIPO_PROTOCOL,
        "student_policy_type": "continuous_density",
        "student_objective": "teacher_weighted_density_mle_kl"
        if str(args.student_target_mode) == STUDENT_TARGET_MODE_SOFT_MIXTURE
        else "teacher_weighted_density_margin_hard_soft_kl",
        "teacher_objective": "pairwise_rank_plus_huber_regression",
        "model_payload_version": 2,
        "teacher_architecture": str(args.teacher_architecture),
        "student_architecture": str(args.student_architecture),
        "teacher_model_config": teacher_model_config,
        "student_model_config": student_model_config,
        "teacher_utility_weights": teacher_utility_weights,
        "student_target_mode": str(args.student_target_mode),
        "teacher_hard_margin": float(args.teacher_hard_margin),
        "setting_feature_mode": setting_feature_mode,
        "setting_encoder_mode": setting_encoder_config.mode,
        "setting_encoder_config": setting_encoder_config.to_payload(),
        "density_representation": density_meta,
        "support_schedule_keys": list(support_keys),
        "sampled_context_count": int(len(selected_context_ids)),
        "split_counts": {
            "fit": _split_counts(fit_rows),
            "context_disjoint": _split_counts(context_holdout_rows),
            "series_disjoint": _split_counts(series_holdout_rows),
        },
        "split_membership": {
            "fit": _split_membership_summary(fit_rows),
            "context_disjoint": _split_membership_summary(context_holdout_rows),
            "series_disjoint": _split_membership_summary(series_holdout_rows),
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
    teacher_training = train_gipo_teacher(
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
        setting_feature_mode=setting_feature_mode,
        setting_encoder_config=setting_encoder_config,
        device=device,
    )
    student_training = train_gipo_student(
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
        teacher_temperature_mode=str(args.teacher_temperature_mode),
        teacher_target_ess=float(args.teacher_target_ess),
        teacher_min_temperature=float(args.teacher_min_temperature),
        teacher_max_temperature=float(args.teacher_max_temperature),
        student_target_mode=str(args.student_target_mode),
        teacher_hard_margin=float(args.teacher_hard_margin),
        setting_feature_mode=setting_feature_mode,
        setting_encoder_config=setting_encoder_config,
        series_unknown_dropout=float(args.series_unknown_dropout),
        student_nfe_smoothness_weight=float(args.student_nfe_smoothness_weight),
        student_nfe_smoothness_mode=str(args.student_nfe_smoothness_mode),
        device=device,
    )

    teacher_path = out_dir / "gipo_teacher.pt"
    student_path = out_dir / "gipo_student.pt"
    torch.save(
        {
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": 2,
            "teacher_architecture": str(args.teacher_architecture),
            "teacher_model_config": teacher_model_config,
            "teacher_state": teacher.state_dict(),
            "setting_dim": int(setting_dim),
            "setting_feature_mode": setting_feature_mode,
            "setting_encoder_mode": setting_encoder_config.mode,
            "setting_encoder_config": setting_encoder_config.to_payload(),
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
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": 2,
            "student_policy_type": "continuous_density",
            "student_architecture": str(args.student_architecture),
            "student_model_config": student_model_config,
            "student_objective": student_training.get("student_objective", "teacher_weighted_density_mle_kl"),
            "student_state": student.state_dict(),
            "setting_dim": int(setting_dim),
            "setting_feature_mode": setting_feature_mode,
            "setting_encoder_mode": setting_encoder_config.mode,
            "setting_encoder_config": setting_encoder_config.to_payload(),
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
        "protocol": GIPO_PROTOCOL,
        "student_path": str(student_path),
        "reference_grid_hash": reference_grid_hash(reference_time_grid),
        "support_schedule_keys": list(support_keys),
        "teacher_selected_step": teacher_training.get("teacher_checkpoint_selection", {}).get("selected_step"),
        "student_target_mode": str(args.student_target_mode),
        "setting_feature_mode": setting_feature_mode,
        "setting_encoder_config": setting_encoder_config.to_payload(),
        "teacher_architecture": str(args.teacher_architecture),
        "student_architecture": str(args.student_architecture),
        "teacher_utility_weights": teacher_utility_weights,
    }
    policy_id = "gipo_" + hashlib.sha256(
        json.dumps(policy_id_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    summary = {
        **summary_base,
        "status": "completed",
        "policy_id": policy_id,
        "gipo_teacher_checkpoint": str(teacher_path),
        "gipo_student_checkpoint": str(student_path),
        "teacher_training": teacher_training,
        "student_training": student_training,
    }
    (out_dir / "gipo_training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = train_gipo(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
