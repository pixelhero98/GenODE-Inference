from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    project_outputs_root,
    project_paper_dataset_root,
    resolve_project_path,
)
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    LOCKED_TEST_PHASE,
    SOLVER_RUNTIME_NAMES,
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
)
from genode.gipo.density_representation import DENSITY_PROTOCOL
from genode.gipo.evaluate_schedule_summary import build_comparison_summary
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
    DEFAULT_TEACHER_TARGET_TEMPERATURE,
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
    DensityFeatureNormalizer,
    EmbeddingNormalizer,
    build_gipo_teacher_model,
    build_teacher_weighted_density_prediction_rows,
    context_id_from_row,
    load_context_embedding_table,
    normalize_teacher_utility_weights,
    read_metric_rows_csv,
    validate_gipo_attention_heads,
    validate_gipo_conditioning_style,
    validate_gipo_density_token_attention,
    validate_gipo_teacher_output,
    validate_gipo_teacher_training_metadata,
    validate_teacher_metric_target_keys,
)
from genode.gipo.report_locked_test import (
    SELECTION_MODE_CALIBRATION,
    SELECTION_MODE_REPORTING,
    _aggregate_seed_rows,
    _evaluation_seed_from_row,
    _filter_rows_to_contexts,
    _forecast_dataset_for_source_phase,
    _numeric_metric_means,
    _output_prefix,
    _parse_csv,
    _parse_int_csv,
    _read_csvs,
    _selection_split,
    _source_split_phase,
    _teacher_final_retrain_metadata,
    _validate_context_rows,
    _validate_density_bin_count,
)
from genode.gipo.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.gipo.train_gipo import _load_schedule_summary_grids
from genode.gipo.models import SETTING_ENCODER_MODE_CONTINUOUS_V3, setting_encoder_config_from_payload, setting_feature_dim, solver_macro_steps, validate_setting_feature_mode
from genode.schedule_transfer.diffusion_flow_schedules import EXPERIMENTAL_FIXED_SCHEDULE_KEYS
from genode.runtime import ProgressBar, resolve_torch_device

GIPO_TEACHER_ORACLE_SCHEDULE_KEY = "gipo_teacher_oracle"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def _load_teacher_checkpoint(
    path: str | Path,
    *,
    allow_noncanonical_conditioning: bool = False,
) -> Tuple[Any, Dict[str, int], EmbeddingNormalizer, DensityFeatureNormalizer, Tuple[float, ...], Dict[str, Any]]:
    payload = torch.load(resolve_project_path(str(path)), map_location="cpu")
    if str(payload.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError(f"Unsupported GIPO teacher protocol {payload.get('protocol')!r}; expected {GIPO_PROTOCOL!r}.")
    if int(payload.get("model_payload_version", 0)) != MODEL_PAYLOAD_VERSION:
        raise ValueError(
            f"GIPO teacher checkpoint model_payload_version must be {MODEL_PAYLOAD_VERSION}; "
            f"got {payload.get('model_payload_version')!r}."
        )
    if bool(payload.get("locked_test_used_for_selection", False)):
        raise ValueError("GIPO teacher checkpoint indicates locked_test was used for selection.")
    validate_gipo_teacher_training_metadata(payload.get("teacher_training", {}) or {})
    density_meta = dict(payload.get("density_representation", {}))
    if str(density_meta.get("density_protocol", "")) != DENSITY_PROTOCOL:
        raise ValueError("GIPO teacher checkpoint is missing density_mass_v1 metadata.")
    _validate_density_bin_count(payload, density_meta, role="teacher")
    reference_time_grid = tuple(float(x) for x in density_meta["reference_time_grid"])
    setting_feature_mode = validate_setting_feature_mode(str(payload.get("setting_feature_mode", SETTING_ENCODER_MODE_CONTINUOUS_V3)))
    setting_encoder_config = setting_encoder_config_from_payload(
        payload.get("setting_encoder_config", {"mode": setting_feature_mode})
    )
    expected_setting_dim = setting_feature_dim(setting_feature_mode, config=setting_encoder_config)
    if int(payload["setting_dim"]) != int(expected_setting_dim):
        raise ValueError(
            f"GIPO teacher checkpoint setting_dim={payload['setting_dim']} does not match "
            f"setting encoder dim {expected_setting_dim} for {setting_encoder_config.mode}."
        )
    series_index_map = {str(key): int(value) for key, value in dict(payload["series_index_map"]).items()}
    embedding_normalizer = EmbeddingNormalizer.from_payload(payload["embedding_normalizer"])
    density_normalizer = DensityFeatureNormalizer.from_payload(payload["density_feature_normalizer"])
    teacher_architecture = str(payload.get("teacher_architecture", ""))
    if teacher_architecture != ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1:
        raise ValueError(
            f"GIPO teacher checkpoints must use {ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1!r}; got {teacher_architecture!r}."
        )
    teacher_model_config = dict(payload.get("teacher_model_config", {}) or {})
    validate_gipo_conditioning_style(
        teacher_model_config,
        require_present=True,
        allow_noncanonical=bool(allow_noncanonical_conditioning),
    )
    validate_gipo_density_token_attention(teacher_model_config, require_present=True)
    validate_gipo_attention_heads(int(teacher_model_config.get("attention_heads", -1)))
    validate_gipo_teacher_output(teacher_model_config, require_present=True)
    teacher = build_gipo_teacher_model(
        architecture=teacher_architecture,
        setting_dim=int(payload["setting_dim"]),
        density_dim=int(payload["density_dim"]),
        context_dim=int(payload["context_dim"]),
        num_series=len(series_index_map),
        model_config=teacher_model_config,
        allow_noncanonical_conditioning=bool(allow_noncanonical_conditioning),
    )
    teacher.load_state_dict(payload["teacher_state"])
    teacher.eval()
    payload["setting_feature_mode"] = setting_feature_mode
    payload["setting_encoder_mode"] = setting_encoder_config.mode
    payload["setting_encoder_config"] = setting_encoder_config.to_payload()
    payload["teacher_architecture"] = teacher_architecture
    payload["teacher_model_config"] = teacher.model_config()
    return teacher, series_index_map, embedding_normalizer, density_normalizer, reference_time_grid, payload


def _group_support_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str, int, str], list[Mapping[str, Any]]]:
    grouped: Dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("dataset", row.get("dataset_key", ""))),
                str(row["solver_key"]),
                int(row["target_nfe"]),
                context_id_from_row(row),
            )
        ].append(row)
    return grouped


def _validate_teacher_oracle_support_schedule_keys(keys: Sequence[str]) -> Tuple[str, ...]:
    support_keys = tuple(str(key) for key in keys)
    fixed = set(EXPERIMENTAL_FIXED_SCHEDULE_KEYS)
    if not any(key in fixed for key in support_keys) or SER_PTG_SCHEDULE_KEY not in set(support_keys):
        raise ValueError("Teacher-oracle support rows require fixed + SER support schedule keys.")
    return support_keys


def report_gipo_teacher_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    teacher, series_index_map, embedding_normalizer, density_normalizer, reference_time_grid, checkpoint_payload = _load_teacher_checkpoint(
        str(args.gipo_teacher_checkpoint),
        allow_noncanonical_conditioning=bool(getattr(args, "allow_noncanonical_conditioning", False)),
    )
    training_summary = json.loads(resolve_project_path(str(args.training_summary)).read_text(encoding="utf-8"))
    if str(training_summary.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError("training_summary protocol does not match GIPO.")
    if bool(training_summary.get("locked_test_used_for_selection", False)):
        raise ValueError("training_summary indicates locked_test was used for selection.")
    required_checkpoint_selection_mode = str(getattr(args, "require_teacher_checkpoint_selection_mode", "") or "").strip()
    if required_checkpoint_selection_mode:
        if required_checkpoint_selection_mode != TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1:
            raise ValueError(
                "GIPO teacher-oracle reporter only supports "
                f"{TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1!r} checkpoints."
            )
        actual_selection_mode = str(
            checkpoint_payload.get("teacher_checkpoint_selection_mode")
            or training_summary.get("teacher_checkpoint_selection_mode")
            or ""
        )
        if actual_selection_mode != required_checkpoint_selection_mode:
            raise ValueError(
                f"GIPO teacher-oracle reporter requires teacher_checkpoint_selection_mode={required_checkpoint_selection_mode!r}; "
                f"got {actual_selection_mode!r}."
            )
        if required_checkpoint_selection_mode == TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1:
            final_retrain = _teacher_final_retrain_metadata(checkpoint_payload, training_summary)
            if final_retrain.get("enabled") is not True:
                raise ValueError("GIPO teacher-oracle reporter requires final teacher retrain metadata for weighted-normalized-regret checkpoints.")

    selection_mode = str(getattr(args, "selection_mode", SELECTION_MODE_REPORTING))
    split_phase = str(getattr(args, "split_phase", LOCKED_TEST_PHASE))
    support_rows = _read_csvs(str(args.support_rows))
    _validate_context_rows(support_rows, split_phase=split_phase, selection_mode=selection_mode)
    dataset = str(args.dataset)
    seeds = tuple(_parse_int_csv(str(args.seeds)))
    solvers = tuple(_parse_csv(str(args.solver_names)))
    target_nfes = tuple(_parse_int_csv(str(args.target_nfe_values)))

    expected_cells = {(dataset, seed, solver, target_nfe) for seed in seeds for solver in solvers for target_nfe in target_nfes}
    observed_cells = {
        (str(row.get("dataset", row.get("dataset_key", ""))), _evaluation_seed_from_row(row), str(row["solver_key"]), int(row["target_nfe"]))
        for row in support_rows
    }
    missing_cells = sorted(expected_cells - observed_cells)
    if missing_cells:
        raise ValueError(f"support rows are missing seed/solver/NFE cells: {missing_cells[:8]}")

    raw_embeddings = load_context_embedding_table(resolve_project_path(str(args.context_embeddings_npz)))
    missing_embeddings = sorted({context_id_from_row(row) for row in support_rows} - set(raw_embeddings))
    if missing_embeddings:
        raise KeyError(f"context embeddings NPZ is missing contexts: {missing_embeddings[:8]}")
    embeddings = embedding_normalizer.transform_table(raw_embeddings)

    schedule_grids = _load_schedule_summary_grids(_parse_csv(str(args.schedule_summary_json)))
    support_keys = _validate_teacher_oracle_support_schedule_keys(
        tuple(str(key) for key in training_summary.get("support_schedule_keys", checkpoint_payload.get("support_schedule_keys", [])))
    )
    override_mode = str(getattr(args, "setting_feature_mode", "") or "").strip()
    if override_mode:
        requested_config = setting_encoder_config_from_payload({"mode": override_mode})
        if requested_config.mode != str(checkpoint_payload.get("setting_encoder_mode", "")):
            raise ValueError(
                "Teacher-oracle reporting cannot override the checkpoint setting encoder mode: "
                f"requested {requested_config.mode}, checkpoint {checkpoint_payload.get('setting_encoder_mode')}."
            )
        setting_feature_mode = validate_setting_feature_mode(override_mode)
    else:
        setting_feature_mode = validate_setting_feature_mode(
            str(checkpoint_payload.get("setting_feature_mode", SETTING_ENCODER_MODE_CONTINUOUS_V3))
        )
    setting_encoder_config = checkpoint_payload.get("setting_encoder_config")
    teacher_utility_weights = dict(
        training_summary.get("teacher_utility_weights")
        or checkpoint_payload.get("teacher_utility_weights")
        or dict(training_summary.get("teacher_training", {}) or {}).get("teacher_utility_weights")
        or {}
    )
    if not teacher_utility_weights:
        raise ValueError("Teacher-oracle reporting requires teacher_utility_weights for metric-vector scalarization.")
    teacher_metric_targets = validate_teacher_metric_target_keys(
        dict(checkpoint_payload.get("teacher_model_config", {}) or {}).get(
            "teacher_metric_targets",
            dict(training_summary.get("teacher_training", {}) or {}).get("teacher_metric_targets"),
        )
    )
    normalize_teacher_utility_weights(teacher_metric_targets, teacher_utility_weights)
    device = resolve_torch_device(str(args.device))
    teacher.to(device)
    prediction_rows, target_summary = build_teacher_weighted_density_prediction_rows(
        teacher,
        support_rows,
        context_embeddings=embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        supervision_schedule_keys=support_keys,
        temperature=float(args.teacher_temperature),
        teacher_utility_weights=teacher_utility_weights or None,
        setting_feature_mode=setting_feature_mode,
        setting_encoder_config=setting_encoder_config,
        device=device,
    )

    grouped_support = _group_support_rows(support_rows)
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
    output_prefix = _output_prefix(split_phase, selection_mode, str(getattr(args, "report_label", "teacher_oracle"))).replace(
        "_gipo", "_gipo_teacher_oracle"
    )
    decision_rows: list[dict[str, Any]] = []
    per_context_rows: list[dict[str, Any]] = []
    with ProgressBar(len(prediction_rows), f"GIPO teacher oracle {split_phase}") as progress:
        for row_idx, prediction in enumerate(prediction_rows):
            source_phase = _source_split_phase(prediction)
            eval_ds = _forecast_dataset_for_source_phase(splits, source_phase)
            example_idx = int(prediction.get("example_idx", prediction.get("example_index", 0)))
            solver = str(prediction["solver_key"])
            target_nfe = int(prediction["target_nfe"])
            eval_seed = _evaluation_seed_from_row(prediction) + 10_000 * int(row_idx)
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
                scheduler_key=GIPO_TEACHER_ORACLE_SCHEDULE_KEY,
                dataset_key=dataset,
                split_phase=source_phase,
                checkpoint_id=str(checkpoint["checkpoint_id"]),
                example_indices=[example_idx],
                batch_size=int(args.forecast_eval_batch_size),
                progress_label="",
                return_per_example_rows=False,
            )
            key = (
                str(prediction.get("dataset", prediction.get("dataset_key", ""))),
                solver,
                int(target_nfe),
                context_id_from_row(prediction),
            )
            selected_row = {
                "dataset": dataset,
                "split_phase": str(split_phase),
                "source_split_phase": source_phase,
                "selection_mode": selection_mode,
                "selection_split": _selection_split(prediction),
                "seed": int(_evaluation_seed_from_row(prediction)),
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "scheduler_key": GIPO_TEACHER_ORACLE_SCHEDULE_KEY,
                "context_id": context_id_from_row(prediction),
                "example_idx": int(example_idx),
                "series_id": prediction.get("series_id", ""),
                "series_idx": prediction.get("series_idx", ""),
                "target_t": prediction.get("target_t", ""),
                "crps": float(metrics["crps"]),
                "mase": float(metrics["mase"]),
                "mse": metrics.get("mse", ""),
                "time_grid_json": json.dumps(prediction["time_grid"], separators=(",", ":")),
                "density_mass_hash": prediction["density_mass_hash"],
                "schedule_grid_hash": prediction["schedule_grid_hash"],
                "density_protocol": prediction["density_protocol"],
                "reference_grid_hash": prediction["reference_grid_hash"],
                "setting_feature_mode": prediction["setting_feature_mode"],
                "setting_encoder_mode": prediction["setting_encoder_mode"],
                "support_schedule_count": int(len(grouped_support.get(key, []))),
                "teacher_top_schedule_key": prediction["teacher_top_schedule_key"],
                "teacher_candidate_ess": prediction["teacher_candidate_ess"],
                "teacher_candidate_max_weight": prediction["teacher_candidate_max_weight"],
            }
            per_context_rows.append(selected_row)
            decision_rows.append(
                {
                    **selected_row,
                    "density_mass_json": json.dumps(prediction["density_mass"], separators=(",", ":")),
                    "teacher_utilities_json": prediction["teacher_utilities_json"],
                    "teacher_metric_utilities_json": prediction["teacher_metric_utilities_json"],
                    "teacher_weights_json": prediction["teacher_weights_json"],
                    "macro_steps": int(prediction["macro_steps"]),
                    "policy_source": "frozen_gipo_teacher_oracle",
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
        baseline_rows = _filter_rows_to_contexts(baseline_rows, per_context_rows)
        comparator_rows = _filter_rows_to_contexts(comparator_rows, per_context_rows)
    comparison = None
    if baseline_rows:
        comparison_student_rows = per_context_rows if selection_mode == SELECTION_MODE_CALIBRATION else aggregate_rows
        comparison = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
            student_rows=comparison_student_rows,
            dataset=dataset,
            split_phase=split_phase,
            seeds=seeds,
            solver_names=solvers,
            target_nfe_values=target_nfes,
        )
        (out_dir / f"{output_prefix}_comparison_summary.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    crps_values = [float(row["crps"]) for row in aggregate_rows]
    mase_values = [float(row["mase"]) for row in aggregate_rows]
    comparison_file = f"{output_prefix}_comparison_summary.json"
    summary = {
        "artifact": "gipo_teacher_oracle_report",
        "protocol": GIPO_PROTOCOL,
        "scheduler_key": GIPO_TEACHER_ORACLE_SCHEDULE_KEY,
        "dataset": dataset,
        "split_phase": split_phase,
        "selection_mode": selection_mode,
        "source_split_phases": sorted({_source_split_phase(row) for row in per_context_rows}),
        "context_row_count": int(len(per_context_rows)),
        "aggregate_row_count": int(len(aggregate_rows)),
        "missing_expected_cells": missing_cells,
        "mean_crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))) if crps_values else None,
        "mean_mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))) if mase_values else None,
        "metric_means": _numeric_metric_means(aggregate_rows),
        "density_representation": checkpoint_payload.get("density_representation", {}),
        "conditioning_style": dict(checkpoint_payload.get("teacher_model_config", {}) or {}).get("conditioning_style", ""),
        "setting_feature_mode": setting_feature_mode,
        "setting_encoder_mode": checkpoint_payload.get("setting_encoder_mode", ""),
        "setting_encoder_config": checkpoint_payload.get("setting_encoder_config", {}),
        "teacher_checkpoint_selection_mode": checkpoint_payload.get("teacher_checkpoint_selection_mode", training_summary.get("teacher_checkpoint_selection_mode", "")),
        "teacher_final_retrain": _teacher_final_retrain_metadata(checkpoint_payload, training_summary),
        "target_summary": target_summary,
        "locked_test_used_for_selection": False,
        "comparison_summary_path": "" if comparison is None else comparison_file,
    }
    (out_dir / f"{output_prefix}_policy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report forecast performance for frozen GIPO teacher-mixture oracle densities.")
    parser.add_argument("--gipo_teacher_checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--support_rows", required=True, help="Comma-separated measured support-row CSVs for each context/solver/NFE/seed.")
    parser.add_argument("--context_embeddings_npz", required=True)
    parser.add_argument("--schedule_summary_json", default="", help="Comma-separated schedule summaries for non-fixed references such as SER.")
    parser.add_argument("--split_phase", default=LOCKED_TEST_PHASE)
    parser.add_argument("--selection_mode", choices=(SELECTION_MODE_REPORTING, SELECTION_MODE_CALIBRATION), default=SELECTION_MODE_REPORTING)
    parser.add_argument("--require_teacher_checkpoint_selection_mode", default="")
    parser.add_argument(
        "--allow_noncanonical_conditioning",
        action="store_true",
        help="Explicitly allow noncanonical GIPO conditioning checkpoints for sidecar comparisons.",
    )
    parser.add_argument("--report_label", default="teacher_oracle")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--baseline_rows", default="")
    parser.add_argument("--comparator_rows", default="")
    parser.add_argument("--teacher_temperature", type=float, default=DEFAULT_TEACHER_TARGET_TEMPERATURE)
    parser.add_argument("--setting_feature_mode", default="")
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
    summary = report_gipo_teacher_oracle(build_argparser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
