from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.conditional_opd.models import ScheduleTeacherMLP, grid_to_intervals, setting_features, validate_time_grid
from genode.conditional_opd.objectives import rewards_by_setting, seed_mean_metric_rows
from genode.conditional_opd.ser_ptg_reference import grid_geometry
from genode.data.otflow_paths import project_outputs_root, resolve_project_path
from genode.evaluation.otflow_evaluation_support import solver_eval_multiplier, solver_macro_steps
from genode.models.otflow_train_val import save_json
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS

DEFAULT_TEMPERATURE_VALUES: Tuple[float, ...] = (0.85, 1.15)
DEFAULT_LOGIT_NOISE_VALUES: Tuple[float, ...] = (0.05, 0.10)
DEFAULT_DIRICHLET_STUDENT_ALPHAS: Tuple[float, ...] = (100.0, 200.0)
DEFAULT_RANDOM_DIRICHLET_ALPHAS: Tuple[float, ...] = (1.0, 2.0, 5.0)
DEFAULT_SELECTION_MIX = {"exploit": 8, "diverse": 5, "random": 3}


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_float_csv(text: str) -> List[float]:
    return [float(part) for part in _parse_csv(text)]


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _grid_hash(grid: Sequence[float]) -> str:
    return _hash_payload([float(x) for x in grid])


def _family_hash(schedule: Mapping[str, Any]) -> str:
    payload = []
    for item in sorted(schedule.get("predictions", []) or [], key=lambda row: (str(row["solver_key"]), int(row["target_nfe"]))):
        payload.append([str(item["solver_key"]), int(item["target_nfe"]), [float(x) for x in item["time_grid"]]])
    return _hash_payload(payload)


def _rng(seed: int, *parts: Any) -> np.random.Generator:
    token = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _format_float_key(value: float) -> str:
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")


def _intervals_to_grid(intervals: Sequence[float], *, macro_steps: int) -> Tuple[float, ...]:
    arr = np.asarray([float(x) for x in intervals], dtype=np.float64)
    if arr.size != int(macro_steps):
        raise ValueError(f"Expected {macro_steps} intervals, got {arr.size}.")
    arr = np.maximum(arr, 1e-9)
    arr = arr / max(float(np.sum(arr)), 1e-12)
    grid = np.concatenate([[0.0], np.cumsum(arr)])
    grid[0] = 0.0
    grid[-1] = 1.0
    return validate_time_grid(grid.tolist(), macro_steps=int(macro_steps))


def _prediction_from_grid(
    *,
    source: Mapping[str, Any],
    grid: Sequence[float],
    scheduler_key: str,
    candidate_source: str,
    active_round: int,
    perturbation_type: str,
    perturbation_params: Mapping[str, Any],
) -> Dict[str, Any]:
    solver_key = str(source["solver_key"])
    target_nfe = int(source["target_nfe"])
    macro_steps = solver_macro_steps(solver_key, target_nfe)
    values = validate_time_grid(grid, macro_steps=macro_steps)
    intervals = [float(x) for x in grid_to_intervals(values)]
    realized_nfe = int(macro_steps) * int(solver_eval_multiplier(solver_key))
    validity = {
        "finite": bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64)))),
        "monotone": bool(np.all(np.diff(np.asarray(values, dtype=np.float64)) > 0.0)),
        "exact_realized_nfe": bool(realized_nfe == int(target_nfe)),
    }
    if not all(validity.values()):
        raise ValueError(f"Invalid candidate grid for {scheduler_key}/{solver_key}/{target_nfe}: {validity}")
    return {
        "solver_key": solver_key,
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(macro_steps),
        "macro_steps": int(macro_steps),
        "realized_nfe": int(realized_nfe),
        "time_grid": list(values),
        "schedule_grid_hash": _grid_hash(values),
        "grid_geometry": grid_geometry(values),
        "max_macro_steps": source.get("max_macro_steps", ""),
        "candidate_source": candidate_source,
        "active_round": int(active_round),
        "student_seed": source.get("student_seed", ""),
        "opd_steps": source.get("opd_steps", source.get("opd_step_budget", "")),
        "opd_step_budget": source.get("opd_step_budget", ""),
        "perturbation_type": perturbation_type,
        "perturbation_params_json": json.dumps(dict(perturbation_params), sort_keys=True, separators=(",", ":")),
        "intervals_json": json.dumps(intervals, separators=(",", ":")),
        "utility": source.get("utility", ""),
        "validity_flags_json": json.dumps(validity, sort_keys=True, separators=(",", ":")),
    }


def _load_teacher_bundles(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    for raw in paths:
        if not str(raw).strip():
            continue
        path = resolve_project_path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        teacher_input_dim = int(payload["teacher_input_dim"])
        max_macro_steps = int(payload["max_macro_steps"])
        teacher = ScheduleTeacherMLP(teacher_input_dim, hidden_dim=256, hidden_layers=3)
        teacher.load_state_dict(payload["teacher_state"])
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        bundles.append({"path": str(path), "teacher": teacher, "max_macro_steps": max_macro_steps})
    return bundles


def _teacher_score_grid(
    *,
    teacher: ScheduleTeacherMLP,
    max_macro_steps: int,
    solver_key: str,
    target_nfe: int,
    grid: Sequence[float],
) -> float:
    macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
    values = validate_time_grid(grid, macro_steps=macro_steps)
    if int(max_macro_steps) < int(macro_steps):
        raise ValueError(f"Teacher max_macro_steps={max_macro_steps} is smaller than macro_steps={macro_steps}.")
    intervals = torch.zeros((1, int(max_macro_steps)), dtype=torch.float32)
    intervals[:, :macro_steps] = torch.tensor(grid_to_intervals(values), dtype=torch.float32)[None, :]
    features = torch.cat([setting_features(str(solver_key), int(target_nfe))[None, :], intervals], dim=-1)
    with torch.no_grad():
        return float(teacher(features).detach().cpu().item())


def _score_schedule_with_teachers(
    schedule: Mapping[str, Any],
    *,
    teacher_bundles: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    copied = dict(schedule)
    if not teacher_bundles:
        return copied
    predictions: List[Dict[str, Any]] = []
    cell_scores: List[float] = []
    for item in list(schedule.get("predictions", []) or []):
        scored_item = dict(item)
        scores = [
            _teacher_score_grid(
                teacher=bundle["teacher"],
                max_macro_steps=int(bundle["max_macro_steps"]),
                solver_key=str(scored_item["solver_key"]),
                target_nfe=int(scored_item["target_nfe"]),
                grid=scored_item["time_grid"],
            )
            for bundle in teacher_bundles
        ]
        score = float(np.mean(np.asarray(scores, dtype=np.float64)))
        scored_item["utility"] = score
        predictions.append(scored_item)
        cell_scores.append(score)
    copied["predictions"] = predictions
    copied["teacher_predicted_utility_mean"] = float(np.mean(np.asarray(cell_scores, dtype=np.float64))) if cell_scores else float("-inf")
    copied["teacher_predicted_utility_cells"] = int(len(cell_scores))
    copied["teacher_score_source"] = [str(bundle["path"]) for bundle in teacher_bundles]
    return copied


def _copy_schedule(schedule: Mapping[str, Any], *, active_round: int) -> Dict[str, Any]:
    predictions = []
    for item in schedule.get("predictions", []) or []:
        predictions.append(
            _prediction_from_grid(
                source=item,
                grid=item["time_grid"],
                scheduler_key=str(schedule["scheduler_key"]),
                candidate_source=str(schedule.get("candidate_source", "direct_student")),
                active_round=int(active_round),
                perturbation_type=str(schedule.get("perturbation_type", "none")),
                perturbation_params={},
            )
        )
    out = dict(schedule)
    out["candidate_source"] = str(schedule.get("candidate_source", "direct_student"))
    out["active_round"] = int(active_round)
    out["perturbation_type"] = str(schedule.get("perturbation_type", "none"))
    out["perturbation_params_json"] = "{}"
    out["predictions"] = predictions
    out["full_family_grid_hash"] = _family_hash(out)
    return out


def _variant_schedule(
    schedule: Mapping[str, Any],
    *,
    scheduler_key: str,
    active_round: int,
    seed: int,
    perturbation_type: str,
    perturbation_params: Mapping[str, Any],
) -> Dict[str, Any]:
    predictions = []
    for item in schedule.get("predictions", []) or []:
        solver_key = str(item["solver_key"])
        target_nfe = int(item["target_nfe"])
        macro_steps = solver_macro_steps(solver_key, target_nfe)
        intervals = np.asarray(grid_to_intervals(validate_time_grid(item["time_grid"], macro_steps=macro_steps)), dtype=np.float64)
        if perturbation_type == "temperature":
            temperature = float(perturbation_params["temperature"])
            transformed = np.power(np.maximum(intervals, 1e-9), 1.0 / temperature)
        elif perturbation_type == "logit_noise":
            sigma = float(perturbation_params["sigma"])
            transformed = np.log(np.maximum(intervals, 1e-9)) + _rng(seed, scheduler_key, solver_key, target_nfe, sigma).normal(0.0, sigma, size=intervals.shape)
            transformed = np.exp(transformed - float(np.max(transformed)))
        elif perturbation_type == "dirichlet_student":
            alpha = float(perturbation_params["alpha"])
            transformed = _rng(seed, scheduler_key, solver_key, target_nfe, alpha).dirichlet(np.maximum(alpha * intervals, 1e-3))
        else:
            raise ValueError(f"Unsupported perturbation_type={perturbation_type!r}.")
        grid = _intervals_to_grid(transformed, macro_steps=macro_steps)
        predictions.append(
            _prediction_from_grid(
                source=item,
                grid=grid,
                scheduler_key=scheduler_key,
                candidate_source="direct_student_perturbation",
                active_round=int(active_round),
                perturbation_type=perturbation_type,
                perturbation_params=perturbation_params,
            )
        )
    return {
        "scheduler_key": scheduler_key,
        "schedule_name": f"Train20 candidate {scheduler_key}",
        "comparison_role": "train20_generated_candidate",
        "candidate_source": "direct_student_perturbation",
        "active_round": int(active_round),
        "student_seed": schedule.get("student_seed", ""),
        "opd_steps": schedule.get("opd_steps", schedule.get("opd_step_budget", "")),
        "perturbation_type": perturbation_type,
        "perturbation_params_json": json.dumps(dict(perturbation_params), sort_keys=True, separators=(",", ":")),
        "teacher_predicted_utility_mean": schedule.get("teacher_predicted_utility_mean"),
        "predictions": predictions,
    }


def _random_schedule(
    *,
    template_schedule: Mapping[str, Any],
    scheduler_key: str,
    active_round: int,
    seed: int,
    alpha: float,
) -> Dict[str, Any]:
    predictions = []
    for item in template_schedule.get("predictions", []) or []:
        solver_key = str(item["solver_key"])
        target_nfe = int(item["target_nfe"])
        macro_steps = solver_macro_steps(solver_key, target_nfe)
        intervals = _rng(seed, scheduler_key, solver_key, target_nfe, alpha).dirichlet(np.full(int(macro_steps), float(alpha), dtype=np.float64))
        grid = _intervals_to_grid(intervals, macro_steps=macro_steps)
        predictions.append(
            _prediction_from_grid(
                source=item,
                grid=grid,
                scheduler_key=scheduler_key,
                candidate_source="random_dirichlet",
                active_round=int(active_round),
                perturbation_type="random_dirichlet",
                perturbation_params={"alpha": float(alpha)},
            )
        )
    return {
        "scheduler_key": scheduler_key,
        "schedule_name": f"Train20 random Dirichlet alpha={alpha:g}",
        "comparison_role": "train20_generated_candidate",
        "candidate_source": "random_dirichlet",
        "active_round": int(active_round),
        "perturbation_type": "random_dirichlet",
        "perturbation_params_json": json.dumps({"alpha": float(alpha)}, sort_keys=True, separators=(",", ":")),
        "teacher_predicted_utility_mean": float("-inf"),
        "predictions": predictions,
    }


def load_schedules(paths: Sequence[str | Path]) -> Tuple[str, List[Dict[str, Any]]]:
    dataset = ""
    schedules: List[Dict[str, Any]] = []
    for source_idx, raw in enumerate(paths):
        path = resolve_project_path(str(raw))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not dataset:
            dataset = str(payload.get("dataset", ""))
        elif str(payload.get("dataset", dataset)) != dataset:
            raise ValueError(f"Candidate summaries use mixed datasets: {dataset!r} and {payload.get('dataset')!r}.")
        for schedule in list(payload.get("schedules", []) or []):
            copied = dict(schedule)
            copied["source_summary_index"] = int(source_idx)
            copied["source_schedule_summary"] = str(path)
            schedules.append(copied)
    return dataset, schedules


def build_candidate_pool(
    *,
    source_schedule_summaries: Sequence[str | Path],
    active_round: int,
    seed: int,
    temperature_values: Sequence[float] = DEFAULT_TEMPERATURE_VALUES,
    logit_noise_values: Sequence[float] = DEFAULT_LOGIT_NOISE_VALUES,
    dirichlet_student_alpha_values: Sequence[float] = DEFAULT_DIRICHLET_STUDENT_ALPHAS,
    random_dirichlet_alpha_values: Sequence[float] = DEFAULT_RANDOM_DIRICHLET_ALPHAS,
    include_base: bool = True,
    teacher_checkpoint_paths: Sequence[str | Path] = (),
) -> Dict[str, Any]:
    dataset, source_schedules = load_schedules(source_schedule_summaries)
    if not source_schedules:
        raise ValueError("Candidate pool requires at least one source schedule.")
    teacher_bundles = _load_teacher_bundles(teacher_checkpoint_paths)
    schedules: List[Dict[str, Any]] = []
    if include_base:
        schedules.extend(_copy_schedule(schedule, active_round=int(active_round)) for schedule in source_schedules)
    for schedule in source_schedules:
        base_key = str(schedule["scheduler_key"])
        for temperature in temperature_values:
            schedules.append(
                _variant_schedule(
                    schedule,
                    scheduler_key=f"{base_key}_temp{_format_float_key(float(temperature))}",
                    active_round=int(active_round),
                    seed=int(seed),
                    perturbation_type="temperature",
                    perturbation_params={"temperature": float(temperature)},
                )
            )
        for sigma in logit_noise_values:
            schedules.append(
                _variant_schedule(
                    schedule,
                    scheduler_key=f"{base_key}_noise{_format_float_key(float(sigma))}",
                    active_round=int(active_round),
                    seed=int(seed),
                    perturbation_type="logit_noise",
                    perturbation_params={"sigma": float(sigma)},
                )
            )
        for alpha in dirichlet_student_alpha_values:
            schedules.append(
                _variant_schedule(
                    schedule,
                    scheduler_key=f"{base_key}_dirichlet{_format_float_key(float(alpha))}",
                    active_round=int(active_round),
                    seed=int(seed),
                    perturbation_type="dirichlet_student",
                    perturbation_params={"alpha": float(alpha)},
                )
            )
    template = source_schedules[0]
    for alpha in random_dirichlet_alpha_values:
        schedules.append(
            _random_schedule(
                template_schedule=template,
                scheduler_key=f"train20_r{int(active_round)}_random_dirichlet{_format_float_key(float(alpha))}",
                active_round=int(active_round),
                seed=int(seed),
                alpha=float(alpha),
            )
        )
    deduped: List[Dict[str, Any]] = []
    seen_hashes = set()
    seen_keys = set()
    for schedule in schedules:
        key = str(schedule["scheduler_key"])
        family_hash = _family_hash(schedule)
        if key in seen_keys or family_hash in seen_hashes:
            continue
        copied = dict(schedule)
        copied["full_family_grid_hash"] = family_hash
        source_index = copied.get("source_summary_index")
        if source_index in (None, ""):
            scoring_bundles = teacher_bundles
        else:
            idx = int(source_index)
            scoring_bundles = teacher_bundles[idx : idx + 1] if idx < len(teacher_bundles) else teacher_bundles
        deduped.append(_score_schedule_with_teachers(copied, teacher_bundles=scoring_bundles))
        seen_keys.add(key)
        seen_hashes.add(family_hash)
    return {
        "status": "ready",
        "artifact": "train20_candidate_pool_schedule_summary",
        "dataset": dataset,
        "active_round": int(active_round),
        "seed": int(seed),
        "baseline_schedule": False,
        "source_schedule_summaries": [str(resolve_project_path(str(path))) for path in source_schedule_summaries],
        "teacher_checkpoint_paths": [str(resolve_project_path(str(path))) for path in teacher_checkpoint_paths],
        "candidate_generation": {
            "include_base": bool(include_base),
            "temperature_values": [float(x) for x in temperature_values],
            "logit_noise_values": [float(x) for x in logit_noise_values],
            "dirichlet_student_alpha_values": [float(x) for x in dirichlet_student_alpha_values],
            "random_dirichlet_alpha_values": [float(x) for x in random_dirichlet_alpha_values],
        },
        "schedule_count": int(len(deduped)),
        "schedules": deduped,
    }


def _schedule_vector(schedule: Mapping[str, Any]) -> np.ndarray:
    values: List[float] = []
    for item in sorted(schedule.get("predictions", []) or [], key=lambda row: (str(row["solver_key"]), int(row["target_nfe"]))):
        values.extend(float(x) for x in grid_to_intervals(item["time_grid"]))
    return np.asarray(values, dtype=np.float64)


def _schedule_score(schedule: Mapping[str, Any]) -> float:
    value = schedule.get("train_tuning_utility_mean", schedule.get("teacher_predicted_utility_mean", 0.0))
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return score if math.isfinite(score) else -float("inf")


def attach_train_tuning_utilities(
    schedules: Sequence[Mapping[str, Any]],
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    fixed_reference_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not candidate_rows:
        return [dict(schedule) for schedule in schedules]
    aggregate_candidates = seed_mean_metric_rows(candidate_rows)
    aggregate_fixed = seed_mean_metric_rows(fixed_reference_rows)
    rewards = rewards_by_setting([*aggregate_fixed, *aggregate_candidates], fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
    utilities: Dict[str, List[float]] = {}
    for row in aggregate_candidates:
        setting = (str(row["solver_key"]), int(row["target_nfe"]))
        schedule_key = str(row["scheduler_key"])
        if schedule_key in rewards.get(setting, {}):
            utilities.setdefault(schedule_key, []).append(float(rewards[setting][schedule_key]))
    out = []
    for schedule in schedules:
        copied = dict(schedule)
        vals = utilities.get(str(schedule["scheduler_key"]), [])
        if vals:
            copied["train_tuning_utility_mean"] = float(np.mean(np.asarray(vals, dtype=np.float64)))
            copied["train_tuning_utility_cells"] = int(len(vals))
        out.append(copied)
    return out


def select_candidate_schedules(
    schedules: Sequence[Mapping[str, Any]],
    *,
    exploit_count: int = DEFAULT_SELECTION_MIX["exploit"],
    diverse_count: int = DEFAULT_SELECTION_MIX["diverse"],
    random_count: int = DEFAULT_SELECTION_MIX["random"],
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_key = {str(schedule["scheduler_key"]): dict(schedule) for schedule in schedules}
    ordered = sorted(by_key.values(), key=lambda schedule: (-_schedule_score(schedule), str(schedule["scheduler_key"])))
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    for schedule in ordered[: max(0, int(exploit_count))]:
        selected.append(dict(schedule))
        selected_keys.add(str(schedule["scheduler_key"]))
    vectors = {str(schedule["scheduler_key"]): _schedule_vector(schedule) for schedule in ordered}
    remaining = [schedule for schedule in ordered if str(schedule["scheduler_key"]) not in selected_keys]
    for _ in range(max(0, int(diverse_count))):
        if not remaining:
            break
        if not selected:
            chosen = remaining[0]
        else:
            def min_distance(schedule: Mapping[str, Any]) -> Tuple[float, str]:
                vec = vectors[str(schedule["scheduler_key"])]
                distances = [float(np.mean(np.abs(vec - vectors[str(sel["scheduler_key"])]))) for sel in selected]
                return (min(distances), str(schedule["scheduler_key"]))

            chosen = max(remaining, key=min_distance)
        selected.append(dict(chosen))
        selected_keys.add(str(chosen["scheduler_key"]))
        remaining = [schedule for schedule in remaining if str(schedule["scheduler_key"]) not in selected_keys]
    random_order = sorted(
        remaining,
        key=lambda schedule: _hash_payload(["random", int(seed), str(schedule["scheduler_key"])]),
    )
    for schedule in random_order[: max(0, int(random_count))]:
        selected.append(dict(schedule))
        selected_keys.add(str(schedule["scheduler_key"]))
    plan = {
        "selection_unit": "complete_schedule_family",
        "selection_mix": {
            "exploit": int(exploit_count),
            "diverse": int(diverse_count),
            "random": int(random_count),
        },
        "selected_count": int(len(selected)),
        "selected_schedule_keys": [str(schedule["scheduler_key"]) for schedule in selected],
    }
    return selected, plan


def _read_rows_csv(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part in _parse_csv(str(path)):
        resolved = resolve_project_path(str(part))
        if not resolved.exists():
            continue
        with resolved.open("r", newline="", encoding="utf-8") as fh:
            rows.extend(dict(row) for row in csv.DictReader(fh))
    return rows


def build_and_select_candidate_pool(args: argparse.Namespace) -> Dict[str, Any]:
    source_paths = _parse_csv(str(args.source_schedule_summaries))
    pool = build_candidate_pool(
        source_schedule_summaries=source_paths,
        active_round=int(args.active_round),
        seed=int(args.seed),
        temperature_values=_parse_float_csv(str(args.temperature_values)),
        logit_noise_values=_parse_float_csv(str(args.logit_noise_values)),
        dirichlet_student_alpha_values=_parse_float_csv(str(args.dirichlet_student_alpha_values)),
        random_dirichlet_alpha_values=_parse_float_csv(str(args.random_dirichlet_alpha_values)),
        include_base=bool(args.include_base),
        teacher_checkpoint_paths=_parse_csv(str(args.teacher_checkpoint_paths)),
    )
    schedules = list(pool["schedules"])
    if str(args.selection_rows_csv).strip() and str(args.fixed_reference_rows_csv).strip():
        schedules = attach_train_tuning_utilities(
            schedules,
            candidate_rows=_read_rows_csv(args.selection_rows_csv),
            fixed_reference_rows=_read_rows_csv(args.fixed_reference_rows_csv),
        )
    selected, selection = select_candidate_schedules(
        schedules,
        exploit_count=int(args.exploit_count),
        diverse_count=int(args.diverse_count),
        random_count=int(args.random_count),
        seed=int(args.seed),
    )
    pool["schedules"] = schedules
    pool["schedule_count"] = int(len(schedules))
    pool["selection"] = selection
    out_path = resolve_project_path(str(args.out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(pool, str(out_path))
    selected_summary = dict(pool)
    selected_summary["artifact"] = "train20_selected_candidate_schedule_summary"
    selected_summary["schedules"] = selected
    selected_summary["schedule_count"] = int(len(selected))
    selected_summary["selection"] = selection
    selected_path = resolve_project_path(str(args.selected_out_path))
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(selected_summary, str(selected_path))
    return {
        "status": "ready",
        "candidate_pool": str(out_path),
        "selected_schedule_summary": str(selected_path),
        "pool_schedule_count": int(len(schedules)),
        "selected_schedule_count": int(len(selected)),
        "selection": selection,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal Train20 complete-family candidate pool builder and selector.")
    parser.add_argument("--source_schedule_summaries", required=True)
    parser.add_argument("--out_path", default=str(project_outputs_root() / "train20_candidate_pool" / "candidate_pool_schedule_summary.json"))
    parser.add_argument("--selected_out_path", default=str(project_outputs_root() / "train20_candidate_pool" / "selected_candidate_schedule_summary.json"))
    parser.add_argument("--active_round", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature_values", default=",".join(str(x) for x in DEFAULT_TEMPERATURE_VALUES))
    parser.add_argument("--logit_noise_values", default=",".join(str(x) for x in DEFAULT_LOGIT_NOISE_VALUES))
    parser.add_argument("--dirichlet_student_alpha_values", default=",".join(str(x) for x in DEFAULT_DIRICHLET_STUDENT_ALPHAS))
    parser.add_argument("--random_dirichlet_alpha_values", default=",".join(str(x) for x in DEFAULT_RANDOM_DIRICHLET_ALPHAS))
    parser.add_argument("--include_base", action="store_true", default=True)
    parser.add_argument("--selection_rows_csv", default="")
    parser.add_argument("--fixed_reference_rows_csv", default="")
    parser.add_argument("--teacher_checkpoint_paths", default="")
    parser.add_argument("--exploit_count", type=int, default=DEFAULT_SELECTION_MIX["exploit"])
    parser.add_argument("--diverse_count", type=int, default=DEFAULT_SELECTION_MIX["diverse"])
    parser.add_argument("--random_count", type=int, default=DEFAULT_SELECTION_MIX["random"])
    return parser


def main() -> None:
    summary = build_and_select_candidate_pool(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
