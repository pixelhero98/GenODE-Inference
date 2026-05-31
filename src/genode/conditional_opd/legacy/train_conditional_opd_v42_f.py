from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from genode.conditional_opd.evaluate_schedule_summary import (
    SELECTED_STUDENT_SCHEDULE_KEY,
    SELECTED_STUDENT_SCHEDULE_NAME,
)
from genode.conditional_opd.models import (
    ScheduleStudentMLP,
    ScheduleTeacherMLP,
    count_parameters,
    grid_to_intervals,
    setting_features,
    solver_macro_steps,
    validate_time_grid,
)
from genode.conditional_opd.objectives import (
    build_fixed_reference_table,
    source_balanced_rewards_by_setting,
    source_balanced_seed_mean_rows,
)
from genode.conditional_opd.ser_ptg_reference import grid_geometry
from genode.conditional_opd.train_conditional_opd import (
    DEFAULT_LATE_BIASED_DEMO_SCHEDULES,
    DEFAULT_LATE_BIASED_DEMO_WEIGHT,
    DEFAULT_SOLVERS,
    DEFAULT_TARGET_NFES,
    DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION,
    DEFAULT_TEACHER_PAIR_MARGIN,
    DEFAULT_TEACHER_PAIRS_PER_CANDIDATE,
    DEFAULT_TEACHER_RANK_TEMPERATURE,
    DEFAULT_TEACHER_REGRESSION_WEIGHT,
    _assert_complete_seed_rows,
    _assert_expected_train_tuning_metadata,
    _clean_split_rows,
    _load_csv_rows,
    _load_schedule_summary_grids,
    _load_schedule_summary_grids_many,
    _parse_csv,
    _parse_int_csv,
    _schedule_keys_from_summary_paths,
    differentiable_teacher_features,
    split_teacher_diagnostic_holdout,
    teacher_schedule_weights,
    train_teacher,
)
from genode.data.otflow_paths import project_outputs_root, resolve_project_path
from genode.evaluation.otflow_evaluation_support import TRAIN_TUNING_PHASE, VALIDATION_PHASE
from genode.models.otflow_train_val import save_json, seed_all
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, build_schedule_grid

CALIBRATION_TRAIN_SOURCE = "calibration_train_part"
CALIBRATION_VAL_SOURCE = "calibration_val_part"
DEFAULT_OUT_DIR = project_outputs_root() / "train20_v42_f_calibration20_fullval_uniform_seedmean"
DEFAULT_STUDENT_PREFIX = "conditional_opd_student_v42f_"
DEFAULT_STUDENT_CHECKPOINT_MODES: Tuple[str, ...] = ("fixed_epoch", "lowest_internal_loss")


def _parse_teacher_fixed_schedule_spec(text: str) -> Tuple[str, ...]:
    spec = str(text).strip()
    if not spec or spec.lower() == "none":
        return ()
    if spec.lower() == "all":
        return tuple(BASELINE_SCHEDULE_KEYS)
    keys = tuple(_parse_csv(spec))
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"teacher_fixed_schedule_keys contains duplicates: {duplicates}")
    unsupported = sorted(set(keys) - set(BASELINE_SCHEDULE_KEYS))
    if unsupported:
        raise ValueError(f"teacher_fixed_schedule_keys must be fixed baselines; unsupported: {unsupported}")
    return keys


def _read_rows_many(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_csv_rows(path))
    return rows


def _fit_student_to_reference(
    student: ScheduleStudentMLP,
    reference_pairs: Sequence[Tuple[str, int, Sequence[float]]],
    *,
    steps: int,
    lr: float,
    loss_name: str,
) -> List[Dict[str, float]]:
    targets = []
    for solver, target_nfe, grid in reference_pairs:
        macro_steps = solver_macro_steps(str(solver), int(target_nfe))
        target = torch.tensor(grid_to_intervals(validate_time_grid(grid, macro_steps=macro_steps)), dtype=torch.float32)
        targets.append((setting_features(str(solver), int(target_nfe)), target, int(macro_steps)))
    if not targets:
        raise ValueError("Student initialization requires at least one reference target.")
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, float]] = []
    for step in range(int(steps)):
        loss_terms = []
        for feat, target, macro_steps in targets:
            pred = student.intervals(feat[None, :], macro_steps)[0]
            loss_terms.append(F.mse_loss(pred, target))
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            losses.append({"step": int(step + 1), loss_name: float(loss.detach().cpu().item())})
    return losses


def _optimize_student_with_teacher_checkpoints(
    student: ScheduleStudentMLP,
    teacher: ScheduleTeacherMLP,
    settings: Sequence[Tuple[str, int]],
    *,
    max_macro_steps: int,
    max_steps: int,
    lr: float,
    checkpoint_steps: Sequence[int],
) -> Dict[str, Any]:
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    checkpoints = {int(step) for step in checkpoint_steps if int(step) > 0}
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, float]] = []
    states_by_step: Dict[str, Dict[str, torch.Tensor]] = {}
    best_state: Dict[str, torch.Tensor] | None = None
    best_step = 0
    best_loss = float("inf")
    best_utility = float("-inf")
    for step in range(int(max_steps)):
        utility_terms = []
        for solver, target_nfe in settings:
            macro_steps = solver_macro_steps(str(solver), int(target_nfe))
            feat = setting_features(str(solver), int(target_nfe))[None, :]
            intervals = student.intervals(feat, macro_steps)
            padded = torch.zeros((1, int(max_macro_steps)), dtype=intervals.dtype, device=intervals.device)
            padded[:, :macro_steps] = intervals
            teacher_input = differentiable_teacher_features(feat, padded, max_macro_steps=max_macro_steps)
            utility_terms.append(teacher(teacher_input).mean())
        objective = torch.stack(utility_terms).mean()
        loss = -objective
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu().item())
        utility_value = float(objective.detach().cpu().item())
        step_number = int(step + 1)
        if loss_value < best_loss:
            best_loss = loss_value
            best_utility = utility_value
            best_step = step_number
            best_state = copy.deepcopy(student.state_dict())
        if step_number in checkpoints:
            states_by_step[str(step_number)] = copy.deepcopy(student.state_dict())
        if step == 0 or step == int(max_steps) - 1 or step_number in checkpoints or step_number % max(1, int(max_steps) // 5) == 0:
            losses.append(
                {
                    "step": step_number,
                    "teacher_predicted_utility": utility_value,
                    "student_total_loss": loss_value,
                    "checkpointed": bool(step_number in checkpoints),
                    "best_internal_loss_so_far": float(best_loss),
                    "best_internal_step_so_far": int(best_step),
                }
            )
    if best_state is None:
        raise RuntimeError("Student optimization produced no internal checkpoint state.")
    return {
        "losses": losses,
        "states_by_step": states_by_step,
        "best_state": best_state,
        "best_step": int(best_step),
        "best_loss": float(best_loss),
        "best_teacher_predicted_utility": float(best_utility),
    }


def _uniform_reference_targets(settings: Sequence[Tuple[str, int]]) -> List[Tuple[str, int, Tuple[float, ...]]]:
    targets: List[Tuple[str, int, Tuple[float, ...]]] = []
    for solver, target_nfe in settings:
        macro_steps = solver_macro_steps(str(solver), int(target_nfe))
        grid = build_schedule_grid("uniform", macro_steps)
        if grid is None:
            raise ValueError(f"No uniform grid for solver={solver}, target_nfe={target_nfe}.")
        targets.append((str(solver), int(target_nfe), validate_time_grid(grid, macro_steps=macro_steps)))
    return targets


def _teacher_utility_for_grid(
    teacher: ScheduleTeacherMLP,
    *,
    solver: str,
    target_nfe: int,
    grid: Sequence[float],
    max_macro_steps: int,
) -> float:
    macro_steps = solver_macro_steps(str(solver), int(target_nfe))
    intervals = torch.tensor(np.diff(np.asarray(grid, dtype=np.float64)), dtype=torch.float32)
    padded = torch.zeros((1, int(max_macro_steps)), dtype=torch.float32)
    padded[:, :macro_steps] = intervals[None, :]
    with torch.no_grad():
        return float(
            teacher(
                differentiable_teacher_features(
                    setting_features(str(solver), int(target_nfe))[None, :],
                    padded,
                    max_macro_steps=max_macro_steps,
                )
            )
            .detach()
            .cpu()
            .item()
        )


def _seed_mean_predictions_from_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    setting_dim: int,
    max_macro_steps: int,
    teacher: ScheduleTeacherMLP,
    checkpoint_mode: str,
    seed_values: Sequence[int],
    checkpoint_steps_by_seed: Mapping[str, int],
) -> List[Dict[str, Any]]:
    if not states:
        raise ValueError("Seed-mean schedule requires at least one student state.")
    model = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
    predictions: List[Dict[str, Any]] = []
    for solver in solvers:
        for target_nfe in target_nfes:
            macro_steps = solver_macro_steps(str(solver), int(target_nfe))
            feat = setting_features(str(solver), int(target_nfe))[None, :]
            logits: List[torch.Tensor] = []
            for state in states:
                model.load_state_dict(dict(state))
                with torch.no_grad():
                    logits.append(model.interval_logits(feat)[0, :macro_steps].detach().clone())
            mean_logits = torch.stack(logits, dim=0).mean(dim=0)
            intervals = torch.softmax(mean_logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            intervals = np.maximum(intervals, 1e-7)
            intervals = intervals / float(np.sum(intervals))
            grid = np.concatenate([[0.0], np.cumsum(intervals)])
            grid[0] = 0.0
            grid[-1] = 1.0
            values = validate_time_grid([float(x) for x in grid.tolist()], macro_steps=macro_steps)
            utility = _teacher_utility_for_grid(
                teacher,
                solver=str(solver),
                target_nfe=int(target_nfe),
                grid=values,
                max_macro_steps=max_macro_steps,
            )
            predictions.append(
                {
                    "solver_key": str(solver),
                    "target_nfe": int(target_nfe),
                    "runtime_nfe": int(macro_steps),
                    "macro_steps": int(macro_steps),
                    "realized_nfe": int(target_nfe),
                    "time_grid": list(values),
                    "grid_geometry": grid_geometry(values),
                    "max_macro_steps": int(max_macro_steps),
                    "candidate_source": "seed_mean_student",
                    "student_seed": "seed_mean",
                    "student_seed_values": list(int(seed) for seed in seed_values),
                    "student_checkpoint_steps_by_seed": dict(checkpoint_steps_by_seed),
                    "student_checkpoint_mode": str(checkpoint_mode),
                    "averaged_representation": "interval_logits",
                    "perturbation_type": "none",
                    "perturbation_params_json": "{}",
                    "intervals_json": json.dumps([float(x) for x in intervals.tolist()], separators=(",", ":")),
                    "utility": float(utility),
                    "validity_flags_json": json.dumps({"finite": True, "monotone": True, "exact_realized_nfe": True}, separators=(",", ":")),
                }
            )
    return predictions


def _write_selected_seed_mean_summary(
    schedule_summary: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    out_path: str | Path,
) -> Dict[str, Any]:
    selected_key = str(selection["selected_schedule_key"])
    selected = None
    for schedule in list(schedule_summary.get("schedules", []) or []):
        if str(schedule.get("scheduler_key")) == selected_key:
            selected = dict(schedule)
            break
    if selected is None:
        raise ValueError(f"Selected seed-mean schedule {selected_key!r} was not present in student summary.")
    predictions = []
    for item in list(selected.get("predictions", []) or []):
        copied = dict(item)
        copied["source_scheduler_key"] = selected_key
        predictions.append(copied)
    schedule = {
        "scheduler_key": SELECTED_STUDENT_SCHEDULE_KEY,
        "schedule_name": SELECTED_STUDENT_SCHEDULE_NAME,
        "comparison_role": "learned_student_frozen_seed_mean_calibration",
        "source_scheduler_key": selected_key,
        "student_checkpoint_mode": selected.get("student_checkpoint_mode"),
        "uses_validation_labels_for_selection": False,
        "predictions": predictions,
    }
    summary = {
        "status": "ready",
        "artifact": "v42_f_selected_seed_mean_student_schedule_summary",
        "dataset": str(schedule_summary.get("dataset")),
        "selection": dict(selection),
        "selected_source_schedule_key": selected_key,
        "baseline_schedule": False,
        "schedules": [schedule],
        "predictions": predictions,
    }
    out = resolve_project_path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, str(out))
    return summary


def train_conditional_opd_v42_f(args: argparse.Namespace) -> Dict[str, Any]:
    solvers = tuple(_parse_csv(args.solver_names))
    target_nfes = tuple(_parse_int_csv(args.target_nfe_values))
    train_seeds = tuple(_parse_int_csv(args.calibration_train_seeds))
    val_seeds = tuple(_parse_int_csv(args.calibration_val_seeds))
    student_seeds = tuple(_parse_int_csv(args.student_seeds))
    budgets = sorted(set(_parse_int_csv(args.student_opd_step_values) + [int(args.fixed_epoch_steps)]))
    checkpoint_modes = tuple(_parse_csv(args.student_checkpoint_modes))
    unsupported_modes = sorted(set(checkpoint_modes) - set(DEFAULT_STUDENT_CHECKPOINT_MODES))
    if unsupported_modes:
        raise ValueError(f"Unsupported student checkpoint modes: {unsupported_modes}")
    if str(args.final_checkpoint_mode) not in checkpoint_modes:
        raise ValueError("final_checkpoint_mode must be one of student_checkpoint_modes.")
    teacher_fixed_schedule_keys = _parse_teacher_fixed_schedule_spec(str(args.teacher_fixed_schedule_keys))
    reward_reference_schedule_keys = tuple(BASELINE_SCHEDULE_KEYS)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_macro_steps = max(solver_macro_steps(solver, nfe) for solver in solvers for nfe in target_nfes)
    setting_dim = int(setting_features("euler", 4).numel())
    teacher_input_dim = setting_dim + int(max_macro_steps)
    teacher = ScheduleTeacherMLP(teacher_input_dim, hidden_dim=256, hidden_layers=3)
    teacher_param_count = count_parameters(teacher)
    student_param_count = count_parameters(ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2))

    if bool(args.dry_run):
        summary = {
            "status": "dry_run",
            "protocol": "v4.2_f_final_calibration_set",
            "dataset": str(args.dataset),
            "solvers": list(solvers),
            "target_nfes": list(target_nfes),
            "calibration_sources": [CALIBRATION_TRAIN_SOURCE, CALIBRATION_VAL_SOURCE],
            "calibration_train_seeds": list(train_seeds),
            "calibration_val_seeds": list(val_seeds),
            "student_seeds": list(student_seeds),
            "student_initialization": "uniform",
            "student_checkpoint_modes": list(checkpoint_modes),
            "final_checkpoint_mode": str(args.final_checkpoint_mode),
            "fixed_epoch_steps": int(args.fixed_epoch_steps),
            "uses_validation_selection": False,
            "teacher_fixed_schedule_keys": list(teacher_fixed_schedule_keys),
            "reward_reference_schedule_keys": list(reward_reference_schedule_keys),
            "teacher_parameters": int(teacher_param_count),
            "student_parameters": int(student_param_count),
        }
        save_json(summary, str(out_dir / "conditional_opd_v42_f_summary.json"))
        return summary

    train_reference_rows = _clean_split_rows(
        _load_csv_rows(args.train_rows_csv),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=reward_reference_schedule_keys,
        seeds=train_seeds,
        required_split_phase=TRAIN_TUNING_PHASE,
    )
    val_reference_rows = _clean_split_rows(
        _load_csv_rows(args.validation_rows_csv),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=reward_reference_schedule_keys,
        seeds=val_seeds,
        required_split_phase=VALIDATION_PHASE,
    )
    _assert_expected_train_tuning_metadata(
        train_reference_rows,
        expected_sampler=str(args.expected_train_tuning_sampler),
        expected_fraction=str(args.expected_train_tuning_fraction),
        label="V4.2-F calibration train fixed references",
    )
    _assert_complete_seed_rows(
        train_reference_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=reward_reference_schedule_keys,
        seeds=train_seeds,
        label="V4.2-F calibration train fixed references",
    )
    _assert_complete_seed_rows(
        val_reference_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=reward_reference_schedule_keys,
        seeds=val_seeds,
        label="V4.2-F calibration validation fixed references",
    )

    candidate_summary_paths = tuple(_parse_csv(args.candidate_schedule_summary))
    candidate_keys = _schedule_keys_from_summary_paths(candidate_summary_paths)
    if not candidate_keys:
        raise ValueError("V4.2-F teacher training requires evaluated BO candidate schedule summaries.")
    train_candidate_rows = _clean_split_rows(
        _read_rows_many(_parse_csv(args.candidate_train_rows_csv)),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=candidate_keys,
        seeds=train_seeds,
        required_split_phase=TRAIN_TUNING_PHASE,
    )
    val_candidate_rows = _clean_split_rows(
        _read_rows_many(_parse_csv(args.candidate_validation_rows_csv)),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=candidate_keys,
        seeds=val_seeds,
        required_split_phase=VALIDATION_PHASE,
    )
    _assert_expected_train_tuning_metadata(
        train_candidate_rows,
        expected_sampler=str(args.expected_train_tuning_sampler),
        expected_fraction=str(args.expected_train_tuning_fraction),
        label="V4.2-F calibration train BO candidates",
    )
    _assert_complete_seed_rows(
        train_candidate_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=candidate_keys,
        seeds=train_seeds,
        label="V4.2-F calibration train BO candidates",
    )
    _assert_complete_seed_rows(
        val_candidate_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=candidate_keys,
        seeds=val_seeds,
        label="V4.2-F calibration validation BO candidates",
    )

    source_rows = {
        CALIBRATION_TRAIN_SOURCE: [*train_reference_rows, *train_candidate_rows],
        CALIBRATION_VAL_SOURCE: [*val_reference_rows, *val_candidate_rows],
    }
    source_weights = {
        CALIBRATION_TRAIN_SOURCE: float(args.calibration_train_weight),
        CALIBRATION_VAL_SOURCE: float(args.calibration_val_weight),
    }
    rewards = source_balanced_rewards_by_setting(
        source_rows,
        fixed_schedule_keys=reward_reference_schedule_keys,
        source_weights=source_weights,
    )
    aggregate_rows = source_balanced_seed_mean_rows(source_rows, source_weights=source_weights)
    summary_grids = _load_schedule_summary_grids(args.reference_schedule_summary)
    summary_grids.update(_load_schedule_summary_grids_many(candidate_summary_paths))

    teacher_training_keys = set(candidate_keys).union(teacher_fixed_schedule_keys)
    teacher_rows = [row for row in aggregate_rows if str(row["scheduler_key"]) in teacher_training_keys]
    if not teacher_rows:
        raise ValueError("V4.2-F teacher fitting requires BO candidate rows or selected fixed teacher demos.")
    teacher_fit_rows, teacher_diagnostic_rows = split_teacher_diagnostic_holdout(
        teacher_rows,
        fraction=float(args.teacher_diagnostic_holdout_fraction),
        seed=int(args.seed),
        fixed_schedule_keys=BASELINE_SCHEDULE_KEYS,
        rewards=rewards,
    )
    schedule_weights = teacher_schedule_weights(
        sorted({str(row["scheduler_key"]) for row in teacher_rows}),
        late_biased_demo_schedules=tuple(_parse_csv(args.late_biased_demo_schedules)),
        late_biased_demo_weight=float(args.late_biased_demo_weight),
    )
    seed_all(int(args.seed))
    teacher_training = train_teacher(
        teacher,
        teacher_fit_rows,
        rewards=rewards,
        summary_grids=summary_grids,
        max_macro_steps=max_macro_steps,
        steps=int(args.teacher_steps),
        lr=float(args.lr),
        schedule_weights=schedule_weights,
        rank_temperature=float(args.teacher_rank_temperature),
        regression_weight=float(args.teacher_regression_weight),
        pairs_per_candidate=int(args.teacher_pairs_per_candidate),
        pair_margin=float(args.teacher_pair_margin),
        diagnostic_rows=teacher_diagnostic_rows,
        diagnostic_top_k=int(args.teacher_diagnostic_top_k),
    )

    settings = [(str(solver), int(target_nfe)) for solver in solvers for target_nfe in target_nfes]
    uniform_targets = _uniform_reference_targets(settings)
    fixed_states: Dict[int, List[Mapping[str, torch.Tensor]]] = {int(budget): [] for budget in budgets}
    best_states: List[Mapping[str, torch.Tensor]] = []
    best_steps: Dict[str, int] = {}
    best_losses: Dict[str, float] = {}
    init_losses_by_seed: Dict[str, List[Dict[str, float]]] = {}
    opd_losses_by_seed: Dict[str, List[Dict[str, float]]] = {}
    checkpoint_state_payload: Dict[str, Any] = {"fixed_epoch": {}, "lowest_internal_loss": {}}
    for student_seed in student_seeds:
        seed_all(int(student_seed))
        student = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
        init_losses = _fit_student_to_reference(
            student,
            uniform_targets,
            steps=int(args.student_init_steps),
            lr=float(args.lr),
            loss_name="uniform_init_mse",
        )
        opt_summary = _optimize_student_with_teacher_checkpoints(
            student,
            teacher,
            settings,
            max_macro_steps=max_macro_steps,
            max_steps=max(budgets),
            lr=float(args.lr),
            checkpoint_steps=budgets,
        )
        init_losses_by_seed[str(student_seed)] = init_losses
        opd_losses_by_seed[str(student_seed)] = list(opt_summary["losses"])
        for budget, states in fixed_states.items():
            states.append(opt_summary["states_by_step"][str(int(budget))])
        best_states.append(opt_summary["best_state"])
        best_steps[str(student_seed)] = int(opt_summary["best_step"])
        best_losses[str(student_seed)] = float(opt_summary["best_loss"])
        checkpoint_state_payload["lowest_internal_loss"][str(student_seed)] = opt_summary["best_state"]
        checkpoint_state_payload["fixed_epoch"][str(student_seed)] = opt_summary["states_by_step"][str(int(args.fixed_epoch_steps))]

    schedules: List[Dict[str, Any]] = []
    prefix = str(args.student_schedule_key_prefix).strip() or DEFAULT_STUDENT_PREFIX
    fixed_step = int(args.fixed_epoch_steps)
    if "fixed_epoch" in checkpoint_modes:
        predictions = _seed_mean_predictions_from_states(
            fixed_states[fixed_step],
            solvers=solvers,
            target_nfes=target_nfes,
            setting_dim=setting_dim,
            max_macro_steps=max_macro_steps,
            teacher=teacher,
            checkpoint_mode="fixed_epoch",
            seed_values=student_seeds,
            checkpoint_steps_by_seed={str(seed): fixed_step for seed in student_seeds},
        )
        schedules.append(
            {
                "scheduler_key": f"{prefix}fixed_epoch_{fixed_step}",
                "schedule_name": f"V4.2-F uniform-init seed-mean student fixed epoch {fixed_step}",
                "comparison_role": "learned_student_seed_mean_internal_checkpoint",
                "student_checkpoint_mode": "fixed_epoch",
                "fixed_epoch_steps": fixed_step,
                "student_initialization": "uniform",
                "student_seed_values": list(student_seeds),
                "teacher_predicted_utility_mean": float(np.mean([float(item["utility"]) for item in predictions])),
                "predictions": predictions,
            }
        )
    if "lowest_internal_loss" in checkpoint_modes:
        predictions = _seed_mean_predictions_from_states(
            best_states,
            solvers=solvers,
            target_nfes=target_nfes,
            setting_dim=setting_dim,
            max_macro_steps=max_macro_steps,
            teacher=teacher,
            checkpoint_mode="lowest_internal_loss",
            seed_values=student_seeds,
            checkpoint_steps_by_seed=best_steps,
        )
        schedules.append(
            {
                "scheduler_key": f"{prefix}lowest_internal_loss",
                "schedule_name": "V4.2-F uniform-init seed-mean student lowest internal loss",
                "comparison_role": "learned_student_seed_mean_internal_checkpoint",
                "student_checkpoint_mode": "lowest_internal_loss",
                "student_internal_best_steps_by_seed": dict(best_steps),
                "student_internal_best_losses_by_seed": dict(best_losses),
                "student_initialization": "uniform",
                "student_seed_values": list(student_seeds),
                "teacher_predicted_utility_mean": float(np.mean([float(item["utility"]) for item in predictions])),
                "predictions": predictions,
            }
        )

    schedule_summary = {
        "status": "ready",
        "artifact": "v42_f_seed_mean_student_schedule_summary",
        "dataset": str(args.dataset),
        "protocol": "v4.2_f_final_calibration_set",
        "baseline_schedule": False,
        "student_initialization": "uniform",
        "student_checkpoint_modes": list(checkpoint_modes),
        "final_checkpoint_mode": str(args.final_checkpoint_mode),
        "student_seed_values": list(student_seeds),
        "schedules": schedules,
    }
    save_json(schedule_summary, str(out_dir / "student_seed_mean_schedule_summary.json"))
    selected_key = next(schedule["scheduler_key"] for schedule in schedules if schedule["student_checkpoint_mode"] == str(args.final_checkpoint_mode))
    selection = {
        "status": "ready",
        "selection_protocol": "v4.2_f_predeclared_internal_checkpoint_seed_mean",
        "selection_mode": "predeclared_internal_checkpoint",
        "uses_validation_labels_for_selection": False,
        "uses_locked_test_for_selection": False,
        "selected_schedule_key": selected_key,
        "final_checkpoint_mode": str(args.final_checkpoint_mode),
        "fixed_epoch_steps": fixed_step,
        "student_seed_values": list(student_seeds),
        "student_internal_best_steps_by_seed": dict(best_steps),
        "student_internal_best_losses_by_seed": dict(best_losses),
    }
    save_json(selection, str(out_dir / "internal_checkpoint_selection.json"))
    selected_summary = _write_selected_seed_mean_summary(
        schedule_summary,
        selection=selection,
        out_path=out_dir / "selected_schedule_summary.json",
    )

    reward_reference_table = build_fixed_reference_table(
        source_balanced_seed_mean_rows(
            {CALIBRATION_TRAIN_SOURCE: train_reference_rows, CALIBRATION_VAL_SOURCE: val_reference_rows},
            source_weights=source_weights,
        ),
        fixed_schedule_keys=reward_reference_schedule_keys,
    )
    summary = {
        "status": "ready",
        "protocol": "v4.2_f_final_calibration_set",
        "dataset": str(args.dataset),
        "calibration_sources": [CALIBRATION_TRAIN_SOURCE, CALIBRATION_VAL_SOURCE],
        "calibration_source_weights": source_weights,
        "uses_validation_selection": False,
        "locked_test_policy": "run_once_after_predeclared_seed_mean_schedule_is_frozen",
        "student_initialization": "uniform",
        "student_checkpoint_modes": list(checkpoint_modes),
        "final_checkpoint_mode": str(args.final_checkpoint_mode),
        "fixed_epoch_steps": fixed_step,
        "student_seed_values": list(student_seeds),
        "teacher_fixed_schedule_keys": list(teacher_fixed_schedule_keys),
        "reward_reference_schedule_keys": list(reward_reference_schedule_keys),
        "teacher_supervision_sources": {
            "calibration_train_fixed_seed_rows": int(len(train_reference_rows)),
            "calibration_val_fixed_seed_rows": int(len(val_reference_rows)),
            "calibration_train_candidate_seed_rows": int(len(train_candidate_rows)),
            "calibration_val_candidate_seed_rows": int(len(val_candidate_rows)),
            "source_balanced_aggregate_rows": int(len(aggregate_rows)),
            "teacher_training_rows": int(len(teacher_rows)),
            "teacher_fit_rows": int(len(teacher_fit_rows)),
            "teacher_diagnostic_holdout_rows": int(len(teacher_diagnostic_rows)),
            "candidate_schedule_count": int(len(candidate_keys)),
        },
        "teacher_objective": "source_balanced_paired_best_fixed_composite_ranking_huber",
        "teacher_selection_protocol": "v4.2_f_internal_bo_row_holdout_no_terminal_validation",
        "teacher_checkpoint_selection": dict(teacher_training["checkpoint_selection"]),
        "teacher_diagnostics": dict(teacher_training["selected_diagnostics"]),
        "teacher_losses": list(teacher_training["losses"]),
        "teacher_parameters": int(teacher_param_count),
        "student_parameters": int(student_param_count),
        "student_initialization_losses": init_losses_by_seed,
        "student_opd_losses": opd_losses_by_seed,
        "student_internal_best_steps_by_seed": dict(best_steps),
        "student_internal_best_losses_by_seed": dict(best_losses),
        "reward_reference_calibration": [dict(value) for _, value in sorted(reward_reference_table.items(), key=lambda item: item[0])],
        "student_seed_mean_schedule_summary": str(out_dir / "student_seed_mean_schedule_summary.json"),
        "selected_schedule_summary": str(out_dir / "selected_schedule_summary.json"),
        "selected_summary_schedule_count": int(len(selected_summary.get("schedules", []))),
    }
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "checkpoint_student_states": checkpoint_state_payload,
            "setting_dim": int(setting_dim),
            "teacher_input_dim": int(teacher_input_dim),
            "max_macro_steps": int(max_macro_steps),
            "protocol": "v4.2_f_final_calibration_set",
        },
        out_dir / "conditional_opd_v42_f.pt",
    )
    save_json(summary, str(out_dir / "conditional_opd_v42_f_summary.json"))
    save_json(summary, str(out_dir / "conditional_opd_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the V4.2-F final calibration-set neural conditional OPD policy.")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--solver_names", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--target_nfe_values", default=",".join(str(x) for x in DEFAULT_TARGET_NFES))
    parser.add_argument("--train_rows_csv", default="")
    parser.add_argument("--validation_rows_csv", default=str(project_outputs_root() / "diffusion_flow_time_reparameterization_full_val" / "rows.csv"))
    parser.add_argument("--candidate_train_rows_csv", default="")
    parser.add_argument("--candidate_validation_rows_csv", default="")
    parser.add_argument("--candidate_schedule_summary", default="")
    parser.add_argument("--reference_schedule_summary", default="")
    parser.add_argument("--calibration_train_seeds", default="0,1")
    parser.add_argument("--calibration_val_seeds", default="0,1,2")
    parser.add_argument("--calibration_train_weight", type=float, default=0.5)
    parser.add_argument("--calibration_val_weight", type=float, default=0.5)
    parser.add_argument("--teacher_fixed_schedule_keys", default="none")
    parser.add_argument("--expected_train_tuning_sampler", default="")
    parser.add_argument("--expected_train_tuning_fraction", default="")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR / "neural_policy"))
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--teacher_rank_temperature", type=float, default=DEFAULT_TEACHER_RANK_TEMPERATURE)
    parser.add_argument("--teacher_regression_weight", type=float, default=DEFAULT_TEACHER_REGRESSION_WEIGHT)
    parser.add_argument("--teacher_pairs_per_candidate", type=int, default=DEFAULT_TEACHER_PAIRS_PER_CANDIDATE)
    parser.add_argument("--teacher_pair_margin", type=float, default=DEFAULT_TEACHER_PAIR_MARGIN)
    parser.add_argument("--teacher_diagnostic_top_k", type=int, default=5)
    parser.add_argument("--teacher_diagnostic_holdout_fraction", type=float, default=DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION)
    parser.add_argument("--student_init_steps", type=int, default=500)
    parser.add_argument("--student_opd_step_values", default="5,10,15,20,25")
    parser.add_argument("--student_seeds", default="0,1,2")
    parser.add_argument("--student_checkpoint_modes", default=",".join(DEFAULT_STUDENT_CHECKPOINT_MODES))
    parser.add_argument("--fixed_epoch_steps", type=int, default=20)
    parser.add_argument("--final_checkpoint_mode", choices=DEFAULT_STUDENT_CHECKPOINT_MODES, default="lowest_internal_loss")
    parser.add_argument("--student_schedule_key_prefix", default=DEFAULT_STUDENT_PREFIX)
    parser.add_argument("--late_biased_demo_schedules", default=",".join(DEFAULT_LATE_BIASED_DEMO_SCHEDULES))
    parser.add_argument("--late_biased_demo_weight", type=float, default=DEFAULT_LATE_BIASED_DEMO_WEIGHT)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser


def main() -> None:
    summary = train_conditional_opd_v42_f(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
