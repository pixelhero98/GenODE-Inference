from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.conditional_opd.candidate_pool import (
    _family_hash,
    _prediction_from_grid,
    load_schedules,
)
from genode.conditional_opd.clock_lowrank import (
    DEFAULT_DENSITY_GRID_SIZE,
    LOWRANK_THETA_DIM,
    hand_bump_thetas,
    schedule_grid_from_theta,
    sobol_thetas,
    theta_metadata,
    validate_theta,
    zero_theta,
)
from genode.conditional_opd.models import solver_macro_steps
from genode.conditional_opd.objectives import (
    rewards_by_setting,
    seed_mean_metric_rows,
    source_balanced_rewards_by_setting,
    source_balanced_seed_mean_rows,
)
from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.data.otflow_paths import project_outputs_root, resolve_project_path
from genode.models.otflow_train_val import save_json
from genode.runtime import ProgressBar
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


DEFAULT_BO_CANDIDATE_COUNT = 16
DEFAULT_BO_SOBOL_POOL = 512
DEFAULT_BO_THETA_BOUND = 2.5


def require_botorch() -> Dict[str, str]:
    try:
        import botorch
        import gpytorch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:  # pragma: no cover - exercised when dependency is missing
        raise RuntimeError("BoTorch and GPyTorch are required for pooled BO candidate generation.") from exc
    return {
        "botorch": str(botorch.__version__),
        "gpytorch": str(gpytorch.__version__),
        "SingleTaskGP": SingleTaskGP.__name__,
        "LogExpectedImprovement": LogExpectedImprovement.__name__,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood.__name__,
        "fit_gpytorch_mll": fit_gpytorch_mll.__name__,
    }


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _read_rows_csv(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part in _parse_csv(str(path)):
        resolved = resolve_project_path(str(part))
        if not resolved.exists():
            continue
        with resolved.open("r", newline="", encoding="utf-8") as fh:
            rows.extend(dict(row) for row in csv.DictReader(fh))
    return rows


def _load_reference_grids(path: str | Path) -> Dict[Tuple[str, int], Tuple[float, ...]]:
    resolved = resolve_project_path(str(path))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, int], Tuple[float, ...]] = {}
    for schedule in payload.get("schedules", []) or []:
        if str(schedule.get("scheduler_key")) != SER_PTG_SCHEDULE_KEY:
            continue
        for item in schedule.get("predictions", []) or []:
            out[(str(item["solver_key"]), int(item["target_nfe"]))] = tuple(float(x) for x in item["time_grid"])
    if not out:
        raise ValueError(f"No SER-PTG reference grids found in {resolved}.")
    return out


def _settings(reference_grids: Mapping[Tuple[str, int], Sequence[float]]) -> List[Tuple[str, int]]:
    return sorted(reference_grids, key=lambda item: (item[0], item[1]))


def _source_item(*, solver: str, target_nfe: int, max_macro_steps: int) -> Dict[str, Any]:
    macro_steps = solver_macro_steps(str(solver), int(target_nfe))
    return {
        "solver_key": str(solver),
        "target_nfe": int(target_nfe),
        "macro_steps": int(macro_steps),
        "runtime_nfe": int(macro_steps),
        "max_macro_steps": int(max_macro_steps),
        "candidate_source": "bayesian_optimization",
        "student_seed": "",
        "opd_steps": "",
        "opd_step_budget": "",
    }


def _schedule_from_cell_thetas(
    *,
    scheduler_key: str,
    active_round: int,
    seed: int,
    cell_thetas: Mapping[Tuple[str, int], Sequence[float]],
    reference_grids: Mapping[Tuple[str, int], Sequence[float]],
    theta_source: str,
    acquisition_metadata: Mapping[str, Any],
    density_grid_size: int,
) -> Dict[str, Any]:
    max_macro_steps = max(solver_macro_steps(solver, nfe) for solver, nfe in reference_grids)
    predictions: List[Dict[str, Any]] = []
    for solver, target_nfe in _settings(reference_grids):
        theta = validate_theta(cell_thetas[(solver, target_nfe)])
        macro_steps = solver_macro_steps(solver, target_nfe)
        grid = schedule_grid_from_theta(
            theta,
            macro_steps=macro_steps,
            base_grid=reference_grids[(solver, target_nfe)],
            grid_size=int(density_grid_size),
        )
        metadata = theta_metadata(theta, source=theta_source, grid_size=int(density_grid_size))
        metadata.update(dict(acquisition_metadata))
        pred = _prediction_from_grid(
            source=_source_item(solver=solver, target_nfe=target_nfe, max_macro_steps=max_macro_steps),
            grid=grid,
            scheduler_key=scheduler_key,
            candidate_source="bayesian_optimization",
            active_round=int(active_round),
            perturbation_type="lowrank_bo",
            perturbation_params=metadata,
        )
        pred["bo_theta_json"] = json.dumps(metadata["theta"], separators=(",", ":"))
        pred["bo_theta_hash"] = str(metadata["theta_hash"])
        pred["bo_theta_source"] = str(theta_source)
        pred["density_grid_size"] = int(density_grid_size)
        predictions.append(pred)
    schedule = {
        "scheduler_key": str(scheduler_key),
        "schedule_name": f"V4.3 pooled BO {scheduler_key}",
        "comparison_role": "v43_pooled_bo_candidate",
        "candidate_source": "bayesian_optimization",
        "active_round": int(active_round),
        "seed": int(seed),
        "perturbation_type": "lowrank_bo",
        "perturbation_params_json": json.dumps(dict(acquisition_metadata), sort_keys=True, separators=(",", ":")),
        "density_grid_size": int(density_grid_size),
        "predictions": predictions,
    }
    schedule["full_family_grid_hash"] = _family_hash(schedule)
    return schedule


def _candidate_schedule_thetas(schedule: Mapping[str, Any]) -> Dict[Tuple[str, int], Tuple[float, ...]]:
    out: Dict[Tuple[str, int], Tuple[float, ...]] = {}
    for item in schedule.get("predictions", []) or []:
        raw = item.get("bo_theta_json")
        if not raw:
            params = json.loads(str(item.get("perturbation_params_json", "{}") or "{}"))
            raw = params.get("theta", [])
        theta = validate_theta(json.loads(raw) if isinstance(raw, str) else raw)
        out[(str(item["solver_key"]), int(item["target_nfe"]))] = theta
    return out


def _observations_by_cell(
    *,
    observed_schedule_summaries: Sequence[str | Path],
    candidate_rows: Sequence[Mapping[str, Any]],
    fixed_reference_rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int], List[Tuple[Tuple[float, ...], float]]]:
    if not observed_schedule_summaries or not candidate_rows:
        return {}
    _, schedules = load_schedules(observed_schedule_summaries)
    theta_by_key: Dict[str, Dict[Tuple[str, int], Tuple[float, ...]]] = {
        str(schedule["scheduler_key"]): _candidate_schedule_thetas(schedule)
        for schedule in schedules
        if str(schedule.get("candidate_source")) == "bayesian_optimization"
    }
    rewards = rewards_by_setting(
        [*seed_mean_metric_rows(fixed_reference_rows), *seed_mean_metric_rows(candidate_rows)],
        fixed_schedule_keys=BASELINE_SCHEDULE_KEYS,
    )
    out: Dict[Tuple[str, int], List[Tuple[Tuple[float, ...], float]]] = {}
    for row in seed_mean_metric_rows(candidate_rows):
        schedule_key = str(row["scheduler_key"])
        setting = (str(row["solver_key"]), int(row["target_nfe"]))
        if schedule_key not in theta_by_key or setting not in theta_by_key[schedule_key]:
            continue
        if setting not in rewards or schedule_key not in rewards[setting]:
            continue
        out.setdefault(setting, []).append((theta_by_key[schedule_key][setting], float(rewards[setting][schedule_key])))
    return out



def _observations_by_cell_source_balanced(
    *,
    observed_schedule_summaries: Sequence[str | Path],
    candidate_rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    fixed_reference_rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[Tuple[str, int], List[Tuple[Tuple[float, ...], float]]]:
    if not observed_schedule_summaries:
        return {}
    if not candidate_rows_by_source or not all(candidate_rows_by_source.values()):
        return {}
    _, schedules = load_schedules(observed_schedule_summaries)
    theta_by_key: Dict[str, Dict[Tuple[str, int], Tuple[float, ...]]] = {
        str(schedule["scheduler_key"]): _candidate_schedule_thetas(schedule)
        for schedule in schedules
        if str(schedule.get("candidate_source")) == "bayesian_optimization"
    }
    source_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for source_name, candidate_rows in candidate_rows_by_source.items():
        fixed_rows = list(fixed_reference_rows_by_source.get(str(source_name), []) or [])
        if not fixed_rows:
            raise ValueError(f"Missing fixed reference rows for calibration source {source_name!r}.")
        source_rows[str(source_name)] = [*fixed_rows, *list(candidate_rows)]
    rewards = source_balanced_rewards_by_setting(
        source_rows,
        fixed_schedule_keys=BASELINE_SCHEDULE_KEYS,
        source_weights={name: 1.0 for name in source_rows},
    )
    aggregate_candidate_rows = source_balanced_seed_mean_rows(
        {name: list(rows) for name, rows in candidate_rows_by_source.items()},
        source_weights={name: 1.0 for name in candidate_rows_by_source},
    )
    out: Dict[Tuple[str, int], List[Tuple[Tuple[float, ...], float]]] = {}
    for row in aggregate_candidate_rows:
        schedule_key = str(row["scheduler_key"])
        setting = (str(row["solver_key"]), int(row["target_nfe"]))
        if schedule_key not in theta_by_key or setting not in theta_by_key[schedule_key]:
            continue
        if setting not in rewards or schedule_key not in rewards[setting]:
            continue
        out.setdefault(setting, []).append((theta_by_key[schedule_key][setting], float(rewards[setting][schedule_key])))
    return out


def _select_diverse_rows(thetas: np.ndarray, scores: np.ndarray, *, count: int) -> List[int]:
    order = list(np.argsort(-scores))
    selected: List[int] = []
    for idx in order:
        if len(selected) >= int(count):
            break
        if not selected:
            selected.append(int(idx))
            continue
        distances = np.asarray([np.mean(np.abs(thetas[int(idx)] - thetas[j])) for j in selected], dtype=np.float64)
        if float(np.min(distances)) >= 0.05 or len(order) <= int(count):
            selected.append(int(idx))
    for idx in order:
        if len(selected) >= int(count):
            break
        if int(idx) not in selected:
            selected.append(int(idx))
    return selected[: int(count)]


def _propose_cell_thetas(
    observations: Sequence[Tuple[Sequence[float], float]],
    *,
    count: int,
    seed: int,
    bound: float,
    sobol_pool: int,
) -> Tuple[List[Tuple[float, ...]], Dict[str, Any]]:
    versions = require_botorch()
    if len(observations) < 3:
        return sobol_thetas(count, seed=int(seed), bound=float(bound)), {
            "acquisition": "sobol_warmup",
            "observation_count": int(len(observations)),
            "botorch_versions": versions,
        }
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_x_np = np.asarray([row[0] for row in observations], dtype=np.float64)
    train_y_np = np.asarray([[float(row[1])] for row in observations], dtype=np.float64)
    train_x_unit = (train_x_np + float(bound)) / (2.0 * float(bound))
    train_x = torch.tensor(train_x_unit, dtype=torch.double)
    y_mean = float(np.mean(train_y_np))
    y_std = float(np.std(train_y_np))
    if y_std <= 1e-9:
        y_std = 1.0
    train_y = torch.tensor((train_y_np - y_mean) / y_std, dtype=torch.double)
    try:
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        acq = LogExpectedImprovement(model, best_f=float(train_y.max().detach().cpu().item()))
        pool_unit = torch.quasirandom.SobolEngine(dimension=LOWRANK_THETA_DIM, scramble=True, seed=int(seed)).draw(int(sobol_pool)).double()
        with torch.no_grad():
            scores = acq(pool_unit[:, None, :]).detach().cpu().numpy().astype(np.float64)
        pool_theta = (2.0 * pool_unit.detach().cpu().numpy().astype(np.float64) - 1.0) * float(bound)
        selected = _select_diverse_rows(pool_theta, scores, count=int(count))
        return [tuple(float(x) for x in pool_theta[idx].tolist()) for idx in selected], {
            "acquisition": "botorch_log_expected_improvement",
            "observation_count": int(len(observations)),
            "candidate_pool_size": int(sobol_pool),
            "y_mean": y_mean,
            "y_std": y_std,
            "botorch_versions": versions,
        }
    except Exception as exc:
        return sobol_thetas(count, seed=int(seed), bound=float(bound)), {
            "acquisition": "sobol_after_botorch_fit_error",
            "observation_count": int(len(observations)),
            "fit_error": f"{type(exc).__name__}: {exc}",
            "botorch_versions": versions,
        }


def build_bo_candidate_pool(
    *,
    dataset: str = "san_francisco_traffic",
    reference_schedule_summary: str | Path,
    active_round: int,
    seed: int,
    candidate_count: int = DEFAULT_BO_CANDIDATE_COUNT,
    observed_schedule_summaries: Sequence[str | Path] = (),
    observed_rows: Sequence[Mapping[str, Any]] = (),
    fixed_reference_rows: Sequence[Mapping[str, Any]] = (),
    observed_rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    fixed_reference_rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    density_grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
    theta_bound: float = DEFAULT_BO_THETA_BOUND,
    sobol_pool: int = DEFAULT_BO_SOBOL_POOL,
) -> Dict[str, Any]:
    require_botorch()
    reference_grids = _load_reference_grids(reference_schedule_summary)
    if observed_rows_by_source and fixed_reference_rows_by_source:
        observations = _observations_by_cell_source_balanced(
            observed_schedule_summaries=observed_schedule_summaries,
            candidate_rows_by_source=observed_rows_by_source,
            fixed_reference_rows_by_source=fixed_reference_rows_by_source,
        )
    else:
        observations = _observations_by_cell(
            observed_schedule_summaries=observed_schedule_summaries,
            candidate_rows=observed_rows,
            fixed_reference_rows=fixed_reference_rows,
        )
    settings = _settings(reference_grids)
    per_cell: Dict[Tuple[str, int], List[Tuple[float, ...]]] = {}
    acquisition: Dict[str, Any] = {}
    with ProgressBar(len(settings), "BO exploration cells") as cell_progress:
        for solver, target_nfe in settings:
            if int(active_round) == 0:
                designs = [zero_theta(), *hand_bump_thetas(), *sobol_thetas(max(0, int(candidate_count) - 7), seed=int(seed) + target_nfe)]
                meta = {"acquisition": "ser_zero_hand_bump_sobol_warmup", "observation_count": 0}
            else:
                designs, meta = _propose_cell_thetas(
                    observations.get((solver, target_nfe), []),
                    count=int(candidate_count),
                    seed=int(seed) + 1009 * (settings.index((solver, target_nfe)) + 1),
                    bound=float(theta_bound),
                    sobol_pool=int(sobol_pool),
                )
            per_cell[(solver, target_nfe)] = designs[: int(candidate_count)]
            acquisition[f"{solver}:{target_nfe}"] = meta
            cell_progress.update()
    schedules: List[Dict[str, Any]] = []
    with ProgressBar(int(candidate_count), "BO candidate families") as family_progress:
        for idx in range(int(candidate_count)):
            cell_thetas = {setting: per_cell[setting][idx % len(per_cell[setting])] for setting in settings}
            source = "warmup" if int(active_round) == 0 else "bo"
            schedules.append(
                _schedule_from_cell_thetas(
                    scheduler_key=f"train20_v43_bo_r{int(active_round)}_cand{idx:03d}",
                    active_round=int(active_round),
                    seed=int(seed),
                    cell_thetas=cell_thetas,
                    reference_grids=reference_grids,
                    theta_source=source,
                    acquisition_metadata={"bo_round": int(active_round), "candidate_index": int(idx), "theta_bound": float(theta_bound)},
                    density_grid_size=int(density_grid_size),
                )
            )
            family_progress.update()
    return {
        "status": "ready",
        "artifact": "v43_pooled_bo_candidate_pool_schedule_summary",
        "dataset": str(dataset),
        "active_round": int(active_round),
        "seed": int(seed),
        "candidate_source": "bayesian_optimization",
        "density_grid_size": int(density_grid_size),
        "theta_dim": LOWRANK_THETA_DIM,
        "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
        "reference_schedule_summary": str(resolve_project_path(str(reference_schedule_summary))),
        "bo_acquisition": acquisition,
        "schedule_count": int(len(schedules)),
        "schedules": schedules,
    }


def build_and_write_bo_candidate_pool(args: argparse.Namespace) -> Dict[str, Any]:
    observed_rows_by_source: Dict[str, Sequence[Mapping[str, Any]]] = {}
    fixed_reference_rows_by_source: Dict[str, Sequence[Mapping[str, Any]]] = {}
    observed_train_csv = str(getattr(args, "observed_train_rows_csv", "")).strip()
    observed_validation_csv = str(getattr(args, "observed_validation_rows_csv", "")).strip()
    if observed_train_csv or observed_validation_csv:
        if not observed_train_csv or not observed_validation_csv:
            raise ValueError("Source-balanced BO observations require both observed train and validation rows.")
        fixed_train_csv = str(getattr(args, "fixed_train_rows_csv", "")).strip()
        fixed_validation_csv = str(getattr(args, "fixed_validation_rows_csv", "")).strip()
        if not fixed_train_csv or not fixed_validation_csv:
            raise ValueError("Source-balanced BO observations require both fixed train and validation reference rows.")
        observed_rows_by_source = {
            "calibration_train_part": _read_rows_csv(observed_train_csv),
            "calibration_val_part": _read_rows_csv(observed_validation_csv),
        }
        fixed_reference_rows_by_source = {
            "calibration_train_part": _read_rows_csv(fixed_train_csv),
            "calibration_val_part": _read_rows_csv(fixed_validation_csv),
        }
    pool = build_bo_candidate_pool(
        dataset=str(args.dataset),
        reference_schedule_summary=str(args.reference_schedule_summary),
        active_round=int(args.active_round),
        seed=int(args.seed),
        candidate_count=int(args.candidate_count),
        observed_schedule_summaries=_parse_csv(str(args.observed_schedule_summaries)),
        observed_rows=_read_rows_csv(args.observed_rows_csv),
        fixed_reference_rows=_read_rows_csv(args.fixed_reference_rows_csv),
        observed_rows_by_source=observed_rows_by_source,
        fixed_reference_rows_by_source=fixed_reference_rows_by_source,
        density_grid_size=int(args.density_grid_size),
        theta_bound=float(args.theta_bound),
        sobol_pool=int(args.sobol_pool),
    )
    out_path = resolve_project_path(str(args.out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(pool, str(out_path))
    return {"status": "ready", "candidate_pool": str(out_path), "schedule_count": int(pool["schedule_count"])}

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V4.3 pooled BoTorch BO schedule-family candidates.")
    parser.add_argument("--mode", choices=("generate",), default="generate")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--reference_schedule_summary", default="")
    parser.add_argument("--observed_schedule_summaries", default="")
    parser.add_argument("--observed_rows_csv", default="")
    parser.add_argument("--fixed_reference_rows_csv", default="")
    parser.add_argument("--observed_train_rows_csv", default="")
    parser.add_argument("--observed_validation_rows_csv", default="")
    parser.add_argument("--fixed_train_rows_csv", default="")
    parser.add_argument("--fixed_validation_rows_csv", default="")
    parser.add_argument("--out_path", default=str(project_outputs_root() / "train20_v43_pooled_bo" / "candidate_pool_schedule_summary.json"))
    parser.add_argument("--active_round", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate_count", type=int, default=DEFAULT_BO_CANDIDATE_COUNT)
    parser.add_argument("--density_grid_size", type=int, default=DEFAULT_DENSITY_GRID_SIZE)
    parser.add_argument("--theta_bound", type=float, default=DEFAULT_BO_THETA_BOUND)
    parser.add_argument("--sobol_pool", type=int, default=DEFAULT_BO_SOBOL_POOL)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = build_and_write_bo_candidate_pool(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
