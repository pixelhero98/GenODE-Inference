from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.data.molecule_xyz import (
    ATOM_COVALENT_RADIUS,
    DEFAULT_MOLECULE_DATASET_KEY,
    build_molecule_dataset_splits,
    default_molecule_processed_dir,
    kabsch_aligned_rmsd,
    load_molecule_group_manifest,
    molecule_stats_from_mapping,
)
from genode.canonical_experiment_layout import NFE_ROLE_SEEN, SCENARIO_FAMILY_MOLECULE, canonical_nfes_for_role
from genode.data.otflow_paths import project_root, resolve_project_path
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.evaluation.otflow_sampling_support import _apply_sample_overrides, _restore_sample_overrides
from genode.gipo.models import validate_time_grid
from genode.models.otflow_train_val import _temporary_eval_seed, evaluate_average_loss, save_json
from genode.runtime import resolve_torch_device
from genode.solver_protocol import (
    CANONICAL_SOLVER_KEYS,
    expected_realized_nfe,
    normalize_solver_key,
    normalize_solver_keys,
    solver_macro_steps,
)

MOLECULE_CONTEXT_SCHEMA = "molecule_3d_window"
MOLECULE_PRIMARY_METRICS: Tuple[str, ...] = (
    "molecule_kabsch_rmsd_3d",
    "molecule_ensemble_velocity_norm_w1",
    "molecule_ensemble_acceleration_norm_w1",
    "molecule_rollout_velocity_norm_w1",
    "molecule_rollout_acceleration_norm_w1",
)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))


def _project_display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return resolved.name


def _safe_mean(values: Sequence[Any]) -> float:
    nums: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            nums.append(float(numeric))
    if not nums:
        return float("nan")
    return float(np.mean(np.asarray(nums, dtype=np.float64)))


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) == len(y):
        return float(np.mean(np.abs(x - y)))
    qs = np.linspace(0.0, 1.0, max(len(x), len(y)), dtype=np.float64)
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def _coordinate_cw1(pred_rows: np.ndarray, true_rows: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_rows, dtype=np.float64)
    true = np.asarray(true_rows, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return {
            "molecule_coordinate_w1_mean": float("nan"),
            "molecule_coordinate_w1_median": float("nan"),
            "molecule_coordinate_w1_max": float("nan"),
        }
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError(f"Expected matching [samples, dims] coordinate arrays, got {pred.shape} and {true.shape}.")
    values = np.mean(np.abs(np.sort(pred, axis=0) - np.sort(true, axis=0)), axis=0)
    return {
        "molecule_coordinate_w1_mean": float(np.mean(values)),
        "molecule_coordinate_w1_median": float(np.median(values)),
        "molecule_coordinate_w1_max": float(np.max(values)),
    }


def _radius_of_gyration(coords: np.ndarray) -> float:
    centered = coords - coords.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _bond_pairs(reference_coords: np.ndarray, atom_symbols: Sequence[str]) -> np.ndarray:
    pairs = []
    distances = _pairwise_distances(reference_coords)
    for i in range(len(atom_symbols)):
        for j in range(i + 1, len(atom_symbols)):
            cutoff = 1.25 * (ATOM_COVALENT_RADIUS[str(atom_symbols[i])] + ATOM_COVALENT_RADIUS[str(atom_symbols[j])])
            if 0.35 < float(distances[i, j]) <= cutoff:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64)


def _validity_metrics(pred: np.ndarray, true: np.ndarray, atom_symbols: Sequence[str], bond_pairs: np.ndarray) -> Dict[str, float]:
    finite = float(np.isfinite(pred).all())
    pred_dist = _pairwise_distances(pred)
    clash_count = 0
    pair_count = 0
    for i in range(len(atom_symbols)):
        for j in range(i + 1, len(atom_symbols)):
            pair_count += 1
            min_dist = 0.60 * (ATOM_COVALENT_RADIUS[str(atom_symbols[i])] + ATOM_COVALENT_RADIUS[str(atom_symbols[j])])
            if float(pred_dist[i, j]) < min_dist:
                clash_count += 1
    if len(bond_pairs) > 0:
        true_dist = _pairwise_distances(true)
        bond_errors = [
            abs(float(pred_dist[int(i), int(j)]) - float(true_dist[int(i), int(j)]))
            for i, j in bond_pairs
        ]
        bond_violation = float(np.mean(np.asarray(bond_errors) > 0.20))
    else:
        bond_violation = 0.0
    return {
        "finite_rate": finite,
        "clash_rate": float(clash_count / max(1, pair_count)),
        "bond_contact_violation_rate": bond_violation,
    }


def molecule_coordinate_metrics(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    *,
    atom_symbols: Sequence[str],
    bond_pairs: np.ndarray,
) -> Dict[str, float]:
    pred = np.asarray(pred_coords, dtype=np.float32)
    true = np.asarray(true_coords, dtype=np.float32)
    diff = pred - true
    pred_pair = _upper_triangle_values(_pairwise_distances(pred))
    true_pair = _upper_triangle_values(_pairwise_distances(true))
    pair_diff = pred_pair - true_pair
    validity = _validity_metrics(pred, true, atom_symbols, bond_pairs)
    return {
        "molecule_kabsch_rmsd_3d": kabsch_aligned_rmsd(pred, true),
        "raw_coord_mae": float(np.mean(np.abs(diff))),
        "raw_coord_rmse": float(np.sqrt(np.mean(diff * diff))),
        "pairwise_distance_mae": float(np.mean(np.abs(pair_diff))),
        "pairwise_distance_rmse": float(np.sqrt(np.mean(pair_diff * pair_diff))),
        "radius_gyration_abs_error": abs(_radius_of_gyration(pred) - _radius_of_gyration(true)),
        **validity,
    }


def molecule_distributional_metrics(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    current_coords: np.ndarray,
    previous_coords: Optional[np.ndarray],
) -> Dict[str, float]:
    pred = np.asarray(pred_coords, dtype=np.float32)
    true = np.asarray(true_coords, dtype=np.float32)
    current = np.asarray(current_coords, dtype=np.float32)
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected pred/true shape [samples, atoms, 3], got {pred.shape} and {true.shape}.")
    if current.shape != pred.shape:
        raise ValueError(
            f"Expected current coordinate array to match pred/true shape, got current={current.shape}, pred={pred.shape}."
        )
    previous: Optional[np.ndarray]
    if previous_coords is None:
        previous = None
    else:
        previous = np.asarray(previous_coords, dtype=np.float32)
        if previous.shape != pred.shape:
            raise ValueError(
                f"Expected previous coordinate array to match pred/true shape, got previous={previous.shape}, pred={pred.shape}."
            )
    pred_pair = np.concatenate([_upper_triangle_values(_pairwise_distances(frame)) for frame in pred], axis=0)
    true_pair = np.concatenate([_upper_triangle_values(_pairwise_distances(frame)) for frame in true], axis=0)
    pred_velocity_norm = np.linalg.norm(pred - current, axis=2).reshape(-1)
    true_velocity_norm = np.linalg.norm(true - current, axis=2).reshape(-1)
    acceleration_norm_w1 = float("nan")
    if previous is not None:
        context_velocity = current - previous
        pred_acceleration_norm = np.linalg.norm((pred - current) - context_velocity, axis=2).reshape(-1)
        true_acceleration_norm = np.linalg.norm((true - current) - context_velocity, axis=2).reshape(-1)
        acceleration_norm_w1 = _wasserstein_1d(pred_acceleration_norm, true_acceleration_norm)
    return {
        **_coordinate_cw1(pred.reshape(pred.shape[0], -1), true.reshape(true.shape[0], -1)),
        "molecule_pair_distance_w1": _wasserstein_1d(pred_pair, true_pair),
        "molecule_ensemble_velocity_norm_w1": _wasserstein_1d(pred_velocity_norm, true_velocity_norm),
        "molecule_ensemble_acceleration_norm_w1": acceleration_norm_w1,
    }


@torch.no_grad()
def _sample_molecule_ar_rollout(
    *,
    model: torch.nn.Module,
    ds,
    history_coords: np.ndarray,
    rollout_steps: int,
    nfe: int,
    solver: str,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    history = np.asarray(history_coords, dtype=np.float32).copy()
    generated: List[np.ndarray] = []
    for step_idx in range(int(rollout_steps)):
        context = ds.context_features_from_history_coords(history)
        hist = (context - ds.stats.context_mean[None, :]) / ds.stats.context_std[None, :]
        hist_t = torch.from_numpy(hist[None].astype(np.float32)).to(device)
        with _temporary_eval_seed(int(seed) + int(step_idx)):
            pred_norm = model.sample_future(
                hist_t,
                steps=int(nfe),
                solver=str(solver),
            )
        residual = ds.denormalize_target(pred_norm.detach().cpu().numpy()[0])[0]
        next_coords = history[-1] + residual.reshape(ds.data.atom_count, 3)
        generated.append(next_coords.astype(np.float32))
        history = np.concatenate([history, next_coords[None, :, :].astype(np.float32)], axis=0)
    return np.stack(generated, axis=0).astype(np.float32)


def _motion_norms_from_paths(paths: np.ndarray, previous_frame: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected coordinate paths [samples, frames, atoms, 3], got {arr.shape}.")
    velocity = arr[:, 1:] - arr[:, :-1]
    velocity_norm = np.linalg.norm(velocity, axis=3)
    if previous_frame is None:
        return velocity_norm, None
    previous = np.asarray(previous_frame, dtype=np.float32)
    if previous.shape != arr.shape[2:]:
        raise ValueError(f"previous_frame shape {previous.shape} does not match atom frame shape {arr.shape[2:]}.")
    context_velocity = arr[:, 0] - previous[None, :, :]
    previous_velocity = np.concatenate([context_velocity[:, None, :, :], velocity[:, :-1]], axis=1)
    acceleration = velocity - previous_velocity
    acceleration_norm = np.linalg.norm(acceleration, axis=3)
    return velocity_norm, acceleration_norm


def molecule_rollout_motion_metrics(
    pred_rollouts: np.ndarray,
    true_future: np.ndarray,
    history_coords: np.ndarray,
) -> Dict[str, Any]:
    pred = np.asarray(pred_rollouts, dtype=np.float32)
    true = np.asarray(true_future, dtype=np.float32)
    history = np.asarray(history_coords, dtype=np.float32)
    if pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(f"Expected pred_rollouts [samples, steps, atoms, 3], got {pred.shape}.")
    if true.ndim != 3 or true.shape[-1] != 3:
        raise ValueError(f"Expected true_future [steps, atoms, 3], got {true.shape}.")
    steps = min(int(pred.shape[1]), int(true.shape[0]))
    if steps <= 0:
        return {
            "molecule_ensemble_velocity_norm_w1": float("nan"),
            "molecule_ensemble_acceleration_norm_w1": float("nan"),
            "molecule_rollout_velocity_norm_w1": float("nan"),
            "molecule_rollout_acceleration_norm_w1": float("nan"),
            "molecule_rollout_velocity_norm_w1_by_horizon": {},
            "molecule_rollout_acceleration_norm_w1_by_horizon": {},
        }
    pred = pred[:, :steps]
    true = true[:steps]
    current = history[-1]
    previous = history[-2] if history.ndim == 3 and history.shape[0] >= 2 else None
    pred_paths = np.concatenate(
        [np.broadcast_to(current[None, None, :, :], (pred.shape[0], 1, pred.shape[2], 3)), pred],
        axis=1,
    )
    true_paths = np.concatenate(
        [np.broadcast_to(current[None, None, :, :], (pred.shape[0], 1, true.shape[1], 3)), np.broadcast_to(true[None], pred.shape)],
        axis=1,
    )
    pred_velocity, pred_acceleration = _motion_norms_from_paths(pred_paths, previous)
    true_velocity, true_acceleration = _motion_norms_from_paths(true_paths, previous)
    velocity_by_horizon = {
        str(h + 1): _wasserstein_1d(pred_velocity[:, h, :], true_velocity[:, h, :])
        for h in range(steps)
    }
    if pred_acceleration is None or true_acceleration is None:
        acceleration_by_horizon: Dict[str, float] = {str(h + 1): float("nan") for h in range(steps)}
        acceleration_ensemble = float("nan")
    else:
        acceleration_by_horizon = {
            str(h + 1): _wasserstein_1d(pred_acceleration[:, h, :], true_acceleration[:, h, :])
            for h in range(steps)
        }
        acceleration_ensemble = _wasserstein_1d(pred_acceleration, true_acceleration)
    velocity_values = np.asarray([value for value in velocity_by_horizon.values() if np.isfinite(value)], dtype=np.float64)
    acceleration_values = np.asarray([value for value in acceleration_by_horizon.values() if np.isfinite(value)], dtype=np.float64)
    return {
        "molecule_ensemble_velocity_norm_w1": _wasserstein_1d(pred_velocity, true_velocity),
        "molecule_ensemble_acceleration_norm_w1": acceleration_ensemble,
        "molecule_rollout_velocity_norm_w1": float(np.mean(velocity_values)) if velocity_values.size else float("nan"),
        "molecule_rollout_acceleration_norm_w1": float(np.mean(acceleration_values)) if acceleration_values.size else float("nan"),
        "molecule_rollout_velocity_norm_w1_by_horizon": velocity_by_horizon,
        "molecule_rollout_acceleration_norm_w1_by_horizon": acceleration_by_horizon,
    }


def _molecule_context_embedding(
    *,
    model: torch.nn.Module,
    ds,
    item: Mapping[str, Any],
    device: torch.device,
    context_embedding_kind: str,
) -> List[float]:
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "precompute"):
        raise ValueError("Molecule context export requires model.backbone.precompute(hist).")
    context = np.asarray(item["context"], dtype=np.float32)
    hist = (context - ds.stats.context_mean[None, :]) / ds.stats.context_std[None, :]
    hist_t = torch.from_numpy(hist[None].astype(np.float32)).to(device)
    cache = backbone.precompute(hist_t.float())
    if not hasattr(cache, str(context_embedding_kind)):
        raise ValueError(f"Unknown molecule context_embedding_kind={context_embedding_kind!r}.")
    embedding = getattr(cache, str(context_embedding_kind))
    if not torch.is_tensor(embedding) or embedding.ndim != 2 or int(embedding.shape[0]) != 1:
        raise ValueError(f"Molecule context embedding {context_embedding_kind!r} must have shape [1, dim].")
    return [float(x) for x in embedding.detach().cpu().numpy().astype(np.float32)[0].tolist()]


def molecule_context_embeddings_for_indices(
    *,
    model: torch.nn.Module,
    ds,
    example_indices: Sequence[int],
    checkpoint_id: str,
    dataset_key: str,
    member_key: str,
    stratum: str,
    split_phase: str,
    device: torch.device,
    context_embedding_kind: str = "ctx_summary",
) -> Dict[str, List[float]]:
    embeddings: Dict[str, List[float]] = {}
    for example_idx in [int(idx) for idx in example_indices]:
        item = ds.eval_item(int(example_idx))
        raw_id = molecule_context_id(
            dataset_key=dataset_key,
            member_key=member_key,
            stratum=stratum,
            split_phase=split_phase,
            item=item,
            example_idx=int(example_idx),
        )
        embeddings[molecule_context_embedding_id(str(checkpoint_id), raw_id)] = _molecule_context_embedding(
            model=model,
            ds=ds,
            item=item,
            device=device,
            context_embedding_kind=str(context_embedding_kind),
        )
    return embeddings


def molecule_context_id(
    *,
    dataset_key: str,
    member_key: str,
    stratum: str,
    split_phase: str,
    item: Mapping[str, Any],
    example_idx: int,
) -> str:
    payload = {
        "context_schema": MOLECULE_CONTEXT_SCHEMA,
        "dataset": str(dataset_key),
        "member": str(member_key),
        "stratum": str(stratum),
        "split_phase": str(split_phase),
        "example_idx": int(example_idx),
        "target_idx": int(item.get("target_idx", example_idx)),
        "trajectory": str(item.get("trajectory_key", item.get("trajectory_id", ""))),
        "iso_id": str(item.get("iso_id", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def molecule_context_embedding_id(checkpoint_id: str, context_id: str) -> str:
    return f"{str(checkpoint_id)}:{str(context_id)}"


def _molecule_window_metrics(
    *,
    model: torch.nn.Module,
    ds,
    item: Mapping[str, Any],
    atom_symbols: Sequence[str],
    bond_pairs: np.ndarray,
    rollout_steps: int,
    sample_count: int,
    runtime_nfe: int,
    solver_key: str,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    true_future = np.asarray(item["future_coords"], dtype=np.float32)
    current = np.asarray(item["current_coords"], dtype=np.float32)
    history_coords = np.asarray(item.get("history_coords", []), dtype=np.float32)
    previous = history_coords[-2] if history_coords.ndim == 3 and history_coords.shape[0] >= 2 else None
    sample_rollouts: List[np.ndarray] = []
    kabsch_values: List[float] = []
    first_horizon_metrics: List[Dict[str, float]] = []
    for sample_idx in range(int(sample_count)):
        pred_future = _sample_molecule_ar_rollout(
            model=model,
            ds=ds,
            history_coords=history_coords,
            rollout_steps=int(rollout_steps),
            nfe=int(runtime_nfe),
            solver=str(solver_key),
            device=device,
            seed=int(seed) + int(sample_idx),
        )
        sample_rollouts.append(pred_future)
        if pred_future.shape[0] > 0 and true_future.shape[0] > 0:
            metrics = molecule_coordinate_metrics(
                pred_future[0],
                true_future[0],
                atom_symbols=atom_symbols,
                bond_pairs=bond_pairs,
            )
            first_horizon_metrics.append(metrics)
            kabsch_values.append(float(metrics["molecule_kabsch_rmsd_3d"]))
    if not sample_rollouts:
        return {key: float("nan") for key in MOLECULE_PRIMARY_METRICS}
    stacked = np.stack(sample_rollouts, axis=0)
    first_pred = stacked[:, 0, :, :]
    first_true = np.broadcast_to(true_future[0][None, :, :], first_pred.shape)
    current_many = np.broadcast_to(current[None, :, :], first_pred.shape)
    previous_many = None if previous is None else np.broadcast_to(previous[None, :, :], first_pred.shape)
    distributional = molecule_distributional_metrics(first_pred, first_true, current_many, previous_many)
    motion = molecule_rollout_motion_metrics(stacked, true_future, history_coords)
    row: Dict[str, Any] = {
        "molecule_kabsch_rmsd_3d": _safe_mean(kabsch_values),
        "molecule_ensemble_velocity_norm_w1": distributional.get("molecule_ensemble_velocity_norm_w1"),
        "molecule_ensemble_acceleration_norm_w1": distributional.get("molecule_ensemble_acceleration_norm_w1"),
        "molecule_rollout_velocity_norm_w1": motion.get("molecule_rollout_velocity_norm_w1"),
        "molecule_rollout_acceleration_norm_w1": motion.get("molecule_rollout_acceleration_norm_w1"),
        "molecule_coordinate_w1_mean": distributional.get("molecule_coordinate_w1_mean"),
        "molecule_pair_distance_w1": distributional.get("molecule_pair_distance_w1"),
        "molecule_kabsch_rmsd_3d_sample_std": float(np.std(np.asarray(kabsch_values, dtype=np.float64))) if kabsch_values else float("nan"),
    }
    for key in ("raw_coord_mae", "raw_coord_rmse", "pairwise_distance_mae", "pairwise_distance_rmse", "finite_rate", "clash_rate", "bond_contact_violation_rate"):
        row[key] = _safe_mean([metrics.get(key) for metrics in first_horizon_metrics])
    return row


@torch.no_grad()
def evaluate_molecule_rollout_schedule(
    *,
    model: torch.nn.Module,
    ds,
    cfg,
    scheduler_key: str,
    solver_key: str,
    target_nfe: int,
    runtime_nfe: int,
    time_grid: Sequence[float],
    example_indices: Sequence[int],
    sample_count: int,
    rollout_steps: int,
    seed: int,
    split_phase: str,
    checkpoint_id: str,
    dataset_key: str,
    member_key: str = "",
    stratum: str = "",
    formula: str = "",
    source_zip_name: str = "",
    device: torch.device | None = None,
) -> Dict[str, Any]:
    dev = torch.device("cpu") if device is None else device
    indices = [int(idx) for idx in example_indices]
    if not indices:
        raise ValueError("Molecule schedule evaluation requires at least one example index.")
    grid = validate_time_grid(time_grid, macro_steps=int(runtime_nfe))
    atom_symbols = ds.data.atom_symbols
    bond_pairs = _bond_pairs(ds.stats.reference_coords, atom_symbols)
    backup = _apply_sample_overrides(model, cfg, time_grid=tuple(float(x) for x in grid))
    per_context: List[Dict[str, Any]] = []
    try:
        for example_idx in indices:
            item = ds.eval_item(int(example_idx))
            metrics = _molecule_window_metrics(
                model=model,
                ds=ds,
                item=item,
                atom_symbols=atom_symbols,
                bond_pairs=bond_pairs,
                rollout_steps=int(rollout_steps),
                sample_count=int(sample_count),
                runtime_nfe=int(runtime_nfe),
                solver_key=str(solver_key),
                device=dev,
                seed=int(seed) + 10_000 * int(example_idx),
            )
            target_idx = int(item.get("target_idx", example_idx))
            flags = {
                "transition": bool(item.get("transition", False)),
                "transition_window": bool(item.get("transition_window", False)),
                "discontinuity": bool(item.get("discontinuity", False)),
                "discontinuity_window": bool(item.get("discontinuity_window", False)),
                "duplicate": bool(item.get("duplicate", False)),
            }
            context_id = molecule_context_id(
                dataset_key=str(dataset_key),
                member_key=str(member_key),
                stratum=str(stratum),
                split_phase=str(split_phase),
                item=item,
                example_idx=int(example_idx),
            )
            per_context.append(
                {
                    **metrics,
                    "context_schema": MOLECULE_CONTEXT_SCHEMA,
                    "context_id": context_id,
                    "context_embedding_id": molecule_context_embedding_id(str(checkpoint_id), context_id),
                    "example_idx": int(example_idx),
                    "target_t": int(target_idx),
                    "history_start": int(target_idx) - int(getattr(ds, "H", 0)),
                    "history_stop": int(target_idx),
                    "target_stop": int(target_idx) + int(rollout_steps),
                    "axis_dataset": str(dataset_key),
                    "axis_member": str(member_key),
                    "axis_stratum": str(stratum),
                    "axis_formula": str(formula),
                    "axis_atom_count": int(ds.data.atom_count),
                    "axis_trajectory": str(item.get("trajectory_key", item.get("trajectory_id", ""))),
                    "axis_iso_id": str(item.get("iso_id", "")),
                    "axis_window": str(target_idx),
                    "axis_flags": json.dumps(flags, sort_keys=True, separators=(",", ":")),
                    "member_key": str(member_key),
                    "stratum": str(stratum),
                    "formula": str(formula),
                    "source_zip_name": str(source_zip_name),
                    "num_eval_samples": int(sample_count),
                    "sample_seed_start": int(seed) + 10_000 * int(example_idx),
                    "sample_seed_values_json": json.dumps(
                        [int(seed) + 10_000 * int(example_idx) + int(sample_idx) for sample_idx in range(int(sample_count))],
                        separators=(",", ":"),
                    ),
                }
            )
    finally:
        _restore_sample_overrides(model, cfg, backup)
    summary: Dict[str, Any] = {
        "benchmark_family": SCENARIO_FAMILY_MOLECULE,
        "dataset_key": str(dataset_key),
        "member_key": str(member_key),
        "stratum": str(stratum),
        "scheduler_key": str(scheduler_key),
        "solver_key": str(solver_key),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "realized_nfe": int(expected_realized_nfe(str(solver_key), int(target_nfe))),
        "num_eval_samples": int(sample_count),
        "eval_windows": int(len(indices)),
        "rollout_steps": int(rollout_steps),
        "per_context_rows": per_context,
    }
    for metric in MOLECULE_PRIMARY_METRICS:
        summary[metric] = _safe_mean([row.get(metric) for row in per_context])
    summary["selection_metric_value"] = summary.get("molecule_kabsch_rmsd_3d")
    return summary


def load_molecule_checkpoint_splits(
    *,
    checkpoint_path: str | Path,
    dataset_key: str,
    stratum: str,
    processed_dir: str | Path | None,
    rollout_steps: int,
    stride_eval: int,
    device: torch.device,
) -> Dict[str, Any]:
    resolved_checkpoint = resolve_project_path(str(checkpoint_path))
    checkpoint_payload = torch.load(str(resolved_checkpoint), map_location="cpu", weights_only=False)
    if "molecule_stats" not in checkpoint_payload:
        raise RuntimeError("Molecule checkpoint is missing molecule_stats; refusing to rebuild normalization for evaluation.")
    checkpoint_stats = molecule_stats_from_mapping(checkpoint_payload["molecule_stats"])
    resolved_dataset_key = str(dataset_key or checkpoint_payload.get("dataset_key", "") or DEFAULT_MOLECULE_DATASET_KEY)
    resolved_stratum = str(stratum or checkpoint_payload.get("stratum", "") or "")
    resolved_processed_dir = (
        default_molecule_processed_dir(resolved_dataset_key, resolved_stratum)
        if processed_dir in (None, "")
        else resolve_project_path(str(processed_dir))
    )
    model, cfg = load_checkpoint_model(resolved_checkpoint, device)
    splits = build_molecule_dataset_splits(
        processed_dir=resolved_processed_dir,
        cfg=cfg,
        history_len=int(cfg.history_len),
        future_horizon=max(1, int(rollout_steps)),
        stride_train=1,
        stride_eval=int(stride_eval),
        stats=checkpoint_stats,
        dataset_key=resolved_dataset_key,
        stratum=resolved_stratum,
    )
    return {
        "model": model,
        "cfg": cfg,
        "splits": splits,
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_stats": checkpoint_stats,
        "dataset_key": resolved_dataset_key,
        "stratum": resolved_stratum,
        "processed_dir": resolved_processed_dir,
    }


def _aggregate(rows: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if len(values) == 0:
            continue
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return out


def _numeric_leaves(value: Any, prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], float]]:
    if isinstance(value, Mapping):
        rows: List[Tuple[Tuple[str, ...], float]] = []
        for key, child in value.items():
            rows.extend(_numeric_leaves(child, (*prefix, str(key))))
        return rows
    if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
        return [(prefix, float(value))]
    return []


def _set_nested_metric(payload: Dict[str, Any], path: Sequence[str], value: float) -> None:
    cursor = payload
    for part in path[:-1]:
        cursor = cursor.setdefault(str(part), {})
    cursor[str(path[-1])] = float(value)


def aggregate_molecule_group_evaluation(
    *,
    dataset_key: str,
    stratum_summaries: Sequence[Mapping[str, Any]],
    group_root: str | Path | None = None,
) -> Dict[str, Any]:
    manifest = load_molecule_group_manifest(str(dataset_key), group_root)
    allowed_strata = {str(row["stratum"]) for row in manifest.get("strata", [])}
    weighted: Dict[Tuple[str, ...], List[Tuple[float, float]]] = {}
    per_stratum: List[Dict[str, Any]] = []
    for summary in stratum_summaries:
        stratum = str(summary.get("stratum", ""))
        if stratum not in allowed_strata:
            raise ValueError(f"Stratum {stratum!r} is not part of molecule group {dataset_key!r}.")
        weight = float(max(1, int(summary.get("examples", 1))))
        per_stratum.append(
            {
                "stratum": stratum,
                "examples": int(weight),
                "metrics": dict(summary.get("metrics", {}) or {}),
            }
        )
        for path, value in _numeric_leaves(summary.get("metrics", {})):
            weighted.setdefault(path, []).append((float(value), weight))
    metrics: Dict[str, Any] = {}
    for path, values in sorted(weighted.items()):
        total_weight = float(sum(weight for _, weight in values))
        if total_weight <= 0.0:
            continue
        mean = float(sum(value * weight for value, weight in values) / total_weight)
        _set_nested_metric(metrics, path, mean)
    return {
        "dataset_key": str(dataset_key),
        "benchmark_family": str(manifest.get("benchmark_family", "molecule_3d")),
        "stratum_count": int(len(per_stratum)),
        "examples": int(sum(row["examples"] for row in per_stratum)),
        "metrics": metrics,
        "per_stratum": per_stratum,
    }


@torch.no_grad()
def evaluate_molecule_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_torch_device(str(args.device))
    checkpoint_path = resolve_project_path(str(args.checkpoint))
    checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "molecule_stats" not in checkpoint_payload:
        raise RuntimeError("Molecule checkpoint is missing molecule_stats; refusing to rebuild normalization for evaluation.")
    checkpoint_stats = molecule_stats_from_mapping(checkpoint_payload["molecule_stats"])
    dataset_key = str(
        getattr(args, "dataset_key", "")
        or checkpoint_payload.get("dataset_key", "")
        or DEFAULT_MOLECULE_DATASET_KEY
    )
    stratum = str(getattr(args, "stratum", "") or checkpoint_payload.get("stratum", "") or "")
    processed_dir = (
        default_molecule_processed_dir(dataset_key, stratum)
        if getattr(args, "processed_dir", None) in (None, "")
        else resolve_project_path(str(args.processed_dir))
    )
    model, cfg = load_checkpoint_model(checkpoint_path, device)
    loss_splits = build_molecule_dataset_splits(
        processed_dir=processed_dir,
        cfg=cfg,
        history_len=int(cfg.history_len),
        future_horizon=int(cfg.prediction_horizon),
        stride_train=1,
        stride_eval=int(args.stride_eval),
        stats=checkpoint_stats,
        dataset_key=dataset_key,
        stratum=stratum,
    )
    solver_key = normalize_solver_key(str(getattr(args, "solver_key", getattr(args, "solver", "euler"))))
    target_nfe = int(getattr(args, "target_nfe", getattr(args, "nfe", 16)))
    runtime_nfe = int(getattr(args, "runtime_nfe", solver_macro_steps(solver_key, target_nfe)))
    realized_nfe = int(expected_realized_nfe(solver_key, target_nfe))
    nfe_role = str(getattr(args, "nfe_role", NFE_ROLE_SEEN))
    scenario_key = str(getattr(args, "scenario_key", "") or dataset_key)
    rollout_steps = max(1, int(getattr(args, "rollout_steps", 16)))
    splits = build_molecule_dataset_splits(
        processed_dir=processed_dir,
        cfg=cfg,
        history_len=int(cfg.history_len),
        future_horizon=rollout_steps,
        stride_train=1,
        stride_eval=int(args.stride_eval),
        stats=checkpoint_stats,
        dataset_key=dataset_key,
        stratum=stratum,
    )
    split = str(args.split)
    ds = splits[split]
    avg_loss = evaluate_average_loss(
        loss_splits[split],
        model,
        cfg,
        model_name="otflow",
        max_batches=None if args.val_max_batches is None else int(args.val_max_batches),
        shuffle=False,
    )
    rng = np.random.default_rng(int(args.seed))
    total = len(ds)
    if total <= 0:
        raise ValueError(f"Empty molecule {split} split.")
    max_windows = min(int(args.max_windows), total)
    indices = np.arange(total)
    if max_windows < total:
        indices = np.sort(rng.choice(indices, size=max_windows, replace=False))

    atom_symbols = ds.data.atom_symbols
    bond_pairs = _bond_pairs(ds.stats.reference_coords, atom_symbols)
    all_rows: List[Dict[str, float]] = []
    clean_rows: List[Dict[str, float]] = []
    transition_rows: List[Dict[str, float]] = []
    horizon_rows: Dict[int, List[Dict[str, float]]] = {}
    distribution_rows: List[Dict[str, float]] = []
    motion_rows: List[Dict[str, float]] = []
    rollout_velocity_horizon_rows: Dict[int, List[Dict[str, float]]] = {}
    rollout_acceleration_horizon_rows: Dict[int, List[Dict[str, float]]] = {}
    dist_inputs: Dict[str, Dict[str, List[np.ndarray]]] = {
        scope: {"pred": [], "true": [], "current": [], "previous": []}
        for scope in ("all_first_horizon", "clean_first_horizon", "transition_first_horizon")
    }

    for dataset_idx in indices:
        item = ds.eval_item(int(dataset_idx))
        true_future = np.asarray(item["future_coords"], dtype=np.float32)
        current = np.asarray(item["current_coords"], dtype=np.float32)
        history_coords = np.asarray(item.get("history_coords", []), dtype=np.float32)
        previous = history_coords[-2] if history_coords.ndim == 3 and history_coords.shape[0] >= 2 else None
        sample_metrics: List[Dict[str, float]] = []
        sample_kabsch: List[float] = []
        sample_rollouts: List[np.ndarray] = []
        for sample_idx in range(int(args.sample_count)):
            pred_future = _sample_molecule_ar_rollout(
                model=model,
                ds=ds,
                history_coords=history_coords,
                rollout_steps=rollout_steps,
                nfe=int(runtime_nfe),
                solver=str(solver_key),
                device=device,
                seed=int(args.seed) + 10_000 * int(dataset_idx) + int(sample_idx),
            )
            sample_rollouts.append(pred_future)
            horizon_limit = min(pred_future.shape[0], true_future.shape[0])
            for h in range(horizon_limit):
                metrics = molecule_coordinate_metrics(
                    pred_future[h],
                    true_future[h],
                    atom_symbols=atom_symbols,
                    bond_pairs=bond_pairs,
                )
                metrics["horizon"] = float(h + 1)
                sample_metrics.append(metrics)
                horizon_rows.setdefault(h + 1, []).append(metrics)
                sample_kabsch.append(float(metrics["molecule_kabsch_rmsd_3d"]))
                if h == 0:
                    is_transition = bool(item.get("transition_window", item["transition"]))
                    scopes = ["all_first_horizon", "transition_first_horizon" if is_transition else "clean_first_horizon"]
                    for scope in scopes:
                        dist_inputs[scope]["pred"].append(pred_future[h].astype(np.float32))
                        dist_inputs[scope]["true"].append(true_future[h].astype(np.float32))
                        dist_inputs[scope]["current"].append(current.astype(np.float32))
                        if previous is not None:
                            dist_inputs[scope]["previous"].append(previous.astype(np.float32))
        if sample_rollouts:
            motion = molecule_rollout_motion_metrics(
                np.stack(sample_rollouts, axis=0),
                true_future,
                history_coords,
            )
            motion_rows.append(
                {
                    key: float(value)
                    for key, value in motion.items()
                    if isinstance(value, (int, float, np.floating))
                }
            )
            for horizon_key, value in dict(motion["molecule_rollout_velocity_norm_w1_by_horizon"]).items():
                rollout_velocity_horizon_rows.setdefault(int(horizon_key), []).append(
                    {"molecule_rollout_velocity_norm_w1": float(value)}
                )
            for horizon_key, value in dict(motion["molecule_rollout_acceleration_norm_w1_by_horizon"]).items():
                rollout_acceleration_horizon_rows.setdefault(int(horizon_key), []).append(
                    {"molecule_rollout_acceleration_norm_w1": float(value)}
                )
        if sample_metrics:
            first_sample = sample_metrics[0]
            all_rows.append(first_sample)
            if bool(item.get("transition_window", item["transition"])):
                transition_rows.append(first_sample)
            else:
                clean_rows.append(first_sample)
            distribution_rows.append(
                {
                    "molecule_kabsch_rmsd_3d_sample_mean": float(np.mean(sample_kabsch)),
                    "molecule_kabsch_rmsd_3d_sample_std": float(np.std(sample_kabsch)),
                    "molecule_kabsch_rmsd_3d_sample_min": float(np.min(sample_kabsch)),
                }
            )

    horizon_summary = {str(h): _aggregate(rows) for h, rows in sorted(horizon_rows.items())}
    distributional_summary: Dict[str, Dict[str, float]] = {}
    for scope, arrays in dist_inputs.items():
        if not arrays["pred"]:
            distributional_summary[scope] = {}
            continue
        distributional_summary[scope] = molecule_distributional_metrics(
            np.stack(arrays["pred"], axis=0),
            np.stack(arrays["true"], axis=0),
            np.stack(arrays["current"], axis=0),
            np.stack(arrays["previous"], axis=0) if len(arrays["previous"]) == len(arrays["pred"]) else None,
        )
    summary: Dict[str, Any] = {
        "checkpoint": _project_display_path(checkpoint_path),
        "dataset_key": dataset_key,
        "scenario_key": scenario_key,
        "benchmark_family": SCENARIO_FAMILY_MOLECULE,
        "stratum": stratum,
        "split": split,
        "nfe_role": nfe_role,
        "solver_key": solver_key,
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "realized_nfe": int(realized_nfe),
        "examples": int(len(indices)),
        "sample_count": int(args.sample_count),
        "rollout_steps": int(rollout_steps),
        "validation_vector_loss": avg_loss,
        "metrics": {
            "all_first_horizon": _aggregate(all_rows),
            "clean_first_horizon": _aggregate(clean_rows),
            "transition_first_horizon": _aggregate(transition_rows),
            "horizon": horizon_summary,
            "generative_distribution": _aggregate(distribution_rows),
            "distributional": distributional_summary,
            "motion_distribution": _aggregate(motion_rows),
            "rollout_stability_by_horizon": {
                "velocity": {str(h): _aggregate(rows) for h, rows in sorted(rollout_velocity_horizon_rows.items())},
                "acceleration": {str(h): _aggregate(rows) for h, rows in sorted(rollout_acceleration_horizon_rows.items())},
            },
        },
    }
    if args.out_json:
        out_path = resolve_project_path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(summary, str(out_path))
    return summary


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _molecule_solver_cases(args: argparse.Namespace) -> List[Tuple[str, int, int]]:
    solvers = normalize_solver_keys(str(getattr(args, "solver_names", "") or getattr(args, "solver", "euler")))
    role = str(getattr(args, "nfe_role", NFE_ROLE_SEEN) or NFE_ROLE_SEEN)
    raw_nfes = str(getattr(args, "target_nfe_values", "") or "").strip()
    if raw_nfes:
        target_nfes = _parse_int_csv(raw_nfes)
    elif str(getattr(args, "solver_names", "") or "").strip():
        target_nfes = list(canonical_nfes_for_role(role))
    else:
        target_nfes = [int(getattr(args, "target_nfe", getattr(args, "nfe", 16)))]
    cases: List[Tuple[str, int, int]] = []
    for solver in solvers:
        for target_nfe in target_nfes:
            runtime_nfe = solver_macro_steps(solver, int(target_nfe))
            cases.append((solver, int(target_nfe), int(runtime_nfe)))
    return cases


def evaluate_molecule_checkpoint_grid(args: argparse.Namespace) -> Dict[str, Any]:
    cases = _molecule_solver_cases(args)
    evaluations: List[Dict[str, Any]] = []
    for solver_key, target_nfe, runtime_nfe in cases:
        case_args = argparse.Namespace(**vars(args))
        case_args.solver_key = str(solver_key)
        case_args.target_nfe = int(target_nfe)
        case_args.runtime_nfe = int(runtime_nfe)
        evaluations.append(evaluate_molecule_checkpoint(case_args))
    if len(evaluations) == 1:
        return evaluations[0]
    return {
        "benchmark_family": SCENARIO_FAMILY_MOLECULE,
        "scenario_key": str(getattr(args, "scenario_key", "") or getattr(args, "dataset_key", DEFAULT_MOLECULE_DATASET_KEY)),
        "nfe_role": str(getattr(args, "nfe_role", NFE_ROLE_SEEN) or NFE_ROLE_SEEN),
        "solver_keys": sorted({str(row["solver_key"]) for row in evaluations}, key=CANONICAL_SOLVER_KEYS.index),
        "target_nfe_values": sorted({int(row["target_nfe"]) for row in evaluations}),
        "evaluation_count": int(len(evaluations)),
        "evaluations": evaluations,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate molecule 3D coordinate OTFlow checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    parser.add_argument("--stratum", default="")
    parser.add_argument("--split", default="val", choices=("val", "val_clean", "test", "test_clean"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_windows", type=int, default=256)
    parser.add_argument("--sample_count", type=int, default=1)
    parser.add_argument("--rollout_steps", type=int, default=16)
    parser.add_argument("--nfe_role", default=NFE_ROLE_SEEN)
    parser.add_argument("--target_nfe_values", default="")
    parser.add_argument("--solver_names", default=",".join(CANONICAL_SOLVER_KEYS))
    parser.add_argument("--nfe", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--solver", default="", help=argparse.SUPPRESS)
    parser.add_argument("--scenario_key", default="")
    parser.add_argument("--stride_eval", type=int, default=1)
    parser.add_argument("--val_max_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_json", default="")
    return parser


def main() -> None:
    summary = evaluate_molecule_checkpoint_grid(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
