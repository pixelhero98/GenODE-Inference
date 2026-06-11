from __future__ import annotations

import fnmatch
import json
import math
import re
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import torch

from genode.data.otflow_paths import project_root, resolve_project_path
from genode.models.config import OTFlowConfig


MOLECULE_BENCHMARK_FAMILY = "molecule_3d"
DEFAULT_MOLECULE_DATASET_KEY = "molecule_3d"
MOLECULE_GROUP_DATASET_KEYS = ("molecule_3d_set1", "molecule_3d_set2", "molecule_3d_set3")
MOLECULE_TOKEN_DIM = 3
MOLECULE_CONTEXT_ATOM_FEATURE_DIM = 11
DEFAULT_MOLECULE_SPLIT_SEED = 20260610

ATOM_MASS = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}
ATOM_NUMBER = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
ATOM_COVALENT_RADIUS = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


@dataclass(frozen=True)
class MoleculeProcessedData:
    coords: np.ndarray
    atom_symbols: Tuple[str, ...]
    iso_ids: np.ndarray
    trajectory_ids: np.ndarray
    trajectory_keys: Tuple[str, ...]
    segment_ends: np.ndarray
    frame_counts: Dict[int, int]
    duplicate_step_mask: np.ndarray
    transition_step_mask: np.ndarray
    discontinuity_step_mask: np.ndarray
    frame_rmsd: np.ndarray
    metadata: Dict[str, Any]

    @property
    def dataset_key(self) -> str:
        return str(self.metadata.get("dataset_key", DEFAULT_MOLECULE_DATASET_KEY))

    @property
    def stratum(self) -> str:
        return str(self.metadata.get("stratum", ""))

    @property
    def atom_count(self) -> int:
        return int(self.coords.shape[1])

    @property
    def token_dim(self) -> int:
        return int(self.coords.shape[2])

    @property
    def coord_dim(self) -> int:
        return int(self.atom_count) * int(self.token_dim)

    @property
    def context_feature_dim(self) -> int:
        return int(self.atom_count) * int(MOLECULE_CONTEXT_ATOM_FEATURE_DIM)


@dataclass(frozen=True)
class MoleculeStats:
    target_mean: np.ndarray
    target_std: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    reference_coords: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "reference_coords": self.reference_coords.tolist(),
        }


def _clean_path_token(value: str | None, *, label: str, default: str | None = None) -> str:
    token = str(value if value not in (None, "") else default or "").strip()
    if not token:
        raise ValueError(f"{label} is required.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", token):
        raise ValueError(f"{label} must contain only letters, numbers, '.', '_' or '-'; got {value!r}.")
    return token


def _clean_optional_stratum(value: str | None) -> str:
    text = str(value or "").strip().strip("/\\")
    if not text:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"stratum must contain only letters, numbers, '.', '_' or '-'; got {value!r}.")
    return text


def _project_display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return resolved.name


def default_molecule_raw_zip(dataset_key: str | None = None, stratum: str | None = None) -> Path:
    del stratum
    key = _clean_path_token(dataset_key, label="dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    return resolve_project_path(Path("data") / MOLECULE_BENCHMARK_FAMILY / key / "raw" / f"{key}.zip")


def default_molecule_processed_dir(dataset_key: str | None = None, stratum: str | None = None) -> Path:
    key = _clean_path_token(dataset_key, label="dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    root = resolve_project_path(Path("data") / MOLECULE_BENCHMARK_FAMILY / key / "processed")
    stratum_name = _clean_optional_stratum(stratum)
    return root / stratum_name if stratum_name else root


def molecule_processed_npz_path(
    processed_dir: str | Path | None = None,
    *,
    dataset_key: str | None = None,
    stratum: str | None = None,
) -> Path:
    root = default_molecule_processed_dir(dataset_key, stratum) if processed_dir is None else resolve_project_path(processed_dir)
    return root / "molecule_3d_dataset.npz"


def molecule_processed_metadata_path(
    processed_dir: str | Path | None = None,
    *,
    dataset_key: str | None = None,
    stratum: str | None = None,
) -> Path:
    root = default_molecule_processed_dir(dataset_key, stratum) if processed_dir is None else resolve_project_path(processed_dir)
    return root / "molecule_3d_metadata.json"


def _xyz_names(zf: ZipFile) -> List[str]:
    return sorted(name for name in zf.namelist() if name.lower().endswith(".xyz") and not name.endswith("/"))


def _category_from_name(name: str) -> str:
    parts = PurePosixPath(name).parts
    if len(parts) > 1 and parts[0] not in {"", "."}:
        return str(parts[0])
    stem = str(PurePosixPath(name).stem)
    match = re.search(r"(Dynamic|Direct)_([A-Za-z0-9_.-]+?)_Iso\d+(?:\b|[_.-])", stem)
    if match:
        return f"{match.group(1)}_{match.group(2).rstrip('_.-')}"
    match = re.search(r"(Dynamic|Direct)_([A-Za-z0-9_.-]+)", stem)
    return f"{match.group(1)}_{match.group(2).rstrip('_.-')}" if match else ""


def _iso_id_from_name(name: str) -> Optional[int]:
    match = re.search(r"Iso(\d+)", name)
    return int(match.group(1)) if match else None


def _trajectory_key_from_name(name: str, trajectory_idx: int) -> str:
    category = _category_from_name(name)
    iso_id = _iso_id_from_name(name)
    if category and iso_id is not None:
        return f"{category}/Iso{iso_id}"
    if category:
        return f"{category}/{PurePosixPath(name).stem}"
    if iso_id is not None:
        return f"Iso{iso_id}"
    return f"trajectory_{int(trajectory_idx):05d}"


def _name_sort_key(name: str) -> Tuple[str, int, str]:
    iso_id = _iso_id_from_name(name)
    return (_category_from_name(name), int(iso_id) if iso_id is not None else 10**12, name)


def _formula(symbols: Sequence[str]) -> str:
    counts = Counter(str(symbol) for symbol in symbols)
    ordered: List[str] = []
    for symbol in ("C", "H"):
        if symbol in counts:
            ordered.append(symbol)
    ordered.extend(symbol for symbol in sorted(counts) if symbol not in {"C", "H"})
    return "".join(f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}" for symbol in ordered)


def _read_first_symbols_from_zip(zf: ZipFile, name: str) -> Tuple[str, ...]:
    with zf.open(name) as fh:
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{name}: no XYZ frames found.")
            if line.strip():
                break
        try:
            atom_count = int(line.strip())
        except ValueError as exc:
            raise ValueError(f"{name}: expected atom count, got {line!r}.") from exc
        fh.readline()
        symbols = []
        for _ in range(atom_count):
            raw = fh.readline()
            if not raw:
                raise ValueError(f"{name}: truncated first frame.")
            parts = raw.decode("utf-8", errors="replace").split()
            if len(parts) != 4:
                raise ValueError(f"{name}: expected XYZ atom row with 4 columns, got {parts!r}.")
            symbols.append(parts[0])
    return tuple(symbols)


def _read_xyz_trajectory_from_zip(
    zf: ZipFile,
    name: str,
    *,
    atom_count_expected: int | None = None,
) -> Tuple[Tuple[str, ...], np.ndarray]:
    symbols: Optional[Tuple[str, ...]] = None
    frames: List[np.ndarray] = []
    with zf.open(name) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                atom_count = int(line.strip())
            except ValueError as exc:
                raise ValueError(f"{name}: expected atom count, got {line!r}.") from exc
            if atom_count_expected is not None and atom_count != int(atom_count_expected):
                raise ValueError(f"{name}: expected {int(atom_count_expected)} atoms, found {atom_count}.")
            fh.readline()
            frame_symbols: List[str] = []
            frame_coords = np.empty((atom_count, MOLECULE_TOKEN_DIM), dtype=np.float32)
            for atom_idx in range(atom_count):
                raw = fh.readline()
                if not raw:
                    raise ValueError(f"{name}: truncated frame.")
                parts = raw.decode("utf-8", errors="replace").split()
                if len(parts) != 4:
                    raise ValueError(f"{name}: expected XYZ atom row with 4 columns, got {parts!r}.")
                frame_symbols.append(parts[0])
                frame_coords[atom_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
            if not np.isfinite(frame_coords).all():
                raise ValueError(f"{name}: non-finite coordinate encountered.")
            frame_symbols_t = tuple(frame_symbols)
            if symbols is None:
                symbols = frame_symbols_t
            elif frame_symbols_t != symbols:
                raise ValueError(f"{name}: atom order changes across frames.")
            frames.append(frame_coords)
    if symbols is None or not frames:
        raise ValueError(f"{name}: no XYZ frames found.")
    return symbols, np.stack(frames, axis=0)


def discover_molecule_xyz_strata(zip_path: str | Path) -> Dict[str, Dict[str, Any]]:
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing molecule trajectory zip: {zip_path}")
    groups: Dict[str, List[str]] = {}
    with ZipFile(zip_path) as zf:
        for name in _xyz_names(zf):
            groups.setdefault(_category_from_name(name), []).append(name)
        summaries: Dict[str, Dict[str, Any]] = {}
        for category, names in sorted(groups.items()):
            first_symbols = _read_first_symbols_from_zip(zf, names[0])
            fixed_order = True
            formulas = {_formula(first_symbols)}
            atom_counts = {len(first_symbols)}
            for name in names[1:]:
                symbols = _read_first_symbols_from_zip(zf, name)
                formulas.add(_formula(symbols))
                atom_counts.add(len(symbols))
                fixed_order = fixed_order and symbols == first_symbols
            key = category or "default"
            summaries[key] = {
                "stratum": category,
                "xyz_count": int(len(names)),
                "atom_count": int(len(first_symbols)) if len(atom_counts) == 1 else None,
                "formula": next(iter(formulas)) if len(formulas) == 1 else None,
                "fixed_atom_order": bool(fixed_order),
                "mixed_shape": bool(len(atom_counts) > 1 or len(formulas) > 1 or not fixed_order),
                "trainable": bool(not category.startswith("Direct_")),
            }
    return summaries


def default_molecule_group_root() -> Path:
    return resolve_project_path(Path("data") / MOLECULE_BENCHMARK_FAMILY)


def default_molecule_group_manifest_path(dataset_key: str, group_root: str | Path | None = None) -> Path:
    key = _clean_path_token(dataset_key, label="dataset_key")
    root = default_molecule_group_root() if group_root is None else resolve_project_path(group_root)
    return root / key / "group_manifest.json"


def _source_zip_display_name(zip_path: str | Path) -> str:
    return Path(zip_path).expanduser().resolve().name


def _member_key(source_zip_name: str, stratum: str) -> str:
    stem = Path(str(source_zip_name)).stem
    return _clean_path_token(f"{stem}__{str(stratum)}", label="member_key")


def discover_trainable_molecule_strata(zip_paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for raw_path in zip_paths:
        zip_path = Path(raw_path).expanduser().resolve()
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing molecule trajectory zip: {zip_path}")
        source_zip_name = _source_zip_display_name(zip_path)
        for stratum_key, summary in discover_molecule_xyz_strata(zip_path).items():
            stratum_name = str(summary.get("stratum") or stratum_key)
            if not stratum_name or stratum_name.startswith("Direct_"):
                continue
            if not bool(summary.get("trainable", True)) or bool(summary.get("mixed_shape", False)):
                continue
            row_key = (source_zip_name, stratum_name)
            if row_key in seen:
                raise ValueError(f"Duplicate molecule stratum {source_zip_name}/{stratum_name}.")
            seen.add(row_key)
            member_key = _member_key(source_zip_name, stratum_name)
            rows.append(
                {
                    "member_key": member_key,
                    "stratum": stratum_name,
                    "source_zip_name": source_zip_name,
                    "source_zip_path": str(zip_path),
                    "xyz_count": int(summary.get("xyz_count", 0)),
                    "trajectory_count": int(summary.get("xyz_count", 0)),
                    "atom_count": int(summary["atom_count"]),
                    "formula": str(summary["formula"]),
                    "fixed_atom_order": bool(summary.get("fixed_atom_order", True)),
                    "mixed_shape": bool(summary.get("mixed_shape", False)),
                    "trainable": True,
                }
            )
    return sorted(rows, key=lambda row: (str(row["source_zip_name"]), str(row["stratum"])))


def build_balanced_molecule_stratum_groups(
    zip_paths: Sequence[str | Path],
    *,
    dataset_keys: Sequence[str] = MOLECULE_GROUP_DATASET_KEYS,
) -> Dict[str, Any]:
    keys = tuple(_clean_path_token(key, label="dataset_key") for key in dataset_keys)
    if len(keys) <= 0:
        raise ValueError("At least one molecule group dataset key is required.")
    strata = discover_trainable_molecule_strata(zip_paths)
    groups = [
        {
            "dataset_key": key,
            "trajectory_count": 0,
            "strata": [],
        }
        for key in keys
    ]
    for row in sorted(strata, key=lambda item: (-int(item["trajectory_count"]), str(item["source_zip_name"]), str(item["stratum"]))):
        target_idx = min(range(len(groups)), key=lambda idx: (int(groups[idx]["trajectory_count"]), idx))
        member = {
            key: value
            for key, value in row.items()
            if key != "source_zip_path"
        }
        member["processed_dir"] = f"processed/{member['member_key']}"
        groups[target_idx]["strata"].append(member)
        groups[target_idx]["trajectory_count"] = int(groups[target_idx]["trajectory_count"]) + int(row["trajectory_count"])
    group_counts = [int(group["trajectory_count"]) for group in groups]
    total = int(sum(group_counts))
    balance = {
        "group_count": int(len(groups)),
        "total_trajectory_count": total,
        "group_trajectory_counts": group_counts,
        "max_group_trajectory_count": int(max(group_counts, default=0)),
        "min_group_trajectory_count": int(min(group_counts, default=0)),
        "max_min_delta": int(max(group_counts, default=0) - min(group_counts, default=0)),
        "group_trajectory_fractions": [float(count / max(1, total)) for count in group_counts],
    }
    return {
        "dataset_keys": list(keys),
        "source_zip_names": sorted({_source_zip_display_name(path) for path in zip_paths}),
        "trainable_stratum_count": int(len(strata)),
        "groups": groups,
        "balance": balance,
    }


def write_molecule_group_manifests(
    zip_paths: Sequence[str | Path],
    group_root: str | Path | None = None,
    *,
    dataset_keys: Sequence[str] = MOLECULE_GROUP_DATASET_KEYS,
) -> Dict[str, Any]:
    grouping = build_balanced_molecule_stratum_groups(zip_paths, dataset_keys=dataset_keys)
    root = default_molecule_group_root() if group_root is None else resolve_project_path(group_root)
    manifest_paths: Dict[str, str] = {}
    for group in grouping["groups"]:
        dataset_key = str(group["dataset_key"])
        manifest_path = root / dataset_key / "group_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_version": "molecule_3d_group_manifest_v1",
            "dataset_key": dataset_key,
            "benchmark_family": MOLECULE_BENCHMARK_FAMILY,
            "source": "local molecule trajectory zip files",
            "source_zip_names": sorted({str(row["source_zip_name"]) for row in group["strata"]}),
            "strata": group["strata"],
            "balance": {
                **dict(grouping["balance"]),
                "dataset_key": dataset_key,
                "trajectory_count": int(group["trajectory_count"]),
                "stratum_count": int(len(group["strata"])),
            },
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest_paths[dataset_key] = _project_display_path(manifest_path)
    return {
        "manifest_paths": manifest_paths,
        "grouping": grouping,
    }


def load_molecule_group_manifest(dataset_key: str, group_root: str | Path | None = None) -> Dict[str, Any]:
    manifest_path = default_molecule_group_manifest_path(dataset_key, group_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_key = _clean_path_token(dataset_key, label="dataset_key")
    if str(payload.get("dataset_key")) != expected_key:
        raise ValueError(f"Molecule group manifest {manifest_path} is for {payload.get('dataset_key')}, not {expected_key}.")
    for row in payload.get("strata", []):
        processed_dir = PurePosixPath(str(row.get("processed_dir", "")))
        if processed_dir.is_absolute() or ".." in processed_dir.parts:
            raise ValueError(f"Molecule group manifest has unsafe processed_dir={processed_dir}.")
        if Path(str(row.get("source_zip_name", ""))).name != str(row.get("source_zip_name", "")):
            raise ValueError("Molecule group manifest source_zip_name must be a file name, not a path.")
    return payload


def prepare_molecule_xyz_group_datasets(
    zip_paths: Sequence[str | Path],
    group_root: str | Path | None = None,
    *,
    dataset_keys: Sequence[str] = MOLECULE_GROUP_DATASET_KEYS,
    split_seed: int = DEFAULT_MOLECULE_SPLIT_SEED,
) -> Dict[str, Any]:
    zip_by_name = {_source_zip_display_name(path): Path(path).expanduser().resolve() for path in zip_paths}
    manifest_summary = write_molecule_group_manifests(zip_paths, group_root, dataset_keys=dataset_keys)
    root = default_molecule_group_root() if group_root is None else resolve_project_path(group_root)
    prepared: Dict[str, Any] = {}
    for dataset_key in manifest_summary["grouping"]["dataset_keys"]:
        manifest = load_molecule_group_manifest(str(dataset_key), root)
        rows: Dict[str, Any] = {}
        for member_idx, member in enumerate(manifest["strata"]):
            source_zip = zip_by_name[str(member["source_zip_name"])]
            processed_dir = root / str(dataset_key) / str(member["processed_dir"])
            rows[str(member["member_key"])] = prepare_molecule_xyz_zip(
                source_zip,
                processed_dir,
                dataset_key=str(dataset_key),
                stratum=str(member["stratum"]),
                split_seed=int(split_seed) + int(member_idx) + sum(ord(ch) for ch in str(dataset_key)),
                trainable=True,
            )
        prepared[str(dataset_key)] = {
            "dataset_key": str(dataset_key),
            "manifest_path": _project_display_path(default_molecule_group_manifest_path(str(dataset_key), root)),
            "prepared_strata": rows,
        }
    return {
        "manifests": manifest_summary["manifest_paths"],
        "prepared": prepared,
    }


def kabsch_rotation(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    moving_c = np.asarray(moving, dtype=np.float64)
    reference_c = np.asarray(reference, dtype=np.float64)
    if moving_c.shape != reference_c.shape or moving_c.ndim != 2 or moving_c.shape[1] != MOLECULE_TOKEN_DIM:
        raise ValueError(f"Expected [N,3] arrays, got moving={moving_c.shape}, reference={reference_c.shape}.")
    cov = moving_c.T @ reference_c
    u, _, vh = np.linalg.svd(cov, full_matrices=False)
    det = np.linalg.det(u @ vh)
    correction = np.eye(MOLECULE_TOKEN_DIM, dtype=np.float64)
    correction[-1, -1] = 1.0 if det >= 0 else -1.0
    return (u @ correction @ vh).astype(np.float32)


def align_window_to_reference(window: np.ndarray, reference_coords: np.ndarray, anchor_offset: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_f = np.asarray(window, dtype=np.float32)
    anchor = window_f[int(anchor_offset)]
    center = anchor.mean(axis=0, keepdims=True)
    anchor_centered = anchor - center
    reference = np.asarray(reference_coords, dtype=np.float32)
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    rotation = kabsch_rotation(anchor_centered, reference_centered)
    local = (window_f - center) @ rotation
    return local.astype(np.float32), center.reshape(MOLECULE_TOKEN_DIM).astype(np.float32), rotation.astype(np.float32)


def invert_aligned_coords(local: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (np.asarray(local, dtype=np.float32) @ np.asarray(rotation, dtype=np.float32).T) + np.asarray(center, dtype=np.float32)


def kabsch_aligned_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = np.asarray(a, dtype=np.float64) - np.asarray(a, dtype=np.float64).mean(axis=0, keepdims=True)
    b_centered = np.asarray(b, dtype=np.float64) - np.asarray(b, dtype=np.float64).mean(axis=0, keepdims=True)
    rot = kabsch_rotation(a_centered, b_centered).astype(np.float64)
    diff = a_centered @ rot - b_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _frame_quality_masks(coords: np.ndarray, segment_ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(coords.shape[0])
    duplicates = np.zeros(total, dtype=bool)
    transitions = np.zeros(total, dtype=bool)
    discontinuities = np.zeros(total, dtype=bool)
    rmsd = np.zeros(total, dtype=np.float32)
    starts = np.concatenate(([0], np.asarray(segment_ends[:-1], dtype=np.int64)))
    for start, end in zip(starts, segment_ends):
        for frame_idx in range(int(start) + 1, int(end)):
            prev_frame = coords[frame_idx - 1]
            cur_frame = coords[frame_idx]
            duplicates[frame_idx] = bool(np.array_equal(prev_frame, cur_frame))
            value = kabsch_aligned_rmsd(cur_frame, prev_frame)
            rmsd[frame_idx] = value
            transitions[frame_idx] = value > 0.1
            discontinuities[frame_idx] = value > 0.5
    return duplicates, transitions, discontinuities, rmsd


def _length_stratum(frame_count: int) -> str:
    frames = int(frame_count)
    if frames < 250:
        return "short_lt_250"
    if frames < 3000:
        return "medium_250_2999"
    return "long_gte_3000"


def _target_split_counts(n_items: int) -> Dict[str, int]:
    n = int(n_items)
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if n < 3:
        return {"train": max(1, n - 1), "val": 0, "test": 1 if n > 1 else 0}
    val = max(1, int(round(0.1 * n)))
    test = max(1, int(round(0.2 * n)))
    if val + test >= n:
        val = 1
        test = max(1, n - 2)
    train = n - val - test
    return {"train": int(train), "val": int(val), "test": int(test)}


def _deterministic_trajectory_split(
    frame_counts: Mapping[int, int],
    *,
    seed: int,
) -> Dict[str, Tuple[int, ...]]:
    ids = [int(value) for value in sorted(frame_counts)]
    counts = _target_split_counts(len(ids))
    targets = {"train": 0.7, "val": 0.1, "test": 0.2}
    rng = np.random.default_rng(int(seed))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    order = sorted(shuffled, key=lambda value: (int(frame_counts[int(value)]), value), reverse=True)
    assigned: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    assigned_frames = {"train": 0, "val": 0, "test": 0}
    total_frames = float(max(1, sum(int(frame_counts[int(value)]) for value in ids)))
    for trajectory_id in order:
        frame_count = int(frame_counts[int(trajectory_id)])
        available = [split for split in ("train", "val", "test") if len(assigned[split]) < counts[split]]
        if not available:
            available = ["train"]
        split = max(
            available,
            key=lambda name: (
                targets[name] - (assigned_frames[name] / total_frames),
                counts[name] - len(assigned[name]),
            ),
        )
        assigned[split].append(int(trajectory_id))
        assigned_frames[split] += frame_count
    return {split: tuple(sorted(values)) for split, values in assigned.items()}


def _length_strata_counts(ids: Sequence[int], frame_counts: Mapping[int, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for trajectory_id in ids:
        key = _length_stratum(int(frame_counts[int(trajectory_id)]))
        out[key] = out.get(key, 0) + 1
    return out


def _trajectory_split_stats(
    frame_counts: Mapping[int, int],
    split_trajectory_ids: Mapping[str, Sequence[int]],
    *,
    trajectory_keys: Sequence[str],
    iso_by_trajectory: Mapping[int, Optional[int]],
) -> Dict[str, Any]:
    total = int(sum(int(frame_counts[int(value)]) for values in split_trajectory_ids.values() for value in values))
    out: Dict[str, Any] = {
        "split_trajectory_ids": {
            split: [int(value) for value in values]
            for split, values in split_trajectory_ids.items()
        },
    }
    for split, ids_raw in split_trajectory_ids.items():
        ids = [int(value) for value in ids_raw]
        frames = int(sum(int(frame_counts[int(value)]) for value in ids))
        out[split] = {
            "trajectory_count": int(len(ids)),
            "frame_count": frames,
            "frame_fraction": float(frames / max(1, total)),
            "trajectory_ids": ids,
            "trajectory_keys": [str(trajectory_keys[int(value)]) for value in ids],
            "iso_ids": [
                int(iso_by_trajectory[int(value)])
                for value in ids
                if iso_by_trajectory.get(int(value)) is not None
            ],
            "length_strata": _length_strata_counts(ids, frame_counts),
        }
    return out


def _unicode_dtype(values: Sequence[str]) -> str:
    return f"U{max(1, max((len(str(value)) for value in values), default=1))}"


def _matches_stratum(name: str, stratum: str) -> bool:
    if not stratum:
        return True
    return _category_from_name(name).lower() == stratum.lower()


def prepare_molecule_xyz_zip(
    zip_path: str | Path,
    processed_dir: str | Path | None = None,
    *,
    dataset_key: str | None = None,
    stratum: str | None = None,
    split_seed: int = DEFAULT_MOLECULE_SPLIT_SEED,
    trainable: bool | None = None,
) -> Dict[str, Any]:
    key = _clean_path_token(dataset_key, label="dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    stratum_name = _clean_optional_stratum(stratum)
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing molecule trajectory zip: {zip_path}")
    out_dir = default_molecule_processed_dir(key, stratum_name) if processed_dir is None else resolve_project_path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_coords: List[np.ndarray] = []
    iso_ids: List[int] = []
    trajectory_ids: List[int] = []
    trajectory_keys: List[str] = []
    source_names_out: List[str] = []
    segment_ends: List[int] = []
    trajectory_frame_counts: Dict[int, int] = {}
    iso_by_trajectory: Dict[int, Optional[int]] = {}
    expected_symbols: Optional[Tuple[str, ...]] = None
    total = 0

    with ZipFile(zip_path) as zf:
        source_names = [name for name in _xyz_names(zf) if _matches_stratum(name, stratum_name)]
        source_names = sorted(source_names, key=_name_sort_key)
        if not source_names:
            raise ValueError(f"No XYZ trajectories found for dataset_key={key!r}, stratum={stratum_name!r}.")
        categories = sorted({_category_from_name(name) for name in source_names})
        for trajectory_idx, name in enumerate(source_names):
            symbols, coords = _read_xyz_trajectory_from_zip(
                zf,
                name,
                atom_count_expected=len(expected_symbols) if expected_symbols is not None else None,
            )
            if expected_symbols is None:
                expected_symbols = symbols
            elif symbols != expected_symbols:
                hint = " Use --stratum or --all_strata for mixed-shape molecule zips." if not stratum_name else ""
                raise ValueError(f"{name}: atom order differs from first selected trajectory.{hint}")
            all_coords.append(coords)
            n_frames = int(coords.shape[0])
            parsed_iso = _iso_id_from_name(name)
            iso_value = int(parsed_iso) if parsed_iso is not None else int(trajectory_idx)
            trajectory_frame_counts[int(trajectory_idx)] = n_frames
            iso_by_trajectory[int(trajectory_idx)] = int(parsed_iso) if parsed_iso is not None else None
            iso_ids.extend([iso_value] * n_frames)
            trajectory_ids.extend([trajectory_idx] * n_frames)
            trajectory_keys.append(_trajectory_key_from_name(name, trajectory_idx))
            source_names_out.append(name)
            total += n_frames
            segment_ends.append(total)

    coords_full = np.concatenate(all_coords, axis=0).astype(np.float32)
    segment_ends_arr = np.asarray(segment_ends, dtype=np.int64)
    duplicate_mask, transition_mask, discontinuity_mask, frame_rmsd = _frame_quality_masks(coords_full, segment_ends_arr)
    atom_symbols = tuple(expected_symbols or ())
    split_trajectory_ids = _deterministic_trajectory_split(trajectory_frame_counts, seed=int(split_seed))
    split_stats = _trajectory_split_stats(
        trajectory_frame_counts,
        split_trajectory_ids,
        trajectory_keys=trajectory_keys,
        iso_by_trajectory=iso_by_trajectory,
    )
    split_iso_ids = {
        split: [
            int(iso_by_trajectory[int(value)])
            for value in values
            if iso_by_trajectory.get(int(value)) is not None
        ]
        for split, values in split_trajectory_ids.items()
    }
    category = stratum_name or (categories[0] if len(categories) == 1 else "")
    trainable_value = bool(not category.startswith("Direct_")) if trainable is None else bool(trainable)

    np.savez_compressed(
        molecule_processed_npz_path(out_dir),
        coords=coords_full,
        atom_symbols=np.asarray(atom_symbols, dtype="U4"),
        iso_ids=np.asarray(iso_ids, dtype=np.int64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
        trajectory_keys=np.asarray(trajectory_keys, dtype=_unicode_dtype(trajectory_keys)),
        source_names=np.asarray(source_names_out, dtype=_unicode_dtype(source_names_out)),
        segment_ends=segment_ends_arr,
        duplicate_step_mask=duplicate_mask,
        transition_step_mask=transition_mask,
        discontinuity_step_mask=discontinuity_mask,
        frame_rmsd=frame_rmsd,
    )
    metadata: Dict[str, Any] = {
        "dataset_key": key,
        "stratum": stratum_name,
        "trainable": trainable_value,
        "benchmark_family": MOLECULE_BENCHMARK_FAMILY,
        "source_zip": _project_display_path(zip_path),
        "source_zip_name": zip_path.name,
        "source_xyz_count": int(len(source_names_out)),
        "source_total_frames": int(total),
        "xyz_count": int(len(source_names_out)),
        "total_frames": int(total),
        "atom_count": int(coords_full.shape[1]),
        "token_dim": MOLECULE_TOKEN_DIM,
        "context_atom_feature_dim": MOLECULE_CONTEXT_ATOM_FEATURE_DIM,
        "context_feature_dim": int(coords_full.shape[1]) * MOLECULE_CONTEXT_ATOM_FEATURE_DIM,
        "coord_dim": int(coords_full.shape[1]) * MOLECULE_TOKEN_DIM,
        "formula": _formula(atom_symbols),
        "atom_symbols": list(atom_symbols),
        "source_names": source_names_out,
        "categories": [category for category in categories if category],
        "trajectory_keys": trajectory_keys,
        "trajectory_ids": [int(value) for value in sorted(trajectory_frame_counts)],
        "trajectory_iso_ids": {
            str(key): (None if value is None else int(value))
            for key, value in sorted(iso_by_trajectory.items())
        },
        "trajectory_frame_counts": {str(key): int(value) for key, value in sorted(trajectory_frame_counts.items())},
        "split_seed": int(split_seed),
        "split_iso_ids": split_iso_ids,
        "split_trajectory_ids": {
            key: [int(value) for value in values]
            for key, values in split_trajectory_ids.items()
        },
        "split_stats": split_stats,
        "quality": {
            "duplicate_steps": int(duplicate_mask.sum()),
            "transition_steps_gt_0_1": int(transition_mask.sum()),
            "discontinuity_steps_gt_0_5": int(discontinuity_mask.sum()),
            "frame_rmsd_median": float(np.median(frame_rmsd[frame_rmsd > 0])) if np.any(frame_rmsd > 0) else 0.0,
            "frame_rmsd_p95": float(np.percentile(frame_rmsd[frame_rmsd > 0], 95)) if np.any(frame_rmsd > 0) else 0.0,
            "frame_rmsd_p99": float(np.percentile(frame_rmsd[frame_rmsd > 0], 99)) if np.any(frame_rmsd > 0) else 0.0,
            "frame_rmsd_max": float(frame_rmsd.max()) if len(frame_rmsd) else 0.0,
        },
    }
    molecule_processed_metadata_path(out_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def prepare_molecule_xyz_all_strata(
    zip_path: str | Path,
    processed_root: str | Path | None = None,
    *,
    dataset_key: str | None = None,
    include_pattern: str = "*",
    exclude_pattern: str = "",
    split_seed: int = DEFAULT_MOLECULE_SPLIT_SEED,
) -> Dict[str, Any]:
    key = _clean_path_token(dataset_key, label="dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    root = default_molecule_processed_dir(key, None) if processed_root is None else resolve_project_path(processed_root)
    strata = discover_molecule_xyz_strata(zip_path)
    summaries: Dict[str, Any] = {}
    for stratum_key, summary in strata.items():
        stratum_name = str(summary.get("stratum") or "")
        if not stratum_name:
            continue
        if include_pattern and not fnmatch.fnmatch(stratum_name, include_pattern):
            continue
        if exclude_pattern and fnmatch.fnmatch(stratum_name, exclude_pattern):
            continue
        summaries[stratum_name] = prepare_molecule_xyz_zip(
            zip_path,
            root / stratum_name,
            dataset_key=key,
            stratum=stratum_name,
            split_seed=int(split_seed) + sum(ord(ch) for ch in stratum_name),
            trainable=bool(summary.get("trainable", True)),
        )
    if not summaries:
        raise ValueError("No molecule strata matched the requested include/exclude patterns.")
    return {
        "dataset_key": key,
        "processed_root": _project_display_path(root),
        "strata": summaries,
    }


def _metadata_matches_request(
    metadata: Mapping[str, Any],
    *,
    dataset_key: str | None,
    stratum: str | None,
) -> bool:
    if dataset_key not in (None, ""):
        if str(metadata.get("dataset_key", "")) != _clean_path_token(dataset_key, label="dataset_key"):
            return False
    if stratum not in (None, ""):
        if str(metadata.get("stratum", "")).lower() != _clean_optional_stratum(stratum).lower():
            return False
    return True


def load_molecule_processed(
    processed_dir: str | Path | None = None,
    *,
    dataset_key: str | None = None,
    stratum: str | None = None,
) -> MoleculeProcessedData:
    npz_path = molecule_processed_npz_path(processed_dir, dataset_key=dataset_key, stratum=stratum)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing processed molecule dataset: {npz_path}")
    metadata_path = molecule_processed_metadata_path(processed_dir, dataset_key=dataset_key, stratum=stratum)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not _metadata_matches_request(metadata, dataset_key=dataset_key, stratum=stratum):
        raise ValueError(
            f"Processed molecule dataset at {npz_path} is for "
            f"{metadata.get('dataset_key')}/{metadata.get('stratum')}, not {dataset_key}/{stratum}."
        )
    payload = np.load(npz_path, allow_pickle=False)
    coords = payload["coords"].astype(np.float32)
    if coords.ndim != 3 or coords.shape[2] != MOLECULE_TOKEN_DIM:
        raise ValueError(f"Expected coords shape [frames, atoms, 3], got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError("Processed molecule coordinates contain non-finite values.")
    trajectory_ids = payload["trajectory_ids"].astype(np.int64)
    frame_counts = {
        int(trajectory_id): int(np.sum(trajectory_ids == int(trajectory_id)))
        for trajectory_id in sorted(set(int(value) for value in trajectory_ids.tolist()))
    }
    trajectory_keys = tuple(str(value) for value in payload["trajectory_keys"].tolist()) if "trajectory_keys" in payload.files else tuple()
    return MoleculeProcessedData(
        coords=coords,
        atom_symbols=tuple(str(value) for value in payload["atom_symbols"].tolist()),
        iso_ids=payload["iso_ids"].astype(np.int64),
        trajectory_ids=trajectory_ids,
        trajectory_keys=trajectory_keys,
        segment_ends=payload["segment_ends"].astype(np.int64),
        frame_counts=frame_counts,
        duplicate_step_mask=payload["duplicate_step_mask"].astype(bool),
        transition_step_mask=payload["transition_step_mask"].astype(bool),
        discontinuity_step_mask=payload["discontinuity_step_mask"].astype(bool),
        frame_rmsd=payload["frame_rmsd"].astype(np.float32),
        metadata=metadata,
    )


def ensure_molecule_processed(
    *,
    zip_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    prepare: bool = True,
    dataset_key: str | None = None,
    stratum: str | None = None,
) -> Dict[str, Any]:
    metadata_path = molecule_processed_metadata_path(processed_dir, dataset_key=dataset_key, stratum=stratum)
    npz_path = molecule_processed_npz_path(processed_dir, dataset_key=dataset_key, stratum=stratum)
    if metadata_path.exists() and npz_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _metadata_matches_request(metadata, dataset_key=dataset_key, stratum=stratum):
            return metadata
        if not prepare:
            raise ValueError(
                f"Existing processed molecule dataset does not match {dataset_key}/{stratum}: {npz_path}. "
                "Re-run with --prepare_data."
            )
    if not prepare:
        raise FileNotFoundError(f"Missing processed molecule dataset: {npz_path}. Re-run with --prepare_data.")
    raw_zip = default_molecule_raw_zip(dataset_key, stratum) if zip_path is None else zip_path
    return prepare_molecule_xyz_zip(
        raw_zip,
        processed_dir,
        dataset_key=dataset_key,
        stratum=stratum,
    )


def _segment_starts(segment_ends: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.asarray(segment_ends[:-1], dtype=np.int64)))


def _split_segment_indices(data: MoleculeProcessedData, split: str) -> List[Tuple[int, int, int]]:
    split_trajectory_ids = data.metadata.get("split_trajectory_ids")
    if not isinstance(split_trajectory_ids, Mapping) or str(split) not in split_trajectory_ids:
        raise ValueError(f"Processed molecule metadata is missing split_trajectory_ids[{split!r}].")
    selected_trajectory_ids = {int(value) for value in split_trajectory_ids[str(split)]}
    starts = _segment_starts(data.segment_ends)
    out: List[Tuple[int, int, int]] = []
    for start, end in zip(starts, data.segment_ends):
        trajectory_id = int(data.trajectory_ids[int(start)])
        if trajectory_id in selected_trajectory_ids:
            out.append((int(start), int(end), trajectory_id))
    if len(out) != len(selected_trajectory_ids):
        found = {value for _, _, value in out}
        raise ValueError(f"Split {split!r} missing trajectory ids: {sorted(selected_trajectory_ids - found)}")
    return out


def _window_step_slice(
    *,
    seg_start: int,
    seg_end: int,
    target_idx: int,
    history_len: int,
    future_horizon: int,
) -> slice:
    first_step = max(int(seg_start) + 1, int(target_idx) - int(history_len) + 1)
    last_step_exclusive = min(int(seg_end), int(target_idx) + int(future_horizon))
    return slice(first_step, last_step_exclusive)


def _window_has_mask(
    mask: np.ndarray,
    *,
    seg_start: int,
    seg_end: int,
    target_idx: int,
    history_len: int,
    future_horizon: int,
) -> bool:
    step_slice = _window_step_slice(
        seg_start=seg_start,
        seg_end=seg_end,
        target_idx=target_idx,
        history_len=history_len,
        future_horizon=future_horizon,
    )
    return bool(np.any(mask[step_slice]))


def _valid_targets_for_segments(
    data: MoleculeProcessedData,
    segments: Sequence[Tuple[int, int, int]],
    *,
    history_len: int,
    future_horizon: int,
    clean_windows: bool,
    exclude_duplicate_targets: bool,
    stride: int,
) -> np.ndarray:
    starts: List[np.ndarray] = []
    for seg_start, seg_end, _ in segments:
        first_target = int(seg_start) + int(history_len)
        last_target_exclusive = int(seg_end) - int(future_horizon) + 1
        if first_target >= last_target_exclusive:
            continue
        arr = np.arange(first_target, last_target_exclusive, max(1, int(stride)), dtype=np.int64)
        if clean_windows or exclude_duplicate_targets:
            clean = np.ones(len(arr), dtype=bool)
            if exclude_duplicate_targets:
                clean &= ~data.duplicate_step_mask[arr]
            if clean_windows:
                clean &= np.asarray(
                    [
                        not _window_has_mask(
                            data.discontinuity_step_mask,
                            seg_start=seg_start,
                            seg_end=seg_end,
                            target_idx=int(target_idx),
                            history_len=history_len,
                            future_horizon=future_horizon,
                        )
                        for target_idx in arr
                    ],
                    dtype=bool,
                )
            arr = arr[clean]
        starts.append(arr)
    if not starts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(starts)


def _atom_property(symbol: str, table: Mapping[str, float | int], *, label: str) -> float:
    if symbol not in table:
        raise ValueError(f"Unsupported atom symbol {symbol!r}; add its {label} before preprocessing.")
    return float(table[symbol])


def _atom_static_features(atom_symbols: Sequence[str]) -> np.ndarray:
    rows = []
    for symbol in atom_symbols:
        symbol_s = str(symbol)
        rows.append(
            [
                _atom_property(symbol_s, ATOM_NUMBER, label="atomic number") / 100.0,
                _atom_property(symbol_s, ATOM_MASS, label="atomic mass") / 12.011,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _feature_static_mask(atom_count: int) -> np.ndarray:
    mask = np.zeros(int(atom_count) * MOLECULE_CONTEXT_ATOM_FEATURE_DIM, dtype=bool)
    for atom_idx in range(int(atom_count)):
        base = atom_idx * MOLECULE_CONTEXT_ATOM_FEATURE_DIM
        mask[base + 9] = True
        mask[base + 10] = True
    return mask


def _context_features_from_local(local_context: np.ndarray, atom_static: np.ndarray) -> np.ndarray:
    pos = np.asarray(local_context, dtype=np.float32)
    vel = np.zeros_like(pos)
    if pos.shape[0] > 1:
        vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(pos)
    if pos.shape[0] > 2:
        acc[2:] = vel[2:] - vel[1:-1]
    static = np.broadcast_to(atom_static[None, :, :], (pos.shape[0], atom_static.shape[0], atom_static.shape[1]))
    features = np.concatenate([pos, vel, acc, static], axis=2)
    return features.reshape(pos.shape[0], -1).astype(np.float32)


def _target_residuals_from_local(local: np.ndarray, history_len: int, future_horizon: int) -> np.ndarray:
    current = local[int(history_len) - 1]
    future = local[int(history_len) : int(history_len) + int(future_horizon)]
    return (future - current[None, :, :]).reshape(int(future_horizon), -1).astype(np.float32)


def molecule_stats_from_mapping(payload: Mapping[str, Any]) -> MoleculeStats:
    required = ("target_mean", "target_std", "context_mean", "context_std", "reference_coords")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing molecule normalization stats: {missing}")
    stats = MoleculeStats(
        target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
        target_std=np.asarray(payload["target_std"], dtype=np.float32),
        context_mean=np.asarray(payload["context_mean"], dtype=np.float32),
        context_std=np.asarray(payload["context_std"], dtype=np.float32),
        reference_coords=np.asarray(payload["reference_coords"], dtype=np.float32),
    )
    if stats.target_mean.shape != stats.target_std.shape or stats.target_mean.ndim != 1:
        raise ValueError(
            "Expected one-dimensional matching target stats, got "
            f"mean={stats.target_mean.shape}, std={stats.target_std.shape}."
        )
    if stats.context_mean.shape != stats.context_std.shape or stats.context_mean.ndim != 1:
        raise ValueError(
            "Expected one-dimensional matching context stats, got "
            f"mean={stats.context_mean.shape}, std={stats.context_std.shape}."
        )
    if stats.reference_coords.ndim != 2 or stats.reference_coords.shape[1] != MOLECULE_TOKEN_DIM:
        raise ValueError(
            f"Expected reference_coords shape [N,{MOLECULE_TOKEN_DIM}], got {stats.reference_coords.shape}."
        )
    atom_count = int(stats.reference_coords.shape[0])
    if stats.target_mean.shape[0] % (atom_count * MOLECULE_TOKEN_DIM) != 0:
        raise ValueError("Target stats shape is incompatible with reference atom count.")
    if stats.context_mean.shape[0] != atom_count * MOLECULE_CONTEXT_ATOM_FEATURE_DIM:
        raise ValueError("Context stats shape is incompatible with reference atom count.")
    if not all(
        np.isfinite(arr).all()
        for arr in (stats.target_mean, stats.target_std, stats.context_mean, stats.context_std, stats.reference_coords)
    ):
        raise ValueError("Molecule normalization stats contain non-finite values.")
    if np.any(stats.target_std <= 0.0) or np.any(stats.context_std <= 0.0):
        raise ValueError("Molecule normalization std values must be positive.")
    return stats


def _stats_from_training_windows(
    data: MoleculeProcessedData,
    train_starts: np.ndarray,
    *,
    history_len: int,
    future_horizon: int,
) -> MoleculeStats:
    reference_coords = data.coords[0].astype(np.float32)
    atom_static = _atom_static_features(data.atom_symbols)
    coord_dim = int(data.coord_dim)
    context_feature_dim = int(data.context_feature_dim)
    target_sum = np.zeros(coord_dim, dtype=np.float64)
    target_sumsq = np.zeros(coord_dim, dtype=np.float64)
    context_sum = np.zeros(context_feature_dim, dtype=np.float64)
    context_sumsq = np.zeros(context_feature_dim, dtype=np.float64)
    target_count = 0
    context_count = 0
    for target_idx in train_starts:
        window = data.coords[int(target_idx) - int(history_len) : int(target_idx) + int(future_horizon)]
        local, _, _ = align_window_to_reference(window, reference_coords, anchor_offset=int(history_len) - 1)
        target = _target_residuals_from_local(local, history_len, future_horizon)
        context = _context_features_from_local(local[: int(history_len)], atom_static)
        target_sum += target.reshape(-1, coord_dim).sum(axis=0)
        target_sumsq += np.square(target.reshape(-1, coord_dim), dtype=np.float64).sum(axis=0)
        context_sum += context.sum(axis=0)
        context_sumsq += np.square(context, dtype=np.float64).sum(axis=0)
        target_count += int(target.shape[0])
        context_count += int(context.shape[0])
    if target_count <= 0 or context_count <= 0:
        raise ValueError("Cannot fit molecule statistics from empty training windows.")
    target_mean = target_sum / float(target_count)
    target_var = np.maximum(target_sumsq / float(target_count) - target_mean * target_mean, 1e-8)
    context_mean = context_sum / float(context_count)
    context_var = np.maximum(context_sumsq / float(context_count) - context_mean * context_mean, 1e-8)
    context_std = np.sqrt(context_var)
    context_mean = context_mean.astype(np.float32)
    static_mask = _feature_static_mask(data.atom_count)
    context_mean[static_mask] = 0.0
    context_std = context_std.astype(np.float32)
    context_std[static_mask] = 1.0
    return MoleculeStats(
        target_mean=target_mean.astype(np.float32),
        target_std=np.sqrt(target_var).astype(np.float32),
        context_mean=context_mean,
        context_std=context_std,
        reference_coords=reference_coords,
    )


def _target_exclusion_counts(
    data: MoleculeProcessedData,
    segments: Sequence[Tuple[int, int, int]],
    *,
    history_len: int,
    future_horizon: int,
) -> Dict[str, int]:
    total_candidates = 0
    duplicate_target = 0
    discontinuity_target = 0
    discontinuity_window = 0
    transition_target = 0
    transition_window = 0
    for seg_start, seg_end, _ in segments:
        first_target = int(seg_start) + int(history_len)
        last_target_exclusive = int(seg_end) - int(future_horizon) + 1
        if first_target >= last_target_exclusive:
            continue
        arr = np.arange(first_target, last_target_exclusive, dtype=np.int64)
        total_candidates += int(len(arr))
        duplicate_target += int(np.sum(data.duplicate_step_mask[arr]))
        discontinuity_target += int(np.sum(data.discontinuity_step_mask[arr]))
        transition_target += int(np.sum(data.transition_step_mask[arr]))
        for target_idx in arr:
            if _window_has_mask(
                data.discontinuity_step_mask,
                seg_start=seg_start,
                seg_end=seg_end,
                target_idx=int(target_idx),
                history_len=history_len,
                future_horizon=future_horizon,
            ):
                discontinuity_window += 1
            if _window_has_mask(
                data.transition_step_mask,
                seg_start=seg_start,
                seg_end=seg_end,
                target_idx=int(target_idx),
                history_len=history_len,
                future_horizon=future_horizon,
            ):
                transition_window += 1
    return {
        "candidate_targets": int(total_candidates),
        "duplicate_target_excluded": int(duplicate_target),
        "discontinuity_target_excluded": int(discontinuity_target),
        "discontinuity_window_excluded": int(discontinuity_window),
        "transition_target_flagged": int(transition_target),
        "transition_window_flagged": int(transition_window),
    }


class MoleculeWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: MoleculeProcessedData,
        *,
        split: str,
        start_indices: np.ndarray,
        history_len: int,
        future_horizon: int,
        stats: MoleculeStats,
    ):
        super().__init__()
        self.data = data
        self.split = str(split)
        self.start_indices = np.asarray(start_indices, dtype=np.int64)
        self.H = int(history_len)
        self.future_horizon = int(future_horizon)
        self.horizon = int(future_horizon)
        self.stats = stats
        self.atom_static = _atom_static_features(data.atom_symbols)
        self.cond = None
        self.cond_mean = None
        self.cond_std = None
        self.params_mean = stats.target_mean
        self.params_std = stats.target_std
        self.dataset_kind = data.dataset_key
        self.dataset_metadata = dict(data.metadata)
        self.segment_ends = data.segment_ends
        self.sampler_replacement = self.split == "train"
        self.sampler_num_samples = int(len(self.start_indices)) if self.sampler_replacement else None
        self.sampler_weights = None
        if self.sampler_replacement and len(self.start_indices) > 0:
            group_for_start = data.trajectory_ids[self.start_indices]
            counts = {int(group_id): int(np.sum(group_for_start == group_id)) for group_id in np.unique(group_for_start)}
            self.sampler_weights = np.asarray(
                [1.0 / float(max(1, counts[int(group_id)])) for group_id in group_for_start],
                dtype=np.float64,
            )

    def __len__(self) -> int:
        return int(len(self.start_indices))

    @property
    def context_feature_dim(self) -> int:
        return int(self.data.context_feature_dim)

    @property
    def snapshot_dim(self) -> int:
        return int(self.data.coord_dim)

    def segment_end_for_t(self, t: int | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.int64)
        idx = np.searchsorted(self.data.segment_ends, t_arr, side="right")
        return self.data.segment_ends[idx]

    def segment_start_for_t(self, t: int | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.int64)
        idx = np.searchsorted(self.data.segment_ends, t_arr, side="right")
        starts = _segment_starts(self.data.segment_ends)
        return starts[idx]

    def _raw_eval_arrays(self, target_idx: int) -> Dict[str, np.ndarray]:
        window = self.data.coords[int(target_idx) - self.H : int(target_idx) + self.future_horizon]
        local, center, rotation = align_window_to_reference(
            window,
            self.stats.reference_coords,
            anchor_offset=self.H - 1,
        )
        context = _context_features_from_local(local[: self.H], self.atom_static)
        target = _target_residuals_from_local(local, self.H, self.future_horizon)
        current = local[self.H - 1]
        future_coords = local[self.H : self.H + self.future_horizon]
        return {
            "context": context,
            "target_residuals": target,
            "history_coords": local[: self.H].astype(np.float32),
            "current_coords": current.astype(np.float32),
            "future_coords": future_coords.astype(np.float32),
            "center": center.astype(np.float32),
            "rotation": rotation.astype(np.float32),
        }

    def eval_item(self, idx: int) -> Dict[str, Any]:
        target_idx = int(self.start_indices[int(idx)])
        arrays = self._raw_eval_arrays(target_idx)
        seg_start = int(self.segment_start_for_t(target_idx))
        seg_end = int(self.segment_end_for_t(target_idx))
        transition_window = _window_has_mask(
            self.data.transition_step_mask,
            seg_start=seg_start,
            seg_end=seg_end,
            target_idx=target_idx,
            history_len=self.H,
            future_horizon=self.future_horizon,
        )
        discontinuity_window = _window_has_mask(
            self.data.discontinuity_step_mask,
            seg_start=seg_start,
            seg_end=seg_end,
            target_idx=target_idx,
            history_len=self.H,
            future_horizon=self.future_horizon,
        )
        trajectory_id = int(self.data.trajectory_ids[target_idx])
        return {
            **arrays,
            "target_idx": target_idx,
            "iso_id": int(self.data.iso_ids[target_idx]),
            "trajectory_id": trajectory_id,
            "trajectory_key": self.data.trajectory_keys[trajectory_id] if trajectory_id < len(self.data.trajectory_keys) else str(trajectory_id),
            "transition": bool(self.data.transition_step_mask[target_idx]),
            "transition_window": bool(transition_window),
            "discontinuity": bool(self.data.discontinuity_step_mask[target_idx]),
            "discontinuity_window": bool(discontinuity_window),
            "duplicate": bool(self.data.duplicate_step_mask[target_idx]),
            "frame_rmsd": float(self.data.frame_rmsd[target_idx]),
        }

    def __getitem__(self, idx: int):
        target_idx = int(self.start_indices[int(idx)])
        arrays = self._raw_eval_arrays(target_idx)
        hist = (arrays["context"] - self.stats.context_mean[None, :]) / self.stats.context_std[None, :]
        target = (arrays["target_residuals"] - self.stats.target_mean[None, :]) / self.stats.target_std[None, :]
        tgt_t = torch.from_numpy(target[0].astype(np.float32))
        fut_t = torch.from_numpy(target[1:].astype(np.float32))
        seg_start = int(self.segment_start_for_t(target_idx))
        seg_end = int(self.segment_end_for_t(target_idx))
        trajectory_id = int(self.data.trajectory_ids[target_idx])
        meta = {
            "t": target_idx,
            "t_global": target_idx,
            "target_t": target_idx,
            "history_start": int(target_idx) - self.H,
            "history_stop": target_idx,
            "target_stop": int(target_idx) + self.future_horizon,
            "segment_end": seg_end,
            "iso_id": int(self.data.iso_ids[target_idx]),
            "trajectory_id": trajectory_id,
            "trajectory_key": self.data.trajectory_keys[trajectory_id] if trajectory_id < len(self.data.trajectory_keys) else str(trajectory_id),
            "transition": bool(self.data.transition_step_mask[target_idx]),
            "discontinuity": bool(self.data.discontinuity_step_mask[target_idx]),
            "transition_window": bool(
                _window_has_mask(
                    self.data.transition_step_mask,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    target_idx=target_idx,
                    history_len=self.H,
                    future_horizon=self.future_horizon,
                )
            ),
            "discontinuity_window": bool(
                _window_has_mask(
                    self.data.discontinuity_step_mask,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    target_idx=target_idx,
                    history_len=self.H,
                    future_horizon=self.future_horizon,
                )
            ),
        }
        if self.future_horizon <= 1:
            return torch.from_numpy(hist.astype(np.float32)), tgt_t, meta
        return torch.from_numpy(hist.astype(np.float32)), tgt_t, fut_t, meta

    def denormalize_target(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        prefix = (1,) * (arr.ndim - 1)
        return arr * self.stats.target_std.reshape(prefix + (-1,)) + self.stats.target_mean.reshape(prefix + (-1,))


def build_molecule_dataset_splits(
    *,
    processed_dir: str | Path | None,
    cfg: OTFlowConfig,
    history_len: int,
    future_horizon: int,
    stride_train: int = 1,
    stride_eval: int = 1,
    stats: MoleculeStats | Mapping[str, Any] | None = None,
    dataset_key: str | None = None,
    stratum: str | None = None,
) -> Dict[str, Any]:
    del cfg
    data = load_molecule_processed(processed_dir, dataset_key=dataset_key, stratum=stratum)
    train_segments = _split_segment_indices(data, "train")
    val_segments = _split_segment_indices(data, "val")
    test_segments = _split_segment_indices(data, "test")
    train_stat_starts = _valid_targets_for_segments(
        data,
        train_segments,
        history_len=history_len,
        future_horizon=future_horizon,
        clean_windows=True,
        exclude_duplicate_targets=True,
        stride=1,
    )
    train_starts = _valid_targets_for_segments(
        data,
        train_segments,
        history_len=history_len,
        future_horizon=future_horizon,
        clean_windows=True,
        exclude_duplicate_targets=True,
        stride=stride_train,
    )
    val_starts = _valid_targets_for_segments(
        data,
        val_segments,
        history_len=history_len,
        future_horizon=future_horizon,
        clean_windows=False,
        exclude_duplicate_targets=False,
        stride=stride_eval,
    )
    val_clean_starts = _valid_targets_for_segments(
        data,
        val_segments,
        history_len=history_len,
        future_horizon=future_horizon,
        clean_windows=True,
        exclude_duplicate_targets=True,
        stride=stride_eval,
    )
    test_starts = _valid_targets_for_segments(
        data,
        test_segments,
        history_len=history_len,
        future_horizon=future_horizon,
        clean_windows=False,
        exclude_duplicate_targets=False,
        stride=stride_eval,
    )
    test_clean_starts = _valid_targets_for_segments(
        data,
        test_segments,
        history_len=history_len,
        future_horizon=future_horizon,
        clean_windows=True,
        exclude_duplicate_targets=True,
        stride=stride_eval,
    )
    if stats is None:
        fitted_stats = _stats_from_training_windows(
            data,
            train_stat_starts,
            history_len=history_len,
            future_horizon=future_horizon,
        )
        stats_source = "train_clean_windows_stride_1"
    elif isinstance(stats, MoleculeStats):
        fitted_stats = stats
        stats_source = "checkpoint"
    else:
        fitted_stats = molecule_stats_from_mapping(stats)
        stats_source = "checkpoint"
    datasets = {
        "train": MoleculeWindowDataset(data, split="train", start_indices=train_starts, history_len=history_len, future_horizon=future_horizon, stats=fitted_stats),
        "val": MoleculeWindowDataset(data, split="val", start_indices=val_starts, history_len=history_len, future_horizon=future_horizon, stats=fitted_stats),
        "val_clean": MoleculeWindowDataset(data, split="val_clean", start_indices=val_clean_starts, history_len=history_len, future_horizon=future_horizon, stats=fitted_stats),
        "test": MoleculeWindowDataset(data, split="test", start_indices=test_starts, history_len=history_len, future_horizon=future_horizon, stats=fitted_stats),
        "test_clean": MoleculeWindowDataset(data, split="test_clean", start_indices=test_clean_starts, history_len=history_len, future_horizon=future_horizon, stats=fitted_stats),
    }
    split_stats = dict(data.metadata.get("split_stats", {}))
    train_filter_counts = _target_exclusion_counts(data, train_segments, history_len=history_len, future_horizon=future_horizon)
    val_filter_counts = _target_exclusion_counts(data, val_segments, history_len=history_len, future_horizon=future_horizon)
    test_filter_counts = _target_exclusion_counts(data, test_segments, history_len=history_len, future_horizon=future_horizon)
    split_stats.update(
        {
            "history_len": int(history_len),
            "future_horizon": int(future_horizon),
            "dataset_key": data.dataset_key,
            "stratum": data.stratum,
            "atom_count": int(data.atom_count),
            "context_feature_dim": int(data.context_feature_dim),
            "snapshot_dim": int(data.coord_dim),
            "normalization_source": stats_source,
            "n_train_stat_examples": int(len(train_stat_starts)),
            "n_train_examples": int(len(train_starts)),
            "n_val_examples": int(len(val_starts)),
            "n_val_clean_examples": int(len(val_clean_starts)),
            "n_test_examples": int(len(test_starts)),
            "n_test_clean_examples": int(len(test_clean_starts)),
            "filter_counts": {
                "train": train_filter_counts,
                "val": val_filter_counts,
                "test": test_filter_counts,
            },
            "stats": fitted_stats.to_dict(),
        }
    )
    return {**datasets, "stats": split_stats, "data": data}


def configure_molecule_otflow(
    cfg: OTFlowConfig,
    *,
    history_len: int,
    future_horizon: int,
    rollout_mode: str,
    atom_count: int,
    context_feature_dim: int | None = None,
) -> OTFlowConfig:
    levels = int(atom_count)
    if levels <= 0:
        raise ValueError(f"atom_count must be positive, got {atom_count!r}.")
    context_dim = (
        levels * MOLECULE_CONTEXT_ATOM_FEATURE_DIM
        if context_feature_dim is None
        else int(context_feature_dim)
    )
    cfg.apply_overrides(
        levels=levels,
        token_dim=MOLECULE_TOKEN_DIM,
        context_feature_dim=context_dim,
        history_len=int(history_len),
        rollout_mode=str(rollout_mode),
        future_block_len=int(future_horizon),
        use_time_features=False,
        use_time_gaps=False,
    )
    return cfg


def build_molecule_group_dataset_splits(
    *,
    dataset_key: str,
    group_root: str | Path | None = None,
    cfg: OTFlowConfig | None = None,
    history_len: int = 2,
    future_horizon: int = 1,
    rollout_mode: str = "autoregressive",
    stride_train: int = 1,
    stride_eval: int = 1,
) -> Dict[str, Any]:
    manifest = load_molecule_group_manifest(dataset_key, group_root)
    root = default_molecule_group_root() if group_root is None else resolve_project_path(group_root)
    base_cfg = OTFlowConfig() if cfg is None else cfg
    strata: Dict[str, Any] = {}
    aggregate: Dict[str, Any] = {
        "dataset_key": str(dataset_key),
        "benchmark_family": MOLECULE_BENCHMARK_FAMILY,
        "stratum_count": 0,
        "trajectory_count": 0,
        "atom_counts": {},
        "formulas": {},
    }
    for member in manifest.get("strata", []):
        member_key = str(member["member_key"])
        processed_dir = root / str(dataset_key) / str(member["processed_dir"])
        atom_count = int(member["atom_count"])
        member_cfg = configure_molecule_otflow(
            copy.deepcopy(base_cfg),
            history_len=int(history_len),
            future_horizon=int(future_horizon),
            rollout_mode=str(rollout_mode),
            atom_count=atom_count,
        )
        splits = build_molecule_dataset_splits(
            processed_dir=processed_dir,
            cfg=member_cfg,
            history_len=int(history_len),
            future_horizon=int(future_horizon),
            stride_train=int(stride_train),
            stride_eval=int(stride_eval),
            dataset_key=str(dataset_key),
            stratum=str(member["stratum"]),
        )
        strata[member_key] = {
            "member": dict(member),
            "cfg": member_cfg,
            "splits": splits,
        }
        aggregate["stratum_count"] = int(aggregate["stratum_count"]) + 1
        aggregate["trajectory_count"] = int(aggregate["trajectory_count"]) + int(member.get("trajectory_count", member.get("xyz_count", 0)))
        aggregate["atom_counts"][member_key] = atom_count
        aggregate["formulas"][member_key] = str(member.get("formula", ""))
    return {
        "dataset_key": str(dataset_key),
        "manifest": manifest,
        "strata": strata,
        "stats": aggregate,
    }


__all__ = [
    "ATOM_COVALENT_RADIUS",
    "DEFAULT_MOLECULE_DATASET_KEY",
    "MOLECULE_GROUP_DATASET_KEYS",
    "DEFAULT_MOLECULE_SPLIT_SEED",
    "MOLECULE_BENCHMARK_FAMILY",
    "MOLECULE_CONTEXT_ATOM_FEATURE_DIM",
    "MOLECULE_TOKEN_DIM",
    "MoleculeProcessedData",
    "MoleculeStats",
    "MoleculeWindowDataset",
    "align_window_to_reference",
    "build_balanced_molecule_stratum_groups",
    "build_molecule_dataset_splits",
    "build_molecule_group_dataset_splits",
    "configure_molecule_otflow",
    "default_molecule_group_manifest_path",
    "default_molecule_group_root",
    "default_molecule_processed_dir",
    "default_molecule_raw_zip",
    "discover_trainable_molecule_strata",
    "discover_molecule_xyz_strata",
    "ensure_molecule_processed",
    "invert_aligned_coords",
    "kabsch_aligned_rmsd",
    "kabsch_rotation",
    "load_molecule_group_manifest",
    "load_molecule_processed",
    "molecule_processed_metadata_path",
    "molecule_processed_npz_path",
    "molecule_stats_from_mapping",
    "prepare_molecule_xyz_all_strata",
    "prepare_molecule_xyz_group_datasets",
    "prepare_molecule_xyz_zip",
    "write_molecule_group_manifests",
]
